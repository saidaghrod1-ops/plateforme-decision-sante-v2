"""
Couche MOTEURS — solveur HJB-PINN (cf. slides 16-19).

Approche directement la fonction de valeur V(x,t) en résolvant l'EDP HJB par un
réseau (sans grille -> scalable en dimension 4+1). Le contrôle en boucle fermée
u*(x,t) est extrait par minimisation analytique de l'Hamiltonien (coût quadratique
en u). Expose l'interface Controller commune.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from domain.seir import SEIRParams
from ml.network import ValueNetwork
from optimization.base import Controller


@dataclass
class HJBConfig:
    # Coût en compartiments NORMALISÉS (I/N) -> V et résiduelle ~O(1) (réseau bien
    # conditionné) au lieu de ~10^12 en unités brutes. B,C faibles -> politique active.
    A: float = 1.0      # poids du coût sanitaire (sur la fraction d'infectés I/N)
    B: float = 0.02     # coût de la vaccination u1
    C: float = 0.02     # coût du confinement u2
    epochs: int = 6000
    lr: float = 1e-3
    n_collocation: int = 1024
    hidden: int = 96
    depth: int = 5
    horizon: float = 180.0
    seed: int = 0


class HJBController(Controller):
    name = "hjb-pinn"

    def __init__(self, params: SEIRParams, cfg: HJBConfig | None = None):
        self.p = params
        self.cfg = cfg or HJBConfig()
        torch.manual_seed(self.cfg.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.V = ValueNetwork(self.cfg.hidden, self.cfg.depth).to(self.device)
        self.scale = torch.tensor([params.N] * 4, dtype=torch.float32, device=self.device)
        self.T = self.cfg.horizon

    def _drift(self, x, u1, u2):
        S, E, I, R = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        new_inf = self.p.beta * (1 - u2) * S * I / self.p.N
        return torch.cat(
            [-new_inf - u1 * S, new_inf - self.p.sigma * E,
             self.p.sigma * E - self.p.gamma * I, self.p.gamma * I + u1 * S], dim=1)

    def _optimal_u(self, x, dVdx):
        S, I = x[:, 0:1], x[:, 2:3]
        z = torch.zeros_like(S)
        dfu1 = torch.cat([-S, z, z, S], dim=1)
        bSI = self.p.beta * S * I / self.p.N
        dfu2 = torch.cat([bSI, -bSI, z, z], dim=1)
        u1 = torch.clamp(-torch.sum(dVdx * dfu1, 1, keepdim=True) / (2 * self.cfg.B), 0, 1)
        u2 = torch.clamp(-torch.sum(dVdx * dfu2, 1, keepdim=True) / (2 * self.cfg.C), 0, 1)
        return u1, u2

    def _sample(self, n):
        w = torch.rand(n, 4, device=self.device)
        w = w / w.sum(1, keepdim=True) * self.p.N
        return w, torch.rand(n, 1, device=self.device) * self.T

    def fit(self) -> dict:
        opt = torch.optim.Adam(self.V.parameters(), lr=self.cfg.lr)
        loss = torch.tensor(0.0)
        for _ in range(self.cfg.epochs):
            opt.zero_grad()
            x, t = self._sample(self.cfg.n_collocation)
            x.requires_grad_(True)
            t.requires_grad_(True)
            V = self.V(x / self.scale, t / self.T)
            dVdt = torch.autograd.grad(V.sum(), t, create_graph=True)[0]
            dVdx = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
            u1, u2 = self._optimal_u(x, dVdx)
            L = self.cfg.A * x[:, 2:3] / self.p.N + self.cfg.B * u1 ** 2 + self.cfg.C * u2 ** 2
            H = L + torch.sum(dVdx * self._drift(x, u1, u2), 1, keepdim=True)
            loss_pde = torch.mean((dVdt + H) ** 2)

            xT, _ = self._sample(self.cfg.n_collocation // 4)
            tT = torch.full((xT.shape[0], 1), self.T, device=self.device)
            loss_term = torch.mean((self.V(xT / self.scale, tT / self.T) - self.cfg.A * xT[:, 2:3] / self.p.N) ** 2)
            loss = loss_pde + 10.0 * loss_term
            loss.backward()
            opt.step()
        return {"final_loss": float(loss.item())}

    def policy(self):
        def _p(t, x_np):
            x = torch.tensor(x_np, dtype=torch.float32, device=self.device).view(1, 4)
            x.requires_grad_(True)
            tt = torch.tensor([[t]], dtype=torch.float32, device=self.device)
            V = self.V(x / self.scale, tt / self.T)
            dVdx = torch.autograd.grad(V.sum(), x)[0]
            u1, u2 = self._optimal_u(x, dVdx)
            return float(u1.item()), float(u2.item())

        return _p

    # --- Persistance (entraîner une fois, recharger ensuite) ---
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "V": self.V.state_dict(),
                "T": self.T,
                "params": {"beta": self.p.beta, "sigma": self.p.sigma,
                           "gamma": self.p.gamma, "N": self.p.N},
                "cfg": asdict(self.cfg),
            },
            path,
        )

    def load(self, path: str | Path) -> "HJBController":
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.V.load_state_dict(ckpt["V"])
        self.T = ckpt.get("T", self.T)
        return self
