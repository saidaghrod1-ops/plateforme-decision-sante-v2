"""
Couche ML — calibration par PINN inverse (cf. slide 11).

Identifie simultanément la trajectoire (S,E,I,R)(t) et les paramètres
(beta, gamma, sigma), paramétrés via softplus pour rester positifs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ml.network import MLP
from ml import losses


@dataclass
class TrainConfig:
    epochs: int = 8000
    lr: float = 1e-3
    n_collocation: int = 256
    w_data: float = 1.0
    w_phys: float = 1.0
    w_ic: float = 10.0
    hidden: int = 64
    depth: int = 4
    seed: int = 0


class InversePINN:
    def __init__(self, N: float, t_max: float, cfg: TrainConfig | None = None):
        self.cfg = cfg or TrainConfig()
        torch.manual_seed(self.cfg.seed)
        self.N, self.t_max = float(N), float(t_max)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = MLP(1, 4, self.cfg.hidden, self.cfg.depth).to(self.device)
        self._raw = torch.nn.Parameter(torch.tensor([-1.0, -2.3, -1.7], device=self.device))

    def _params(self):
        p = torch.nn.functional.softplus(self._raw)
        return p[0], p[1], p[2]

    def _scale(self, t):
        return t / self.t_max

    def fit(self, t_obs, y_obs, y0):
        cfg = self.cfg
        t_obs_t = torch.tensor(t_obs, dtype=torch.float32, device=self.device).view(-1, 1)
        y_obs_t = torch.tensor(y_obs, dtype=torch.float32, device=self.device)
        y0_t = torch.tensor(y0, dtype=torch.float32, device=self.device).view(1, 4)
        opt = torch.optim.Adam(list(self.net.parameters()) + [self._raw], lr=cfg.lr)

        loss = torch.tensor(0.0)
        for _ in range(cfg.epochs):
            opt.zero_grad()
            l_data = losses.loss_data(self.net(self._scale(t_obs_t)), y_obs_t)

            t_col = torch.rand(cfg.n_collocation, 1, device=self.device) * self.t_max
            t_col.requires_grad_(True)
            beta, gamma, sigma = self._params()
            res = losses.seir_residual(t_col, self.net(self._scale(t_col)), beta, sigma, gamma, self.N)
            l_phys = losses.loss_physics(res)

            t0 = torch.zeros(1, 1, device=self.device)
            l_ic = losses.loss_initial(self.net(self._scale(t0)), y0_t)

            loss = cfg.w_data * l_data + cfg.w_phys * l_phys + cfg.w_ic * l_ic
            loss.backward()
            opt.step()

        beta, gamma, sigma = (float(v) for v in self._params())
        return {"beta": beta, "gamma": gamma, "sigma": sigma, "final_loss": float(loss.item())}
