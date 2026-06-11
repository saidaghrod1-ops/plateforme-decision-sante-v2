"""
Couche MOTEURS — Contrôle optimal par MÉTHODE DIRECTE (cf. slide 8).

Discrétise le contrôle u(t) = (u1, u2) sur une grille temporelle, simule le SEIR,
et minimise la fonctionnelle de coût
    J = ∫ [A·I(t) + B·u1²(t) + C·u2²(t)] dt
sous contraintes 0 <= u <= 1 via un solveur NLP (L-BFGS-B).

Produit un contrôle optimal EN BOUCLE OUVERTE — baseline « directe » classique
face au LQR (boucle fermée) et au HJB-PINN dans le module de benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from domain.seir import SEIRParams, simulate
from optimization.base import Controller


@dataclass
class PMPConfig:
    A: float = 1.0
    B: float = 5e3
    C: float = 5e3
    n_intervals: int = 18      # nb de paliers de contrôle (piecewise-constant)
    horizon: float = 180.0
    maxiter: int = 60


class PMPController(Controller):
    name = "pmp-direct"

    def __init__(self, params: SEIRParams, cfg: PMPConfig | None = None):
        self.p = params
        self.cfg = cfg or PMPConfig()
        self._u_grid: np.ndarray | None = None
        self._knots = np.linspace(0.0, self.cfg.horizon, self.cfg.n_intervals + 1)

    def _piecewise(self, u_flat: np.ndarray):
        """Reconstruit la politique paliers à partir du vecteur d'optimisation."""
        u = u_flat.reshape(self.cfg.n_intervals, 2)

        def _p(t, x):
            k = min(int(t / self.cfg.horizon * self.cfg.n_intervals), self.cfg.n_intervals - 1)
            return float(u[k, 0]), float(u[k, 1])

        return _p

    def _cost(self, u_flat: np.ndarray, x0: np.ndarray) -> float:
        policy = self._piecewise(u_flat)
        traj = simulate(self.p, x0, (0, self.cfg.horizon),
                        int(self.cfg.horizon) + 1, policy=policy)
        t = np.array(traj["t"])
        I = np.array(traj["I"])
        u = u_flat.reshape(self.cfg.n_intervals, 2)
        # coût de contrôle évalué aux paliers, normalisé sur l'horizon
        ctrl_cost = (self.cfg.B * u[:, 0] ** 2 + self.cfg.C * u[:, 1] ** 2).mean() * self.cfg.horizon
        health_cost = self.cfg.A * np.trapezoid(I, t)
        return float(health_cost + ctrl_cost)

    def fit(self, x0: np.ndarray) -> dict:
        n = self.cfg.n_intervals * 2
        u0 = np.full(n, 0.1)
        res = minimize(
            self._cost, u0, args=(x0,), method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * n, options={"maxiter": self.cfg.maxiter},
        )
        self._u_grid = res.x
        return {"converged": bool(res.success), "J_optimal": float(res.fun)}

    def policy(self):
        if self._u_grid is None:
            raise RuntimeError("Appeler fit(x0) avant policy().")
        return self._piecewise(self._u_grid)
