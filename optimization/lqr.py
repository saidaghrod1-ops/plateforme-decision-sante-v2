"""
Couche MOTEURS — Régulateur Linéaire Quadratique (LQR) via Riccati.

Linéarisation autour de l'équilibre, résolution de l'ARE
    AᵀP + PA - P B R⁻¹ Bᵀ P + Q = 0
puis loi de commande en boucle fermée u*(t) = -K (x - x*),  K = R⁻¹ Bᵀ P.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_continuous_are

from domain.seir import SEIRParams
from domain.epidemiology import endemic_equilibrium
from optimization.base import Controller


@dataclass
class LQRConfig:
    q_I: float = 1.0
    r1: float = 0.5
    r2: float = 0.5
    reg: float = 1e-6   # plancher de pénalisation des états (détectabilité du mode R)


class LQRController(Controller):
    name = "lqr"

    def __init__(self, params: SEIRParams, cfg: LQRConfig | None = None):
        self.p = params
        self.cfg = cfg or LQRConfig()
        self.x_star = endemic_equilibrium(params)
        self.A = self._jacobian()
        self.B = self._control_matrix()
        self.R = np.diag([self.cfg.r1, self.cfg.r2])
        self.P = self._solve_are()
        self.K = np.linalg.solve(self.R, self.B.T @ self.P)

    def _solve_are(self) -> np.ndarray:
        """Résout l'ARE avec régularisation de Q (le compartiment R est un
        intégrateur pur -> mode non détectable -> ARE marginale sans plancher).
        Escalade la régularisation si le solveur échoue (robustesse multi-régimes)."""
        reg = self.cfg.reg
        for _ in range(7):
            Q = np.diag([reg, reg, self.cfg.q_I, reg])
            try:
                P = solve_continuous_are(self.A, self.B, Q, self.R)
                self.Q = Q
                return P
            except Exception:
                reg *= 10.0
        raise RuntimeError("ARE non résolue malgré régularisation croissante")

    def _jacobian(self) -> np.ndarray:
        S, E, I, R = self.x_star
        b, N, sig, gam = self.p.beta, self.p.N, self.p.sigma, self.p.gamma
        return np.array(
            [
                [-b * I / N, 0.0, -b * S / N, 0.0],
                [b * I / N, -sig, b * S / N, 0.0],
                [0.0, sig, -gam, 0.0],
                [0.0, 0.0, gam, 0.0],
            ],
            float,
        )

    def _control_matrix(self) -> np.ndarray:
        S, E, I, R = self.x_star
        inf = S * I
        return np.array([[-S, inf], [0.0, -inf], [0.0, 0.0], [S, 0.0]], float)

    def policy(self):
        K, x_star = self.K, self.x_star

        def _p(t, x):
            u = -K @ (np.asarray(x, float) - x_star)
            return float(np.clip(u[0], 0, 1)), float(np.clip(u[1], 0, 1))

        return _p

    def optimal_cost(self, x0: np.ndarray) -> float:
        dx = np.asarray(x0, float) - self.x_star
        return float(dx.T @ self.P @ dx)
