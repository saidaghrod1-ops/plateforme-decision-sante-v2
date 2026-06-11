"""Couche SERVICES — simulation de scénarios What-If."""

from __future__ import annotations

import numpy as np

from domain.seir import SEIRParams, simulate, constant_control
from domain.epidemiology import trajectory_metrics


class ScenarioService:
    def __init__(self, params: SEIRParams):
        self.params = params

    def _x0(self) -> np.ndarray:
        return np.array([self.params.N - 100.0, 50.0, 50.0, 0.0])

    def run(self, u1: float, u2: float, horizon: float = 180.0) -> dict:
        traj = simulate(self.params, self._x0(), (0, horizon),
                        int(horizon) + 1, policy=constant_control(u1, u2))
        return {"trajectory": traj, "metrics": trajectory_metrics(traj)}
