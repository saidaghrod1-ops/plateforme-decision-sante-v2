"""
Couche SERVICES — benchmark des stratégies de contrôle.

Compare de façon interchangeable (via l'interface Controller) :
  - laisser-faire (aucun contrôle)
  - LQR (boucle fermée, Riccati)
  - PMP direct (boucle ouverte, NLP)
  - HJB-PINN (boucle fermée neuronale) — si torch disponible et demandé
"""

from __future__ import annotations

import numpy as np

from core.logging import get_logger
from domain.seir import SEIRParams, simulate
from domain.epidemiology import trajectory_metrics
from optimization.base import make_controller
from optimization.lqr import LQRController

log = get_logger("services.benchmark")


class BenchmarkService:
    def __init__(self, params: SEIRParams):
        self.params = params

    def _x0(self) -> np.ndarray:
        return np.array([self.params.N - 100.0, 50.0, 50.0, 0.0])

    @staticmethod
    def _record_controls(traj: dict, policy) -> dict:
        """Rejoue la politique le long de la trajectoire -> u1(t), u2(t) (pour
        affichage : effort de vaccination/confinement appliqué par la stratégie)."""
        t = np.asarray(traj["t"], float)
        X = np.vstack([traj["S"], traj["E"], traj["I"], traj["R"]]).T
        u = np.array([np.clip(policy(ti, xi), 0.0, 1.0) for ti, xi in zip(t, X)], float)
        return {"u1": u[:, 0].tolist(), "u2": u[:, 1].tolist()}

    def run(self, strategies: list[str] | None = None, horizon: float = 180.0) -> dict:
        strategies = strategies or ["baseline", "lqr", "pmp"]
        x0 = self._x0()
        n = int(horizon) + 1
        results: dict[str, dict] = {}

        baseline = simulate(self.params, x0, (0, horizon), n)
        results["baseline"] = {
            "trajectory": baseline,
            "metrics": trajectory_metrics(baseline),
            "controls": {"u1": [0.0] * len(baseline["t"]), "u2": [0.0] * len(baseline["t"])},
        }

        for strat in strategies:
            if strat == "baseline":
                continue
            try:
                ctrl = make_controller(strat, self.params, **self._kwargs(strat))
                if hasattr(ctrl, "fit") and strat == "pmp":
                    ctrl.fit(x0)
                if strat == "hjb":
                    # Recharge des poids entraînés si disponibles (rapide),
                    # sinon entraînement à la volée (lent).
                    from pathlib import Path as _Path
                    from core.config import settings
                    if hasattr(ctrl, "load") and _Path(settings.hjb_weights_path).exists():
                        ctrl.load(settings.hjb_weights_path)
                        log.info("HJB-PINN : poids rechargés depuis %s", settings.hjb_weights_path)
                    else:
                        ctrl.fit()
                policy = ctrl.policy()
                traj = simulate(self.params, x0, (0, horizon), n, policy=policy)
                results[strat] = {
                    "trajectory": traj,
                    "metrics": trajectory_metrics(traj),
                    "controls": self._record_controls(traj, policy),
                }
            except Exception as exc:  # robustesse : un contrôleur HS n'arrête pas le benchmark
                log.warning("Stratégie %s ignorée : %s", strat, exc)
                results[strat] = {"error": str(exc)}

        return self._summarize(results)

    def _kwargs(self, strat: str) -> dict:
        if strat == "pmp":
            from optimization.pmp import PMPConfig
            return {"cfg": PMPConfig(horizon=180.0)}
        return {}

    def _summarize(self, results: dict) -> dict:
        base_peak = results["baseline"]["metrics"]["peak_I"]
        table = {}
        for name, r in results.items():
            if "metrics" not in r:
                table[name] = r
                continue
            m = r["metrics"]
            table[name] = {
                "peak_I": m["peak_I"],
                "peak_day": m["peak_day"],
                "peak_reduction_pct": 100.0 * (1 - m["peak_I"] / base_peak),
            }
        return {"summary": table, "details": results}
