"""
Couche ML — quantification d'incertitude (UQ) sur la calibration.

Ensemble de PINNs inverses (graines différentes) -> moyenne et intervalle de
confiance sur (beta, gamma, sigma). Sans torch, un repli par bootstrap perturbe
une estimation de base pour fournir des barres d'incertitude indicatives.
"""

from __future__ import annotations

import numpy as np


def ensemble_calibration(data: dict, n_members: int = 5) -> dict:
    """Calibration PINN en ensemble (nécessite torch)."""
    from ml.inverse_pinn import InversePINN, TrainConfig

    t_max = float(np.max(data["t"]))
    samples = {"beta": [], "gamma": [], "sigma": []}
    for seed in range(n_members):
        pinn = InversePINN(N=data["N"], t_max=t_max, cfg=TrainConfig(epochs=4000, seed=seed))
        res = pinn.fit(data["t"], data["y"], data["y0"])
        for k in samples:
            samples[k].append(res[k])
    return _summarize(samples)


def bootstrap_uncertainty(base_params: dict, rel_noise: float = 0.05,
                          n_members: int = 200, seed: int = 0) -> dict:
    """Repli rapide sans torch : barres d'incertitude par perturbation."""
    rng = np.random.default_rng(seed)
    samples = {
        k: (base_params[k] * (1 + rel_noise * rng.standard_normal(n_members))).tolist()
        for k in ("beta", "gamma", "sigma")
    }
    return _summarize(samples)


def _summarize(samples: dict) -> dict:
    out = {}
    for k, vals in samples.items():
        arr = np.asarray(vals, float)
        out[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        }
    return out
