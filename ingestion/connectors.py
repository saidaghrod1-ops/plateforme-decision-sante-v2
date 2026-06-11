"""
Couche INGESTION — connecteurs, validation, ETL.

Connecteurs : CSV, données synthétiques (dev), et squelette de connecteur REST
(API ministère/OMS) à compléter en production. La validation garantit que toute
série injectée respecte le schéma SEIR avant d'atteindre les couches supérieures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.logging import get_logger
from core.logging import DataValidationError
from domain.seir import SEIRParams, simulate

log = get_logger("ingestion")

REQUIRED_COLUMNS = {"t", "S", "E", "I", "R"}


def validate(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataValidationError(f"Colonnes manquantes : {missing}")
    if (df[["S", "E", "I", "R"]] < 0).any().any():
        raise DataValidationError("Compartiments négatifs détectés.")
    if df["t"].duplicated().any():
        raise DataValidationError("Pas de temps dupliqués.")


def from_csv(path: str | Path) -> dict:
    df = pd.read_csv(path).sort_values("t").reset_index(drop=True)
    validate(df)
    log.info("CSV ingéré : %d observations", len(df))
    return _to_payload(df)


def synthetic(
    params: SEIRParams | None = None,
    t_max: float = 120.0,
    n_obs: int = 40,
    noise: float = 0.02,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    params = params or SEIRParams()
    x0 = np.array([params.N - 100.0, 50.0, 50.0, 0.0])
    truth = simulate(params, x0, (0.0, t_max), int(t_max) + 1)

    t_full = np.array(truth["t"])
    y_full = np.stack([truth["S"], truth["E"], truth["I"], truth["R"]], axis=1)
    idx = np.sort(rng.choice(len(t_full), size=n_obs, replace=False))
    y_obs = np.clip(y_full[idx] * (1 + noise * rng.standard_normal(y_full[idx].shape)), 0, None)

    df = pd.DataFrame(
        {"t": t_full[idx], "S": y_obs[:, 0], "E": y_obs[:, 1], "I": y_obs[:, 2], "R": y_obs[:, 3]}
    )
    validate(df)
    payload = _to_payload(df)
    payload["y0"] = x0
    payload["truth_params"] = params.as_dict()
    return payload


def _to_payload(df: pd.DataFrame) -> dict:
    y = df[["S", "E", "I", "R"]].to_numpy(float)
    return {
        "t": df["t"].to_numpy(float),
        "y": y,
        "y0": y[0].copy(),  # état initial (S,E,I,R) — requis par la calibration PINN
        "N": float(y[0].sum()),
    }
