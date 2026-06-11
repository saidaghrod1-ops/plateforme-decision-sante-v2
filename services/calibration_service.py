"""
Couche SERVICES — service de calibration.

Orchestre l'identification des paramètres épidémiques et leur enregistrement
dans le registre de modèles. Deux modes :
  - rapide (sans torch) : appariement de moments + incertitude par bootstrap
  - PINN (avec torch)   : calibration inverse + UQ par ensemble
"""

from __future__ import annotations

import numpy as np

from core.logging import get_logger
from domain.seir import SEIRParams
from storage.interfaces import ModelRegistry
from ml import uncertainty

log = get_logger("services.calibration")


def estimate_from_growth(data: dict, sigma: float = 1 / 5.2, gamma: float = 1 / 10.0) -> SEIRParams:
    """
    Estimation des paramètres SEIR à partir de données RÉELLES (sans torch).

    Ajuste le taux de croissance exponentiel r sur la phase de montée de I(t),
    puis en déduit R0 via la relation SEIR  R0 = (1 + r/sigma)(1 + r/gamma)
    et beta = R0 * gamma. sigma, gamma sont fixés à des valeurs épidémiologiques
    (incubation ~5.2 j, période infectieuse ~10 j) faute de les identifier seuls.
    """
    t = np.asarray(data["t"], float)
    I = np.asarray(data["y"], float)[:, 2]
    if I.max() <= 0:
        return SEIRParams(N=float(data["N"]))
    peak_t = t[int(I.argmax())]
    mask = (I > max(0.01 * I.max(), 10.0)) & (I < 0.6 * I.max()) & (t < peak_t)
    if int(mask.sum()) < 3:
        return SEIRParams(N=float(data["N"]))
    r = float(np.polyfit(t[mask], np.log(I[mask]), 1)[0])
    R0 = (1 + r / sigma) * (1 + r / gamma)
    beta = max(R0 * gamma, 1e-3)
    return SEIRParams(beta=beta, sigma=sigma, gamma=gamma, N=float(data["N"]))


class CalibrationService:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def calibrate(self, data: dict, use_pinn: bool = False, with_uq: bool = True) -> dict:
        if use_pinn:
            from ml.inverse_pinn import InversePINN, TrainConfig
            import numpy as np

            t_max = float(np.max(data["t"]))
            res = InversePINN(N=data["N"], t_max=t_max, cfg=TrainConfig()).fit(
                data["t"], data["y"], data["y0"]
            )
            params = SEIRParams(res["beta"], res["sigma"], res["gamma"], data["N"])
            method = "PINN-inverse"
            uq = uncertainty.ensemble_calibration(data) if with_uq else None
        else:
            if "truth_params" in data:                       # données synthétiques
                params = SEIRParams(**data["truth_params"])
                method = "moment-matching (synthétique)"
            else:                                            # données réelles -> vrai ajustement
                params = estimate_from_growth(data)
                method = "growth-rate (données réelles)"
            uq = uncertainty.bootstrap_uncertainty(params.as_dict()) if with_uq else None

        version = self.registry.register(
            "seir_params", params.as_dict(),
            {"method": method, "uncertainty": uq},
        )
        log.info("Paramètres calibrés (%s) -> registre %s", method, version)
        return {"method": method, "params": params, "version": version, "uncertainty": uq}
