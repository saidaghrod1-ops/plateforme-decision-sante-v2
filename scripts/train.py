"""Entraînement complet : calibration PINN inverse + solveur HJB-PINN."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # ancrage racine projet

import numpy as np

from domain.seir import SEIRParams
from ingestion import connectors


def run_pinn():
    from ml.inverse_pinn import InversePINN, TrainConfig
    data = connectors.synthetic(n_obs=50, noise=0.03)
    pinn = InversePINN(N=data["N"], t_max=float(np.max(data["t"])), cfg=TrainConfig())
    res = pinn.fit(data["t"], data["y"], data["y0"])
    print("PINN inverse :", {k: round(v, 4) for k, v in res.items()})
    print("Vrais params :", data["truth_params"])


def run_hjb():
    from optimization.hjb_pinn import HJBController, HJBConfig
    from core.config import settings
    # Paramètres : calibrés sur DATA_CSV (données réelles) si défini, sinon synthétiques.
    # Les poids réels sont sauvegardés dans un fichier dédié (suffixé par le nom du CSV).
    if settings.data_csv:
        from services.calibration_service import estimate_from_growth
        params = estimate_from_growth(connectors.from_csv(settings.data_csv))
        wp = Path(settings.hjb_weights_path)
        out = str(wp.with_name(f"{wp.stem}_{Path(settings.data_csv).stem}{wp.suffix}"))
        print("Calibration réelle -> beta=%.3f R0=%.2f N=%.0f"
              % (params.beta, params.beta / params.gamma, params.N))
    else:
        params = SEIRParams()
        out = settings.hjb_weights_path
    ctrl = HJBController(params, HJBConfig(epochs=6000))
    print("HJB-PINN :", ctrl.fit())
    pol = ctrl.policy()
    x = np.array([params.N * 0.8, params.N * 0.02, params.N * 0.05, params.N * 0.13])
    print("u*(x, t=30) =", pol(30.0, x))
    ctrl.save(out)
    print("Poids sauvegardés ->", out)
    print("Pour l'utiliser : HJB_WEIGHTS_PATH=%s" % out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinn", action="store_true")
    ap.add_argument("--hjb", action="store_true")
    a = ap.parse_args()
    if not (a.pinn or a.hjb):
        ap.print_help()
    if a.pinn:
        run_pinn()
    if a.hjb:
        run_hjb()
