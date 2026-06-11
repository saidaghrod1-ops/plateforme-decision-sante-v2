"""Router — calibration des paramètres épidémiques."""

from fastapi import APIRouter, Depends

from api.deps import AppState, get_state
from api.schemas import CalibrateRequest, CalibrateResponse
from domain.epidemiology import basic_reproduction_number

router = APIRouter(prefix="/calibration", tags=["calibration"])


@router.post("", response_model=CalibrateResponse)
def calibrate(req: CalibrateRequest, state: AppState = Depends(get_state)):
    state.params = None  # force une recalibration
    p = state.ensure_calibrated(use_pinn=req.use_pinn)
    latest = state.registry.get_latest("seir_params")
    return CalibrateResponse(
        method=latest["metadata"]["method"],
        beta=p.beta, gamma=p.gamma, sigma=p.sigma,
        R0=round(basic_reproduction_number(p), 3),
        uncertainty=latest["metadata"].get("uncertainty"),
    )


@router.get("/versions")
def versions(state: AppState = Depends(get_state)):
    return state.registry.list_versions("seir_params")
