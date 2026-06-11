"""Router — observabilité (santé + dérive)."""

from fastapi import APIRouter, Depends

from api.deps import AppState, get_state
from api.schemas import DriftRequest

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/health")
def health(state: AppState = Depends(get_state)):
    return {"status": "ok", "calibrated": state.params is not None}


@router.post("/drift")
def drift(req: DriftRequest, state: AppState = Depends(get_state)):
    state.ensure_calibrated()
    return state.drift_monitor().check(req.new_I)
