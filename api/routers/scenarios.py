"""Router — scénarios What-If."""

from fastapi import APIRouter, Depends

from api.deps import AppState, get_state
from api.schemas import WhatIfRequest, TrajectoryResponse

router = APIRouter(prefix="/scenarios", tags=["scenarios"])


@router.post("/whatif", response_model=TrajectoryResponse)
def whatif(req: WhatIfRequest, state: AppState = Depends(get_state)):
    res = state.scenario_service().run(req.u1, req.u2, req.horizon)
    return TrajectoryResponse(**res["trajectory"], metrics=res["metrics"])
