"""Router — benchmark des stratégies de contrôle."""

from fastapi import APIRouter, Depends

from api.deps import AppState, get_state
from api.schemas import BenchmarkRequest

router = APIRouter(prefix="/benchmark", tags=["benchmark"])


@router.post("")
def benchmark(req: BenchmarkRequest, state: AppState = Depends(get_state)):
    res = state.benchmark_service().run(req.strategies, req.horizon)
    return res["summary"]
