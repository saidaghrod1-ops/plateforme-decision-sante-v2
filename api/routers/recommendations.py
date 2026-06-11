"""Router — recommandations décisionnelles."""

from fastapi import APIRouter, Depends

from api.deps import AppState, get_state

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("")
def recommend(horizon: float = 180.0, state: AppState = Depends(get_state)):
    bench = state.benchmark_service().run(["baseline", "lqr"], horizon)
    details = bench["details"]
    # Meilleure trajectoire contrôlée disponible (le LQR peut échouer selon le régime).
    traj = next(
        (details[k]["trajectory"] for k in ("lqr", "pmp", "baseline")
         if k in details and "trajectory" in details[k]),
        None,
    )
    if traj is None:
        traj = details["baseline"]["trajectory"]
    return state.recommendation_engine().recommend(traj, horizon)
