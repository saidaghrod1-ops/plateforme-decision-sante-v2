"""
Couche PRÉSENTATION — composition root (injection de dépendances).

Assemble les implémentations concrètes (stockage, registre) et les services.
C'est le SEUL endroit qui connaît les détails concrets : les couches internes
ne dépendent que des interfaces. Maintient l'état applicatif partagé.
"""

from __future__ import annotations

import numpy as np

from core.config import settings
from domain.seir import SEIRParams
from storage.memory import make_timeseries_repo, make_model_registry
from ingestion import connectors
from services.calibration_service import CalibrationService
from services.scenario_service import ScenarioService
from services.benchmark_service import BenchmarkService
from services.recommendation_engine import RecommendationEngine
from observability.drift import DriftMonitor


class AppState:
    """Singleton applicatif assemblant les couches."""

    def __init__(self) -> None:
        self.timeseries = make_timeseries_repo()
        self.registry = make_model_registry()
        self.calibration = CalibrationService(self.registry)
        self.params: SEIRParams | None = None
        self.data: dict | None = None

    def ensure_calibrated(self, use_pinn: bool = False) -> SEIRParams:
        if self.params is None:
            if settings.data_csv:                    # données réelles (CSV t,S,E,I,R)
                self.data = connectors.from_csv(settings.data_csv)
            else:                                    # données synthétiques (défaut)
                self.data = connectors.synthetic()
            self.timeseries.save("latest_ingest", {"I": self.data["y"][:, 2].tolist()})
            result = self.calibration.calibrate(self.data, use_pinn=use_pinn)
            self.params = result["params"]
        return self.params

    def scenario_service(self) -> ScenarioService:
        return ScenarioService(self.ensure_calibrated())

    def benchmark_service(self) -> BenchmarkService:
        return BenchmarkService(self.ensure_calibrated())

    def recommendation_engine(self) -> RecommendationEngine:
        return RecommendationEngine(self.ensure_calibrated())

    def drift_monitor(self) -> DriftMonitor:
        ref = self.data["y"][:, 2] if self.data else np.zeros(10)
        return DriftMonitor(ref)


state = AppState()


def get_state() -> AppState:
    return state
