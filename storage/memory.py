"""
Couche STOCKAGE — implémentations en mémoire (par défaut) + fabrique.

En production, ajouter SqliteTimeSeriesRepository / MlflowModelRegistry et les
sélectionner via settings.storage_backend, sans toucher aux services.
"""

from __future__ import annotations

import time
from typing import Any

from core.config import settings
from storage.interfaces import TimeSeriesRepository, ModelRegistry


class InMemoryTimeSeries(TimeSeriesRepository):
    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    def save(self, key: str, payload: dict) -> None:
        self._store[key] = payload

    def load(self, key: str) -> dict | None:
        return self._store.get(key)

    def list_keys(self) -> list[str]:
        return list(self._store.keys())


class InMemoryRegistry(ModelRegistry):
    def __init__(self) -> None:
        self._versions: dict[str, list[dict]] = {}

    def register(self, name: str, artifact: Any, metadata: dict) -> str:
        version = f"v{len(self._versions.get(name, [])) + 1}"
        record = {
            "version": version,
            "artifact": artifact,
            "metadata": metadata,
            "created_at": time.time(),
        }
        self._versions.setdefault(name, []).append(record)
        return version

    def get_latest(self, name: str) -> dict | None:
        versions = self._versions.get(name)
        return versions[-1] if versions else None

    def list_versions(self, name: str) -> list[dict]:
        return [
            {k: v for k, v in r.items() if k != "artifact"}
            for r in self._versions.get(name, [])
        ]


def make_timeseries_repo() -> TimeSeriesRepository:
    backend = settings.storage_backend
    if backend == "memory":
        return InMemoryTimeSeries()
    if backend == "sqlite":
        from storage.sqlite import SqliteTimeSeries

        return SqliteTimeSeries(settings.db_path)
    raise NotImplementedError(f"Backend non supporté : {backend}")


def make_model_registry() -> ModelRegistry:
    backend = settings.storage_backend
    if backend == "memory":
        return InMemoryRegistry()
    if backend == "sqlite":
        from storage.sqlite import SqliteRegistry

        return SqliteRegistry(settings.db_path)
    raise NotImplementedError(f"Backend non supporté : {backend}")
