"""
Couche STOCKAGE — interfaces (ports de l'architecture hexagonale).

Définit les contrats que toute implémentation (mémoire, SQLite, PostgreSQL,
MLflow...) doit respecter. Les services dépendent de ces abstractions, jamais
d'une implémentation concrète -> facilement remplaçable en production.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TimeSeriesRepository(ABC):
    """Stockage des séries temporelles épidémiques."""

    @abstractmethod
    def save(self, key: str, payload: dict) -> None: ...

    @abstractmethod
    def load(self, key: str) -> dict | None: ...

    @abstractmethod
    def list_keys(self) -> list[str]: ...


class ModelRegistry(ABC):
    """Registre versionné des modèles/paramètres calibrés."""

    @abstractmethod
    def register(self, name: str, artifact: Any, metadata: dict) -> str: ...

    @abstractmethod
    def get_latest(self, name: str) -> dict | None: ...

    @abstractmethod
    def list_versions(self, name: str) -> list[dict]: ...
