"""Configuration centrale (couche transversale)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    population: float = float(os.getenv("POPULATION", "1000000"))
    default_horizon: float = float(os.getenv("DEFAULT_HORIZON", "180"))
    device: str = os.getenv("DEVICE", "cpu")
    use_pinn: bool = os.getenv("USE_PINN", "0") == "1"
    storage_backend: str = os.getenv("STORAGE_BACKEND", "memory")  # memory | sqlite
    db_path: str = os.getenv("DB_PATH", "data/platform.db")  # utilisé si backend=sqlite
    hjb_weights_path: str = os.getenv("HJB_WEIGHTS_PATH", "artifacts/hjb_value.pt")
    data_csv: str = os.getenv("DATA_CSV", "")  # chemin CSV réel (t,S,E,I,R) ; vide -> synthétique
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
