"""
Couche STOCKAGE — adaptateurs SQLite (persistance pour la production).

Implémentations concrètes des ports `TimeSeriesRepository` et `ModelRegistry`
au-dessus de la bibliothèque standard `sqlite3`. Sélectionnées via
`STORAGE_BACKEND=sqlite` (voir `storage.memory.make_*`), sans toucher aux services.

Conçu pour un déploiement conteneurisé : un fichier de base unique (monté sur un
volume), mode WAL pour la concurrence lecture/écriture, et une connexion par
opération — sûr vis-à-vis du multithreading d'uvicorn.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from storage.interfaces import TimeSeriesRepository, ModelRegistry


@contextmanager
def _connect(db_path: str) -> Iterator[sqlite3.Connection]:
    """Connexion éphémère configurée (WAL + clés étrangères), auto-commit/rollback."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class SqliteTimeSeries(TimeSeriesRepository):
    """Séries temporelles persistées dans une table clé -> payload JSON (upsert)."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS timeseries (
                    key        TEXT PRIMARY KEY,
                    payload    TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def save(self, key: str, payload: dict) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO timeseries (key, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (key, json.dumps(payload), time.time()),
            )

    def load(self, key: str) -> dict | None:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT payload FROM timeseries WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def list_keys(self) -> list[str]:
        with _connect(self.db_path) as conn:
            rows = conn.execute("SELECT key FROM timeseries ORDER BY key").fetchall()
        return [r["key"] for r in rows]


class SqliteRegistry(ModelRegistry):
    """Registre de modèles versionné (une ligne par version, artefact JSON)."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_versions (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    name       TEXT NOT NULL,
                    version    TEXT NOT NULL,
                    artifact   TEXT NOT NULL,
                    metadata   TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(name, version)
                )
                """
            )

    def register(self, name: str, artifact: Any, metadata: dict) -> str:
        with _connect(self.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS n FROM model_versions WHERE name = ?", (name,)
            ).fetchone()["n"]
            version = f"v{count + 1}"
            conn.execute(
                """
                INSERT INTO model_versions
                    (name, version, artifact, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, version, json.dumps(artifact), json.dumps(metadata), time.time()),
            )
        return version

    def get_latest(self, name: str) -> dict | None:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT version, artifact, metadata, created_at
                FROM model_versions WHERE name = ?
                ORDER BY id DESC LIMIT 1
                """,
                (name,),
            ).fetchone()
        if row is None:
            return None
        return {
            "version": row["version"],
            "artifact": json.loads(row["artifact"]),
            "metadata": json.loads(row["metadata"]),
            "created_at": row["created_at"],
        }

    def list_versions(self, name: str) -> list[dict]:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT version, metadata, created_at
                FROM model_versions WHERE name = ?
                ORDER BY id ASC
                """,
                (name,),
            ).fetchall()
        return [
            {
                "version": r["version"],
                "metadata": json.loads(r["metadata"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]
