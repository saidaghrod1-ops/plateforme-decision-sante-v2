"""Tests des adaptateurs de stockage (mémoire + SQLite) — sans torch.

Vérifient le contrat des ports `TimeSeriesRepository` / `ModelRegistry` sur les
deux implémentations, plus la persistance réelle sur disque pour SQLite.
"""

import pytest

from storage.memory import InMemoryTimeSeries, InMemoryRegistry
from storage.sqlite import SqliteTimeSeries, SqliteRegistry


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "test.db")


def _timeseries_impls(db):
    return [InMemoryTimeSeries(), SqliteTimeSeries(db)]


def _registry_impls(db):
    return [InMemoryRegistry(), SqliteRegistry(db)]


@pytest.mark.parametrize("idx", [0, 1], ids=["memory", "sqlite"])
def test_timeseries_save_load_list(db, idx):
    repo = _timeseries_impls(db)[idx]
    assert repo.load("absent") is None
    assert repo.list_keys() == []

    repo.save("k1", {"I": [1, 2, 3]})
    repo.save("k2", {"I": [4, 5]})
    assert repo.load("k1") == {"I": [1, 2, 3]}
    assert set(repo.list_keys()) == {"k1", "k2"}

    # upsert : ré-écriture sur la même clé
    repo.save("k1", {"I": [9]})
    assert repo.load("k1") == {"I": [9]}
    assert len(repo.list_keys()) == 2


@pytest.mark.parametrize("idx", [0, 1], ids=["memory", "sqlite"])
def test_registry_versioning(db, idx):
    reg = _registry_impls(db)[idx]
    assert reg.get_latest("seir_params") is None
    assert reg.list_versions("seir_params") == []

    v1 = reg.register("seir_params", {"beta": 0.5}, {"method": "moment-matching"})
    v2 = reg.register("seir_params", {"beta": 0.6}, {"method": "PINN-inverse"})
    assert (v1, v2) == ("v1", "v2")

    latest = reg.get_latest("seir_params")
    assert latest["version"] == "v2"
    assert latest["artifact"] == {"beta": 0.6}
    assert latest["metadata"]["method"] == "PINN-inverse"

    versions = reg.list_versions("seir_params")
    assert [r["version"] for r in versions] == ["v1", "v2"]
    # list_versions n'expose pas l'artefact (métadonnées seulement)
    assert "artifact" not in versions[0]


def test_sqlite_persists_across_connections(db):
    """Une nouvelle instance pointant sur le même fichier retrouve les données."""
    SqliteTimeSeries(db).save("latest_ingest", {"I": [1, 2, 3]})
    SqliteRegistry(db).register("seir_params", {"beta": 0.55}, {"method": "m"})

    assert SqliteTimeSeries(db).load("latest_ingest") == {"I": [1, 2, 3]}
    reg2 = SqliteRegistry(db)
    assert reg2.get_latest("seir_params")["artifact"] == {"beta": 0.55}
    # le compteur de versions repart de l'état persté, pas de zéro
    assert reg2.register("seir_params", {"beta": 0.7}, {"method": "m"}) == "v2"


def test_make_factories_dispatch_on_backend(db, monkeypatch):
    """Les fabriques renvoient l'implémentation correspondant à STORAGE_BACKEND."""
    import storage.memory as mem
    from core.config import Settings

    monkeypatch.setattr(mem, "settings", Settings(storage_backend="sqlite", db_path=db))
    assert isinstance(mem.make_timeseries_repo(), SqliteTimeSeries)
    assert isinstance(mem.make_model_registry(), SqliteRegistry)

    monkeypatch.setattr(mem, "settings", Settings(storage_backend="memory"))
    assert isinstance(mem.make_timeseries_repo(), InMemoryTimeSeries)
    assert isinstance(mem.make_model_registry(), InMemoryRegistry)
