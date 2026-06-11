# Plateforme d'Aide à la Décision en Santé Publique — v2

Architecture **en couches (hexagonale) orientée services + MLOps**, alternative
plus développée au pipeline « à plat » de la slide 25.

Contrôle optimal (**HJB-PINN · PMP · LQR**) et **PINNs** appliqués au modèle **SEIR**.

> Diagramme et explication détaillée des couches : voir `ARCHITECTURE.md`.

---

## Nouveautés vs v1

- Interface `Controller` unifiée → trois stratégies comparables : LQR (Riccati),
  PMP (méthode directe NLP), HJB-PINN (fonction valeur neuronale).
- Couche stockage en **ports & adaptateurs** + **registre de modèles versionné**.
- **Quantification d'incertitude** sur la calibration (ensemble / bootstrap).
- **Orchestration** par DAG de tâches.
- **Observabilité** : logging structuré + **détection de dérive** (KS-test).
- API **versionnée** (`/api/v1/...`) avec routers par service.

---

## Structure

```
plateforme-v2/
├── core/              # config, logging (transversal)
├── domain/            # SEIR + épidémiologie (noyau pur)
├── ingestion/         # connecteurs, validation, ETL
├── storage/           # interfaces (ports) + impl. mémoire + registry
├── ml/                # réseau, pertes, PINN inverse, incertitude
├── optimization/      # base/factory + LQR + PMP + HJB-PINN
├── orchestration/     # exécuteur de DAG
├── services/          # calibration, scénarios, benchmark, reco
├── observability/     # détection de dérive
├── api/               # FastAPI : deps (composition root), routers, schemas
├── dashboard/         # Streamlit
├── scripts/train.py   # entraînement PINN + HJB
├── tests/             # tests pile complète (sans torch)
└── Dockerfile · docker-compose.yml · requirements.txt
```

---

## Démarrage

### Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
export PYTHONPATH=$PWD          # exécution depuis la racine du projet

pytest tests/ -q                                  # tests
uvicorn api.main:app --reload --port 8000         # API → /docs
streamlit run dashboard/app.py                    # dashboard → :8501
```

### Docker

```bash
docker compose up --build
# API → http://localhost:8000/docs   |   Dashboard → http://localhost:8501
```

### Entraînement IA

```bash
python scripts/train.py --pinn     # calibration inverse (β, γ, σ)
python scripts/train.py --hjb      # solveur HJB-PINN
```

---

## API (`/api/v1`)

| Méthode | Route                        | Rôle                                  |
|---------|------------------------------|---------------------------------------|
| POST    | `/calibration`               | Calibrer β, γ, σ (+ incertitude)      |
| GET     | `/calibration/versions`      | Historique versionné des paramètres   |
| POST    | `/scenarios/whatif`          | Simuler un scénario (u₁, u₂)          |
| POST    | `/benchmark`                 | Comparer baseline / lqr / pmp / hjb   |
| GET     | `/recommendations`           | Timing, seuils, niveau de risque (R₀) |
| GET     | `/monitoring/health`         | Sonde de vivacité                     |
| POST    | `/monitoring/drift`          | Détection de dérive sur nouvelles données |

Exemple :

```bash
curl -X POST localhost:8000/api/v1/benchmark \
  -H "Content-Type: application/json" \
  -d '{"strategies":["baseline","lqr","pmp"],"horizon":180}'
```

---

## Notes

- Calibration rapide par défaut (sans torch) ; PINN complet via `use_pinn=true`.
- Le benchmark est **robuste** : si un contrôleur échoue (ex. torch absent pour
  HJB), il est ignoré et les autres stratégies sont tout de même comparées.
- **Persistance** : `STORAGE_BACKEND=sqlite` active les adaptateurs
  `SqliteTimeSeries` / `SqliteRegistry` (fichier `DB_PATH`, mode WAL, registre
  versionné sur disque). `memory` reste le défaut pour les tests/démos. Le
  `docker-compose` utilise SQLite sur un volume nommé (`sante-data`).
- Pile production : conteneurs **non-root**, **HEALTHCHECK** sur
  `/api/v1/monitoring/health`, `restart: unless-stopped`, et démarrage du
  dashboard conditionné à la santé de l'API (`depends_on: service_healthy`).
- Extensions futures : adaptateurs PostgreSQL/MLflow derrière les mêmes
  interfaces `storage/`, et orchestrateur branché sur Celery/Airflow.
