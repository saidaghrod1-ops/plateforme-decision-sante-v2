# Architecture — v2 (en couches, orientée services + MLOps)

## Principe

Contrairement au pipeline « à plat » de la slide 25 (6 étapes en séquence), cette
version adopte une **architecture en couches de type hexagonal** : un cœur métier
pur (domaine + moteurs scientifiques) entouré de couches de service et
d'infrastructure. Les dépendances pointent toujours **vers l'intérieur** — les
couches internes ignorent tout des couches externes.

```
┌──────────────────────────────────────────────────────────────┐
│  PRÉSENTATION    Dashboard (Streamlit)  ·  API gateway (FastAPI) │   ← externe
├──────────────────────────────────────────────────────────────┤
│  SERVICES        Calibration · Scénarios · Benchmark · Reco     │
├──────────────────────────────────────────────────────────────┤
│  ORCHESTRATION   Pipeline DAG (tâches ordonnées + contexte)     │
├──────────────────────────────────────────────────────────────┤
│  MOTEURS         ML/PINN (forward·inverse·UQ) · Contrôle        │
│                  optimal (HJB-PINN · PMP · LQR)                 │
├──────────────────────────────────────────────────────────────┤
│  DOMAINE         Modèle SEIR · R₀ · équilibres · métriques      │   ← noyau pur
├──────────────────────────────────────────────────────────────┤
│  DONNÉES         Ingestion (connecteurs·ETL) · Stockage/registry│
└──────────────────────────────────────────────────────────────┘
   TRANSVERSAL : core (config, logging) · observabilité (drift)
```

## Ce qui change par rapport à la v1

| Aspect            | v1 (pipeline plat)            | v2 (en couches)                                   |
|-------------------|-------------------------------|---------------------------------------------------|
| Structure         | modules chaînés               | couches à dépendances dirigées vers le noyau      |
| Contrôleurs       | LQR seul (baseline)           | interface `Controller` unifiée : LQR · PMP · HJB  |
| Méthode directe   | absente                       | PMP par NLP (`scipy.optimize`) — boucle ouverte   |
| Incertitude       | absente                       | UQ par ensemble (PINN) / bootstrap (rapide)       |
| Stockage          | état en mémoire implicite     | ports `TimeSeriesRepository` / `ModelRegistry`    |
| Versionnement     | aucun                         | registre de modèles versionné                     |
| Orchestration     | appels directs                | DAG de tâches (tri topologique)                   |
| Observabilité     | aucune                        | logging structuré + détection de dérive (KS-test) |
| API               | endpoints à plat              | routers versionnés `/api/v1/...`                  |

## Couches en détail

### Domaine (`domain/`) — noyau pur
`seir.py` (dynamique contrôlée) et `epidemiology.py` (R₀, R_eff, équilibres,
métriques de trajectoire). Aucune dépendance ML/API : entièrement testable seul.

### Moteurs (`optimization/`, `ml/`)
- `optimization/base.py` : interface `Controller` + `make_controller()` (fabrique).
- `lqr.py` : Riccati (boucle fermée). `pmp.py` : méthode directe NLP (boucle ouverte).
  `hjb_pinn.py` : fonction valeur neuronale (boucle fermée). Tous interchangeables.
- `ml/` : réseau, pertes autograd, calibration inverse, quantification d'incertitude.

### Stockage (`storage/`) — ports & adaptateurs
`interfaces.py` définit les contrats ; `memory.py` (volatile) et `sqlite.py`
(persistant, mode WAL) les implémentent. Le backend est choisi via
`STORAGE_BACKEND` (`memory` | `sqlite`) dans les fabriques `make_*`, sans toucher
aux services. Étendre = ajouter `PostgresTimeSeries` / `MlflowRegistry` selon le
même contrat.

### Orchestration (`orchestration/pipeline.py`)
Mini-moteur de DAG : tâches + dépendances, tri topologique, contexte partagé.
Remplaçable par Airflow/Prefect/Celery en gardant les signatures.

### Services (`services/`)
Logique métier : `CalibrationService`, `ScenarioService`, `BenchmarkService`,
`RecommendationEngine`. Dépendent des interfaces, jamais des implémentations.

### Présentation (`api/`, `dashboard/`)
`api/deps.py` est la **composition root** : seul endroit qui assemble les
implémentations concrètes. Routers versionnés sous `/api/v1`.

### Transversal (`core/`, `observability/`)
Config par variables d'environnement, logging structuré, et `DriftMonitor`
(test de Kolmogorov-Smirnov) qui déclenche une recalibration.

## Flux de bout en bout

```
ingestion.synthetic / from_csv
        ↓
CalibrationService → β, γ, σ (+ UQ) → ModelRegistry (versionné)
        ↓
make_controller("lqr"|"pmp"|"hjb") → policy()
        ↓
BenchmarkService → comparaison vs laisser-faire
        ↓
RecommendationEngine → timing, seuils, niveau de risque
        ↓
API /api/v1/...  +  Dashboard  +  DriftMonitor (boucle de rétroaction)
```
