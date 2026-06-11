"""
Couche PRÉSENTATION — application FastAPI (API gateway).

Assemble les routers versionnés de chaque service. Lancement :
    uvicorn api.main:app --reload --port 8000   ->  http://localhost:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import calibration, scenarios, benchmark, recommendations, monitoring

app = FastAPI(
    title="Plateforme d'Aide à la Décision en Santé Publique — v2 (architecture en couches)",
    description="Contrôle optimal (HJB · PMP · LQR) + PINNs · modèle SEIR · MLOps.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

API_PREFIX = "/api/v1"
for module in (calibration, scenarios, benchmark, recommendations, monitoring):
    app.include_router(module.router, prefix=API_PREFIX)


@app.get("/")
def root():
    return {"service": "plateforme-decision-sante", "version": "2.0.0", "docs": "/docs"}
