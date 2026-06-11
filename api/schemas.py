"""Couche PRÉSENTATION — schémas Pydantic."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CalibrateRequest(BaseModel):
    use_pinn: bool = Field(False, description="Calibration PINN complète (lente) vs rapide")


class CalibrateResponse(BaseModel):
    method: str
    beta: float
    gamma: float
    sigma: float
    R0: float
    uncertainty: dict | None = None


class WhatIfRequest(BaseModel):
    u1: float = Field(0.0, ge=0, le=1)
    u2: float = Field(0.0, ge=0, le=1)
    horizon: float = Field(180.0, gt=0)


class TrajectoryResponse(BaseModel):
    t: list[float]
    S: list[float]
    E: list[float]
    I: list[float]
    R: list[float]
    metrics: dict


class BenchmarkRequest(BaseModel):
    strategies: list[str] = Field(default_factory=lambda: ["baseline", "lqr", "pmp"])
    horizon: float = 180.0


class DriftRequest(BaseModel):
    new_I: list[float]
