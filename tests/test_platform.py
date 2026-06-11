"""Tests de la pile v2 (cœur numérique, contrôleurs, services) — sans torch."""

import numpy as np

from domain.seir import SEIRParams, simulate
from domain.epidemiology import basic_reproduction_number, trajectory_metrics, endemic_equilibrium
from optimization.lqr import LQRController
from optimization.pmp import PMPController, PMPConfig
from services.benchmark_service import BenchmarkService
from observability.drift import DriftMonitor


def _x0(p):
    return np.array([p.N - 100, 50, 50, 0.0])


def test_seir_conservation():
    p = SEIRParams()
    traj = simulate(p, _x0(p), (0, 120), 121)
    total = sum(np.array(traj[k]) for k in "SEIR")
    assert np.allclose(total, p.N, rtol=1e-4)


def test_r0():
    p = SEIRParams(beta=0.4, gamma=0.1)
    assert abs(basic_reproduction_number(p) - 4.0) < 1e-9


def test_lqr_stabilizes():
    p = SEIRParams()
    ctrl = LQRController(p)
    assert np.allclose(ctrl.P, ctrl.P.T, atol=1e-6)
    eig = np.linalg.eigvals(ctrl.A - ctrl.B @ ctrl.K).real
    assert (eig <= 1e-6).all()


def test_pmp_reduces_peak():
    p = SEIRParams()
    ctrl = PMPController(p, PMPConfig(n_intervals=8, maxiter=20))
    ctrl.fit(_x0(p))
    base = simulate(p, _x0(p), (0, 180), 181)
    opt = simulate(p, _x0(p), (0, 180), 181, policy=ctrl.policy())
    assert max(opt["I"]) < max(base["I"])


def test_benchmark_summary():
    p = SEIRParams()
    res = BenchmarkService(p).run(["baseline", "lqr"], horizon=180)
    assert res["summary"]["lqr"]["peak_reduction_pct"] > 0


def test_drift_detects_shift():
    ref = np.random.default_rng(0).normal(100, 5, 50)
    dm = DriftMonitor(ref)
    assert dm.check(ref + 80)["drift_detected"] is True
    assert dm.check(ref + np.random.default_rng(1).normal(0, 0.1, 50))["drift_detected"] is False
