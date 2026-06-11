"""
Couche DOMAINE — modèle épidémique SEIR contrôlé.

Compartiments S, E, I, R. Contrôles u1 (vaccination), u2 (confinement).
    Ṡ = -beta*(1-u2)*S*I/N - u1*S
    Ė =  beta*(1-u2)*S*I/N - sigma*E
    İ =  sigma*E - gamma*I
    Ṙ =  gamma*I + u1*S

Couche pure (aucune dépendance ML/API) : c'est le « noyau » du système hexagonal.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class SEIRParams:
    beta: float = 0.55
    sigma: float = 1 / 5.2
    gamma: float = 1 / 10.0
    N: float = 1.0e6

    def as_dict(self) -> dict:
        return asdict(self)


ControlPolicy = Callable[[float, np.ndarray], Sequence[float]]


def no_control(t: float, x: np.ndarray) -> tuple[float, float]:
    return 0.0, 0.0


def constant_control(u1: float, u2: float) -> ControlPolicy:
    def _p(t: float, x: np.ndarray) -> tuple[float, float]:
        return float(u1), float(u2)

    return _p


def seir_rhs(t, x, params: SEIRParams, policy: ControlPolicy = no_control) -> np.ndarray:
    S, E, I, R = x
    u1, u2 = policy(t, x)
    u1 = float(np.clip(u1, 0.0, 1.0))
    u2 = float(np.clip(u2, 0.0, 1.0))
    new_inf = params.beta * (1.0 - u2) * S * I / params.N
    return np.array(
        [
            -new_inf - u1 * S,
            new_inf - params.sigma * E,
            params.sigma * E - params.gamma * I,
            params.gamma * I + u1 * S,
        ],
        dtype=float,
    )


def simulate(
    params: SEIRParams,
    x0: np.ndarray,
    t_span: tuple[float, float] = (0.0, 180.0),
    n_points: int = 181,
    policy: ControlPolicy = no_control,
) -> dict:
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    # LSODA (bascule auto stiff/non-stiff) : indispensable pour les systèmes CONTRÔLÉS,
    # qui deviennent raides aux grandes populations (RK45 y prend des pas minuscules :
    # ~50 s à N=37M contre ~0 s ici). atol absolu adapté à des effectifs de personnes.
    sol = solve_ivp(
        fun=lambda t, x: seir_rhs(t, x, params, policy),
        t_span=t_span, y0=np.asarray(x0, float), t_eval=t_eval,
        method="LSODA", rtol=1e-6, atol=1e-3,
    )
    if not sol.success:
        raise RuntimeError(f"Intégration SEIR échouée : {sol.message}")
    return {
        "t": sol.t.tolist(),
        "S": sol.y[0].tolist(), "E": sol.y[1].tolist(),
        "I": sol.y[2].tolist(), "R": sol.y[3].tolist(),
    }
