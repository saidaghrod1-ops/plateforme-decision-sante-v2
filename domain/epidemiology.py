"""
Couche DOMAINE — indicateurs épidémiologiques dérivés du modèle SEIR.

R0, taux de reproduction effectif, point d'équilibre pour la linéarisation LQR,
et métriques de trajectoire (pic, jour du pic, attaque finale).
"""

from __future__ import annotations

import numpy as np

from domain.seir import SEIRParams


def basic_reproduction_number(p: SEIRParams) -> float:
    """R0 = beta / gamma pour un modèle SEIR à population fermée."""
    return p.beta / p.gamma


def effective_reproduction_number(p: SEIRParams, S: float) -> float:
    """R_eff = R0 * S / N."""
    return basic_reproduction_number(p) * S / p.N


def endemic_equilibrium(p: SEIRParams) -> np.ndarray:
    """Point d'opération (S*, E*, I*, R*) pour la linéarisation locale."""
    N = p.N
    I_star = 0.005 * N
    E_star = p.gamma / p.sigma * I_star
    S_star = p.gamma * N / p.beta
    R_star = N - S_star - E_star - I_star
    return np.array([S_star, E_star, I_star, R_star], float)


def trajectory_metrics(traj: dict) -> dict:
    """Extrait les indicateurs clés d'une trajectoire simulée."""
    t = np.array(traj["t"])
    I = np.array(traj["I"])
    R = np.array(traj["R"])
    peak_idx = int(np.argmax(I))
    return {
        "peak_I": float(I[peak_idx]),
        "peak_day": float(t[peak_idx]),
        "final_attack_rate": float(R[-1]),
        "total_infected_proxy": float(np.trapezoid(I, t)),
    }
