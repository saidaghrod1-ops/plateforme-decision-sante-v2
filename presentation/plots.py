"""Figures de benchmark stylées (échelle log) — source unique de vérité.

Le dashboard Streamlit ET les scripts de slides/rapport produisent EXACTEMENT
la même figure : couleurs par stratégie, axe I(t) en log (clip à 1 pour rester
lisible quand le contrôle élimine l'épidémie), titre + R0, légende et grille.

    from presentation.plots import benchmark_figure
    fig = benchmark_figure(details, ["baseline", "lqr", "pmp", "hjb"], "Benchmark …")
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

# Palette et libellés communs (cohérence dashboard <-> slides <-> rapport).
COLORS = {"baseline": "#E74C3C", "lqr": "#F39C12", "pmp": "#149E8E", "hjb": "#7C3AED"}
LABELS = {"baseline": "Baseline", "lqr": "LQR", "pmp": "PMP", "hjb": "HJB-PINN"}

_STYLE = {
    "font.size": 13,
    "axes.edgecolor": "#14304A", "axes.labelcolor": "#14304A",
    "text.color": "#14304A", "xtick.color": "#14304A", "ytick.color": "#14304A",
}


# Couleurs des compartiments SEIR (fig. scénario).
SEIR_COLORS = {"S": "#2E86C1", "E": "#F39C12", "I": "#E74C3C", "R": "#2ECC71"}


def benchmark_figure(details: dict, keys: Iterable[str], title: str):
    """Construit la figure de benchmark à partir des `details` de BenchmarkService.

    details : mapping {stratégie -> {"trajectory": {"t", "I", ...}, ...}}.
    keys    : ordre d'affichage des stratégies (les absentes/échouées sont ignorées).
    title   : titre (inclut usuellement le R0, ex. « … (R0=5.5) »).
    Retourne une figure matplotlib (au caller de l'afficher / la sauvegarder).
    """
    import matplotlib.pyplot as plt

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7.3, 4.2), dpi=170)
        for k in keys:
            d = details.get(k, {})
            if "trajectory" not in d:
                continue  # stratégie échouée (ex. HJB sans torch) : ignorée silencieusement
            # clip à 1 : sous l'échelle log, une élimination (I -> 0) reste lisible
            # et plafonne proprement le « plancher d'un infecté ».
            infected = np.clip(np.asarray(d["trajectory"]["I"], float), 1.0, None)
            ax.plot(d["trajectory"]["t"], infected,
                    label=LABELS.get(k, k), color=COLORS.get(k), lw=2.5)
        ax.set_yscale("log")
        ax.set_xlabel("Jours")
        ax.set_ylabel("Infectés I(t) — log")
        ax.set_title(title, fontweight="bold")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25, which="both")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
    return fig


def seir_figure(traj: dict, title: str, log: bool = False):
    """Trajectoire des 4 compartiments S, E, I, R d'un scénario."""
    import matplotlib.pyplot as plt

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7.3, 4.2), dpi=170)
        t = traj["t"]
        for comp in "SEIR":
            y = np.asarray(traj[comp], float)
            if log:
                y = np.clip(y, 1.0, None)
            ax.plot(t, y, label=comp, color=SEIR_COLORS[comp], lw=2.5)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("Jours")
        ax.set_ylabel("Population" + (" — log" if log else ""))
        ax.set_title(title, fontweight="bold")
        ax.legend(frameon=False, ncol=4)
        ax.grid(alpha=0.25, which="both")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
    return fig


def controls_figure(details: dict, keys: Iterable[str], title: str):
    """Trajectoires de contrôle u1 (vaccination) / u2 (confinement) par stratégie."""
    import matplotlib.pyplot as plt

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(7.3, 5.2), dpi=170, sharex=True)
        for k in keys:
            d = details.get(k, {})
            if "controls" not in d or "trajectory" not in d:
                continue
            t = d["trajectory"]["t"]
            axes[0].plot(t, d["controls"]["u1"], label=LABELS.get(k, k), color=COLORS.get(k), lw=2.2)
            axes[1].plot(t, d["controls"]["u2"], color=COLORS.get(k), lw=2.2)
        axes[0].set_ylabel("u₁ — vaccination")
        axes[1].set_ylabel("u₂ — confinement")
        axes[1].set_xlabel("Jours")
        axes[0].set_title(title, fontweight="bold")
        axes[0].legend(frameon=False, ncol=4)
        for ax in axes:
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
    return fig


def reff_figure(traj: dict, R0: float, N: float, title: str):
    """Taux de reproduction effectif R_eff(t) = R0 · S(t)/N, seuil épidémique à 1."""
    import matplotlib.pyplot as plt

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7.3, 3.6), dpi=170)
        S = np.asarray(traj["S"], float)
        reff = R0 * S / N
        ax.plot(traj["t"], reff, color="#7C3AED", lw=2.5, label="R_eff(t)")
        ax.axhline(1.0, color="#E74C3C", ls="--", lw=1.5, label="Seuil épidémique (R_eff = 1)")
        ax.set_xlabel("Jours")
        ax.set_ylabel("R_eff")
        ax.set_title(title, fontweight="bold")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
    return fig


def fit_figure(t_obs, I_obs, model_traj: dict, title: str, t_max: float | None = None):
    """Données observées (points) vs modèle SEIR calibré (ligne), échelle log."""
    import matplotlib.pyplot as plt

    t_obs = np.asarray(t_obs, float)
    I_obs = np.asarray(I_obs, float)
    tm = np.asarray(model_traj["t"], float)
    im = np.clip(np.asarray(model_traj["I"], float), 1.0, None)
    if t_max is not None:
        mo, mm = t_obs <= t_max, tm <= t_max
        t_obs, I_obs, tm, im = t_obs[mo], I_obs[mo], tm[mm], im[mm]
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7.3, 4.2), dpi=170)
        ax.semilogy(t_obs, np.clip(I_obs, 1.0, None), "o", ms=4, color="#E74C3C",
                    alpha=0.7, label="Observé")
        ax.semilogy(tm, im, color="#149E8E", lw=2.5, label="Modèle SEIR calibré")
        ax.set_xlabel("Jours")
        ax.set_ylabel("Infectés I(t) — log")
        ax.set_title(title, fontweight="bold")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25, which="both")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
    return fig
