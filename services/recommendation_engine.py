"""
Couche SERVICES — moteur de recommandation.

Transforme les sorties du contrôle optimal en décisions actionnables :
timing d'intervention, seuils d'alerte, niveau de risque (basé sur R0), et
message synthétique pour le décideur.
"""

from __future__ import annotations

import numpy as np

from domain.seir import SEIRParams
from domain.epidemiology import basic_reproduction_number


class RecommendationEngine:
    def __init__(self, params: SEIRParams):
        self.params = params

    def recommend(self, controlled_traj: dict, horizon: float = 180.0) -> dict:
        t = np.array(controlled_traj["t"])
        I = np.array(controlled_traj["I"])
        peak_day = int(t[int(np.argmax(I))])

        r0 = basic_reproduction_number(self.params)
        alert_threshold = 0.01 * self.params.N
        first_alert = next((int(t[i]) for i, v in enumerate(I) if v > alert_threshold), None)

        if r0 >= 2.5:
            risk = "élevé"
        elif r0 >= 1.3:
            risk = "modéré"
        else:
            risk = "faible"

        start = max(0, (first_alert or peak_day) - 14)
        return {
            "R0": round(r0, 2),
            "risk_level": risk,
            "intervention_start_day": start,
            "projected_peak_day": peak_day,
            "alert_threshold_I": alert_threshold,
            "advice": (
                f"Risque {risk} (R0={r0:.2f}). Déclencher vaccination + confinement "
                f"modéré vers J+{start}, soit ~2 semaines avant le seuil d'alerte, "
                f"pour aplatir le pic projeté vers J+{peak_day}."
            ),
        }
