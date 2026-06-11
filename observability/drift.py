"""
Couche OBSERVABILITÉ — détection de dérive (data drift).

Compare la distribution des nouvelles observations à une référence (test de
Kolmogorov-Smirnov sur le compartiment I normalisé). Au-delà d'un seuil, signale
qu'une recalibration est nécessaire — brique clé d'une plateforme en production.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from core.logging import get_logger

log = get_logger("observability.drift")


class DriftMonitor:
    def __init__(self, reference_I: np.ndarray, p_threshold: float = 0.05):
        self.reference = np.asarray(reference_I, float)
        self.p_threshold = p_threshold

    def check(self, new_I: np.ndarray) -> dict:
        new_I = np.asarray(new_I, float)
        if len(new_I) < 5 or len(self.reference) < 5:
            return {"drift_detected": False, "reason": "échantillon insuffisant"}

        stat, p_value = stats.ks_2samp(self.reference, new_I)
        drift = bool(p_value < self.p_threshold)
        if drift:
            log.warning("Dérive détectée (p=%.4f) -> recalibration conseillée", p_value)
        return {
            "drift_detected": drift,
            "ks_statistic": float(stat),
            "p_value": float(p_value),
            "action": "recalibrer le modèle" if drift else "aucune action",
        }
