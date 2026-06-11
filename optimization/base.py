"""
Couche MOTEURS — interface unifiée des contrôleurs + fabrique.

Tous les contrôleurs (LQR, PMP, HJB-PINN) exposent la même interface `policy()`,
ce qui permet au service de benchmark de les comparer de façon interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from domain.seir import SEIRParams


class Controller(ABC):
    name: str = "abstract"

    @abstractmethod
    def policy(self):
        """Retourne une politique callable (t, x) -> (u1, u2) bornée dans [0,1]."""
        ...


def make_controller(kind: str, params: SEIRParams, **kwargs) -> Controller:
    """Fabrique : 'lqr' | 'pmp' | 'hjb'."""
    kind = kind.lower()
    if kind == "lqr":
        from optimization.lqr import LQRController
        return LQRController(params, **kwargs)
    if kind == "pmp":
        from optimization.pmp import PMPController
        return PMPController(params, **kwargs)
    if kind == "hjb":
        from optimization.hjb_pinn import HJBController
        return HJBController(params, **kwargs)
    raise ValueError(f"Contrôleur inconnu : {kind}")
