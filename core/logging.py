"""Logging structuré et exceptions du domaine (couche transversale)."""

from __future__ import annotations

import logging
import sys

from core.config import settings


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(settings.log_level)
        logger.propagate = False
    return logger


class PlatformError(Exception):
    """Erreur générique de la plateforme."""


class CalibrationError(PlatformError):
    """Échec de calibration des paramètres épidémiques."""


class DataValidationError(PlatformError):
    """Données d'entrée invalides."""
