"""
Lanceur API auto-ancré (isolation des projets).

Démarre l'API de CE projet quel que soit le répertoire courant ou le PYTHONPATH
ambiant — `uvicorn api.main:app` résout le package `api` avant d'exécuter le code,
donc on force la racine de ce projet en tête de sys.path ici.

    python run_api.py            # http://127.0.0.1:8010
    python run_api.py --port 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)  # port distinct de la v1 (8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()
    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.reload)
