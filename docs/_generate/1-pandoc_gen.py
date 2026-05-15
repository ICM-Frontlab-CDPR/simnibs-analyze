#!/usr/bin/env python3
"""
Étape 1 — Génération de la référence API avec pdoc.

Génère docs/api/ à partir du package simnibs_analyze.
Ne touche pas à docs/assets/ ni à docs/index.html.

Usage (direct):        python docs/_generate/1-pandoc_gen.py
Usage (orchestrateur): python generate_docs_for_githubPages.py

Dépendance: pip install pdoc
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent   # simnibs-analyze/
PIPELINE_DIR = ROOT / "simnibs_analyze"
API_DIR = ROOT / "docs" / "api"

MODULES = ["simnibs_analyze"]


def run() -> None:
    try:
        import pdoc  # noqa: F401
    except ImportError:
        print("pdoc non trouvé — installation...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pdoc"])

    # Nettoyer uniquement docs/api/ (pas les assets ni index.html)
    if API_DIR.exists():
        shutil.rmtree(API_DIR)
    API_DIR.mkdir(parents=True, exist_ok=True)

    pythonpath = str(ROOT) + os.pathsep + str(PIPELINE_DIR)
    env = {**os.environ, "PYTHONPATH": pythonpath}

    subprocess.check_call(
        [sys.executable, "-m", "pdoc", "--output-dir", str(API_DIR)] + MODULES,
        env=env,
    )

    print(f"\n✓ [1-pandoc_gen] API docs générées dans : {API_DIR}")
    print(f"   Entrée : {API_DIR / 'simnibs_analyze.html'}")


if __name__ == "__main__":
    run()
