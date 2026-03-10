#!/usr/bin/env python3
"""
Génère la documentation HTML des modules principaux du pipeline.

Usage:
    cd simnibs-pipeline
    python generate_docs.py

Sortie: docs/api/  (fichiers HTML navigables)
Dépendance: pip install pdoc
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PIPELINE_DIR = ROOT / "simnibs_pipeline"
OUTPUT_DIR = ROOT / "docs" / "api"

# Documenter le package entier — __init__.__all__ contrôle quels sous-modules
# sont inclus (les 6 modules principaux uniquement).
MODULES = ["simnibs_pipeline"]

try:
    import pdoc  # noqa: F401
except ImportError:
    print("pdoc non trouvé — installation...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pdoc"])

# Nettoyage de l'ancienne génération pour éviter les index périmés
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

# ROOT must be on PYTHONPATH so `simnibs_pipeline` is importable as a package.
# PIPELINE_DIR must also be on PYTHONPATH so flat `from _pipeline_io import ...`
# inside the modules resolves correctly.
pythonpath = str(ROOT) + os.pathsep + str(PIPELINE_DIR)
env = {**os.environ, "PYTHONPATH": pythonpath}

subprocess.check_call(
    [sys.executable, "-m", "pdoc", "--output-dir", str(OUTPUT_DIR)] + MODULES,
    env=env,
)

print(f"\n✓ Documentation générée dans : {OUTPUT_DIR}")
print(f"  Ouvrir : {OUTPUT_DIR / 'simnibs_pipeline.html'}")
