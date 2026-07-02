#!/usr/bin/env python3
"""
Orchestrateur de génération de la documentation GitHub Pages.

Appelle successivement :
  1. docs/_generate/1-pandoc_gen.py  → génère docs/api/ via pdoc
  2. docs/_generate/2-supplementary.py → génère docs/index.html (landing page)

Ne supprime jamais docs/assets/.

Usage:
    cd simnibs-analyze
    python generate_docs_for_githubPages.py

Dépendance: pip install pdoc
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parent
GENERATE_DIR = ROOT / "docs" / "_generate"

STEPS = [
    ("1-pandoc_gen", "Génération API pdoc → docs/api/"),
    ("2-supplementary", "Landing page + .nojekyll → docs/index.html"),
]

print("=" * 60)
print("  simnibs-analyze — Génération GitHub Pages")
print("=" * 60)

for module_name, description in STEPS:
    print(f"\n── {description}")
    spec = importlib.util.spec_from_file_location(
        module_name, GENERATE_DIR / f"{module_name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.run()

print("\n" + "=" * 60)
print("✓ Documentation complète prête dans docs/")
print("  GitHub Pages : docs/index.html")
print("  API ref      : docs/api/simnibs_analyze.html")
print("=" * 60)
