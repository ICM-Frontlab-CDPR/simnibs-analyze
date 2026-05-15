#!/usr/bin/env python3
"""
Étape 2 — Génération de la landing page et vérifications supplémentaires.

- Crée docs/index.html (landing page custom pointant vers docs/api/)
- Crée docs/.nojekyll si absent
- Ne touche pas à docs/assets/ ni à docs/api/

Usage (direct):        python docs/_generate/2-supplementary.py
Usage (orchestrateur): python generate_docs_for_githubPages.py
"""
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent   # simnibs-analyze/
DOCS_DIR = ROOT / "docs"
EXAMPLES_SRC = ROOT / "simnibs_analyze" / "examples"


def _yaml_to_html_card(name: str, content: str) -> str:
    """Retourne un bloc HTML <details> avec le contenu YAML coloré via <pre>."""
    safe = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""
    <details open>
      <summary><strong>{name}</strong></summary>
      <pre><code class="language-yaml">{safe}</code></pre>
    </details>
"""


def generate_examples_page() -> None:
    """Génère docs/examples/index.html à partir des fichiers YAML d'exemples."""
    examples_dir = DOCS_DIR / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    yaml_files = sorted(EXAMPLES_SRC.glob("*.yaml"))
    if not yaml_files:
        print("  ⚠ Aucun fichier YAML trouvé dans simnibs_analyze/examples/")
        return

    cards_html = "".join(
        _yaml_to_html_card(f.name, f.read_text(encoding="utf-8"))
        for f in yaml_files
    )

    page = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Exemples — simnibs-analyze</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 3rem auto; padding: 0 1.5rem; color: #222; }}
    a.back {{ font-size: .9rem; color: #555; text-decoration: none; }}
    a.back:hover {{ text-decoration: underline; }}
    h1 {{ margin: .5rem 0 .25rem; }}
    p.sub {{ color: #555; margin-top: 0; }}
    details {{ border: 1px solid #ddd; border-radius: 8px; margin: 1.5rem 0; padding: .75rem 1rem; }}
    summary {{ cursor: pointer; font-size: 1.05rem; padding: .25rem 0; }}
    pre {{ background: #f6f8fa; border-radius: 6px; padding: 1rem; overflow-x: auto; font-size: .85rem; margin: .75rem 0 0; }}
    code {{ font-family: "SFMono-Regular", Consolas, monospace; }}
    footer {{ margin-top: 3rem; font-size: .8rem; color: #999; }}
  </style>
</head>
<body>
  <a class="back" href="../index.html">← Accueil</a>
  <h1>Exemples de configuration</h1>
  <p class="sub">Fichiers <code>config.yaml</code> d'exemple pour démarrer rapidement.</p>
  {cards_html}
  <footer>Généré automatiquement — simnibs-analyze</footer>
</body>
</html>
"""

    out = examples_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    print(f"  ✓ Page exemples écrite : {out} ({len(yaml_files)} fichiers YAML)")


LANDING_HTML = """\
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>simnibs-analyze — Documentation</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 800px; margin: 4rem auto; padding: 0 1.5rem; color: #222; }
    h1 { font-size: 2rem; margin-bottom: .25rem; }
    p.sub { color: #555; margin-top: 0; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-top: 2rem; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 1.25rem 1.5rem; text-decoration: none; color: inherit; transition: box-shadow .15s; }
    .card:hover { box-shadow: 0 4px 12px rgba(0,0,0,.12); }
    .card h2 { font-size: 1.1rem; margin: 0 0 .4rem; }
    .card p { font-size: .9rem; color: #555; margin: 0; }
    footer { margin-top: 3rem; font-size: .8rem; color: #999; }
  </style>
</head>
<body>
  <h1>simnibs-analyze</h1>
  <p class="sub">Pipeline d'analyse des champs électriques SimNIBS (tDCS / TMS)</p>

  <div class="cards">
    <a class="card" href="api/simnibs_analyze.html">
      <h2>📖 Référence API</h2>
      <p>Documentation auto-générée de tous les modules du package.</p>
    </a>
    <a class="card" href="examples/index.html">
      <h2>🗂 Exemples de config</h2>
      <p>Fichiers config.yaml annotés pour démarrer rapidement.</p>
    </a>
    <a class="card" href="assets/global-env.svg">
      <h2>🗺 Environnement global</h2>
      <p>Vue d'ensemble de l'architecture du pipeline.</p>
    </a>
  </div>

  <footer>Généré automatiquement — simnibs-analyze</footer>
</body>
</html>
"""


def run() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # .nojekyll — désactive Jekyll sur GitHub Pages
    nojekyll = DOCS_DIR / ".nojekyll"
    if not nojekyll.exists():
        nojekyll.touch()
        print("  ✓ .nojekyll créé")

    # Landing page — toujours régénérée
    index = DOCS_DIR / "index.html"
    index.write_text(LANDING_HTML, encoding="utf-8")
    print(f"  ✓ Landing page écrite : {index}")

    # Page d'exemples YAML
    generate_examples_page()

    print(f"\n✓ [2-supplementary] docs/ prêt pour GitHub Pages")


if __name__ == "__main__":
    run()
