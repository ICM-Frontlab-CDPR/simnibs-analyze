"""Guard rails on packaging and portability.

These tests exist to catch regressions that broke previous releases:
machine-specific paths, dead-code directories shipped to PyPI, and version
strings drifting apart across metadata files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import simnibs_analyze

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = Path(simnibs_analyze.__file__).parent

ACTIVE_DIRS_SKIP = {"__old__", "__later__", "__TODO", "tests", "__pycache__"}


def _active_sources() -> list[Path]:
    return [
        p
        for p in PKG_ROOT.rglob("*.py")
        if not any(part in ACTIVE_DIRS_SKIP for part in p.parts)
    ]


class TestNoMachineSpecificPaths:
    def test_no_absolute_user_paths(self) -> None:
        offenders = [
            p.name for p in _active_sources() if "/Users/" in p.read_text() or "/home/" in p.read_text()
        ]
        assert offenders == []

    def test_no_sys_path_manipulation(self) -> None:
        offenders = [p.name for p in _active_sources() if "sys.path.insert" in p.read_text()]
        assert offenders == []


class TestVersionConsistency:
    def test_pyproject_matches_dunder_version(self) -> None:
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        assert f'version = "{simnibs_analyze.__version__}"' in pyproject

    def test_citation_cff_matches(self) -> None:
        cff = (REPO_ROOT / "CITATION.cff").read_text()
        assert f"version: {simnibs_analyze.__version__}" in cff

    def test_zenodo_json_is_valid_and_matches(self) -> None:
        data = json.loads((REPO_ROOT / ".zenodo.json").read_text())
        assert data["version"] == simnibs_analyze.__version__


class TestDependencyDeclared:
    def test_simnibs_reader_is_a_declared_dependency(self) -> None:
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        assert "simnibs-reader>=" in pyproject

    def test_reader_importable_at_required_version(self) -> None:
        import simnibs_reader

        major, minor = (int(x) for x in simnibs_reader.__version__.split(".")[:2])
        assert (major, minor) >= (0, 2)


class TestLazyImportsDeclared:
    """Imports inside function bodies are easy to miss when auditing
    dependencies - playwright shipped undeclared because of exactly this."""

    THIRD_PARTY_LAZY: ClassVar[dict[str, str | None]] = {
        "playwright": "viz3d",
        "yaml": None,      # core dependency
        "scipy": None,     # core dependency
        "duecredit": "citations",
    }

    def test_every_lazy_third_party_import_is_declared(self) -> None:
        import re

        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        pattern = re.compile(r"^\s+(?:from|import)\s+([a-zA-Z_][\w]*)", re.MULTILINE)

        found: set[str] = set()
        for path in _active_sources():
            for mod in pattern.findall(path.read_text()):
                if mod in self.THIRD_PARTY_LAZY:
                    found.add(mod)

        undeclared = [m for m in found if m.replace("yaml", "pyyaml") not in pyproject]
        assert undeclared == [], f"lazy-imported but not in pyproject: {undeclared}"

    def test_playwright_is_an_optional_extra(self) -> None:
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        assert "viz3d = [" in pyproject
        assert "playwright" in pyproject

    def test_missing_playwright_raises_actionable_error(self) -> None:
        """The error must name both install steps, not just the pip one."""
        src = (PKG_ROOT / "steps" / "viz.py").read_text()
        assert "simnibs-analyze[viz3d]" in src
        assert "playwright install chromium" in src
