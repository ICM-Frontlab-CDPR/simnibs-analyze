"""Tests for the two console entry points.

These guard the public CLI contract:
    simnibs-analyze --config <yaml>
    simnibs-viz     --config <yaml>
"""

from __future__ import annotations

import importlib.metadata as md
from pathlib import Path

import pytest

from simnibs_analyze import run_analyze, run_viz


class TestEntryPointsDeclared:
    def test_both_scripts_registered(self) -> None:
        scripts = {
            ep.name: ep.value
            for ep in md.distribution("simnibs-analyze").entry_points
            if ep.group == "console_scripts"
        }
        assert scripts == {
            "simnibs-analyze": "simnibs_analyze.run_analyze:cli",
            "simnibs-viz": "simnibs_analyze.run_viz:cli",
        }

    def test_entry_points_are_callable(self) -> None:
        assert callable(run_analyze.cli)
        assert callable(run_viz.cli)


class TestVizCli:
    def test_config_is_required(self) -> None:
        with pytest.raises(SystemExit) as exc:
            run_viz.cli([])
        assert exc.value.code == 2

    def test_missing_config_file_exits(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exc:
            run_viz.cli(["--config", str(tmp_path / "absent.yaml")])
        assert exc.value.code == 2

    def test_help_exits_zero(self, capsys: pytest.CaptureFixture) -> None:
        with pytest.raises(SystemExit) as exc:
            run_viz.cli(["--help"])
        assert exc.value.code == 0
        assert "--config" in capsys.readouterr().out

    def test_positional_path_rejected(self, tmp_path: Path) -> None:
        """The old `run-viz.py <path>` form must not silently work anymore."""
        p = tmp_path / "cfg.yaml"
        p.write_text("{}")
        with pytest.raises(SystemExit):
            run_viz.cli([str(p)])


class TestAnalyzeCli:
    def test_config_is_required(self) -> None:
        with pytest.raises(SystemExit) as exc:
            run_analyze.cli([])
        assert exc.value.code == 2

    def test_help_exits_zero(self, capsys: pytest.CaptureFixture) -> None:
        with pytest.raises(SystemExit) as exc:
            run_analyze.cli(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "--config" in out
        assert "--skip-features" in out
