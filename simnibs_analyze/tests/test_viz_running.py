"""The viz schema has a top-level `running.if_exists`, like the analyse one.

It previously had only a per-figure `if_exists`, carrying a
`# TODO enforce in viz` — the field was declared, stored on SimnibsViz, and
never read, so figures were always re-rendered.
"""

from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from simnibs_analyze import run_viz
from simnibs_analyze._config_schema_viz import RunningConfig, VizConfig


@pytest.fixture
def viz_cfg(tmp_path) -> dict:
    return {
        "subjects": {"ids": ["0001"], "stim_pattern": "AFFT"},
        "paths": {
            "sim_base": str(tmp_path / "simu"),
            "seg_base": str(tmp_path / "preps"),
            "out_root": str(tmp_path / "out"),
        },
        "fields_scale": {"min": 0.05, "max": 0.4},
        "vols": {"field_E": {"kind": "field", "name": "magnE"}},
        "figures": [{"name": "f1", "type": "2D", "vols": ["field_E"]}],
    }


class TestRunningBlock:
    def test_default_matches_analyse_schema(self) -> None:
        assert RunningConfig().if_exists == "skip"

    def test_absent_running_block_is_fine(self, viz_cfg: dict) -> None:
        cfg = VizConfig.model_validate(viz_cfg)
        assert cfg.running.if_exists == "skip"

    def test_explicit_value_applied(self, viz_cfg: dict) -> None:
        viz_cfg["running"] = {"if_exists": "overwrite"}
        cfg = VizConfig.model_validate(viz_cfg)
        assert cfg.running.if_exists == "overwrite"

    def test_invalid_value_rejected(self, viz_cfg: dict) -> None:
        viz_cfg["running"] = {"if_exists": "maybe"}
        with pytest.raises(ValidationError):
            VizConfig.model_validate(viz_cfg)

    def test_unknown_key_rejected(self, viz_cfg: dict) -> None:
        viz_cfg["running"] = {"if_exist": "skip"}  # typo
        with pytest.raises(ValidationError):
            VizConfig.model_validate(viz_cfg)


class TestPerFigureOverride:
    def test_figure_inherits_global(self, viz_cfg: dict) -> None:
        viz_cfg["running"] = {"if_exists": "overwrite"}
        cfg = VizConfig.model_validate(viz_cfg)
        assert cfg.if_exists_for(cfg.figures[0]) == "overwrite"

    def test_figure_overrides_global(self, viz_cfg: dict) -> None:
        viz_cfg["running"] = {"if_exists": "overwrite"}
        viz_cfg["figures"][0]["if_exists"] = "error"
        cfg = VizConfig.model_validate(viz_cfg)
        assert cfg.if_exists_for(cfg.figures[0]) == "error"


class TestActuallyEnforced:
    """The TODO said 'enforce in viz' — this is the guard that it happened."""

    def test_render_figure_checks_the_policy(self) -> None:
        src = inspect.getsource(run_viz.render_figure)
        assert "if_exists_for" in src
        assert "out_png.exists()" in src

    def test_skip_returns_before_rendering(self) -> None:
        src = inspect.getsource(run_viz.render_figure)
        skip_at = src.index('policy == "skip"')
        render_at = src.index("resolve_layer")
        assert skip_at < render_at, "the existence check must short-circuit the work"
