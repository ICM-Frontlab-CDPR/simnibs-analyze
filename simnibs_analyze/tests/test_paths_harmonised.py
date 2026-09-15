"""Paths use the same key names in the analyse and viz schemas.

They used to diverge (simnibs_simu/simnibs_preps/results_dir on one side,
sim_base/seg_base/out_root on the other) for the same three folders, so a path
block could not be moved between the two configs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from simnibs_analyze._config_schema_analyze import PipelineConfig
from simnibs_analyze._config_schema_viz import PathsConfig as VizPaths


class TestNamesMatchViz:
    def test_canonical_keys_are_the_viz_keys(self) -> None:
        viz_keys = set(VizPaths.model_fields)
        analyse_keys = set(PipelineConfig.model_fields["paths"].annotation.model_fields)
        assert viz_keys <= analyse_keys

    def test_new_names_accepted(self, minimal_analyze_cfg: dict) -> None:
        cfg = PipelineConfig.model_validate(minimal_analyze_cfg)
        assert cfg.paths.sim_base is not None
        assert cfg.paths.seg_base is not None
        assert cfg.paths.out_root is not None


class TestLegacyAliases:
    """Old configs must keep working — they are on people's disks."""

    def test_legacy_split_names_still_work(
        self, minimal_analyze_cfg: dict, tmp_path: Path
    ) -> None:
        minimal_analyze_cfg["paths"] = {
            "simnibs_preps": str(tmp_path / "preps"),
            "simnibs_simu": str(tmp_path / "simu"),
            "results_dir": str(tmp_path / "results"),
        }
        cfg = PipelineConfig.model_validate(minimal_analyze_cfg)
        assert cfg.paths.seg_base == tmp_path / "preps"
        assert cfg.paths.sim_base == tmp_path / "simu"
        assert cfg.paths.out_root == tmp_path / "results"

    def test_legacy_single_root_still_works(
        self, minimal_analyze_cfg: dict, tmp_path: Path
    ) -> None:
        minimal_analyze_cfg["paths"] = {
            "simnibs_output": str(tmp_path / "simnibs"),
            "results_dir": str(tmp_path / "results"),
        }
        cfg = PipelineConfig.model_validate(minimal_analyze_cfg)
        assert cfg.paths.out_root == tmp_path / "results"

    def test_new_name_wins_over_alias(
        self, minimal_analyze_cfg: dict, tmp_path: Path
    ) -> None:
        minimal_analyze_cfg["paths"] = {
            "sim_base": str(tmp_path / "new"),
            "simnibs_simu": str(tmp_path / "old"),
            "seg_base": str(tmp_path / "preps"),
            "out_root": str(tmp_path / "results"),
        }
        cfg = PipelineConfig.model_validate(minimal_analyze_cfg)
        assert cfg.paths.sim_base == tmp_path / "new"


class TestRequiredPaths:
    def test_out_root_required(self, minimal_analyze_cfg: dict, tmp_path: Path) -> None:
        minimal_analyze_cfg["paths"] = {"sim_base": str(tmp_path / "simu")}
        with pytest.raises(ValidationError, match="out_root"):
            PipelineConfig.model_validate(minimal_analyze_cfg)

    def test_some_input_path_required(
        self, minimal_analyze_cfg: dict, tmp_path: Path
    ) -> None:
        minimal_analyze_cfg["paths"] = {"out_root": str(tmp_path / "results")}
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(minimal_analyze_cfg)


class TestDeadTemplateKeysRemoved:
    """mni_template and mni_brain_mask were accepted then never read."""

    @pytest.mark.parametrize("key", ["mni_template", "mni_brain_mask"])
    def test_no_longer_declared(self, key: str) -> None:
        paths_model = PipelineConfig.model_fields["paths"].annotation
        assert key not in paths_model.model_fields
