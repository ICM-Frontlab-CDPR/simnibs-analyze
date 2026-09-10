"""Tests for the analysis PipelineConfig schema."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from simnibs_analyze._config_schema_analyze import (
    AtlasROI,
    PipelineConfig,
    SphereROI,
    load_and_validate,
)


class TestROIDiscrimination:
    def test_sphere_roi_parsed(self, minimal_analyze_cfg: dict) -> None:
        cfg = PipelineConfig.model_validate(minimal_analyze_cfg)
        assert isinstance(cfg.target_generation.rois["fef"], SphereROI)

    def test_atlas_roi_parsed(self, minimal_analyze_cfg: dict) -> None:
        minimal_analyze_cfg["target_generation"]["rois"]["fef"] = {
            "method": "atlas",
            "atlas": "harvard-oxford",
            "regions": "Frontal Pole",
        }
        cfg = PipelineConfig.model_validate(minimal_analyze_cfg)
        assert isinstance(cfg.target_generation.rois["fef"], AtlasROI)

    def test_sphere_requires_three_coords(self, minimal_analyze_cfg: dict) -> None:
        minimal_analyze_cfg["target_generation"]["rois"]["fef"]["coords"] = [1.0, 2.0]
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(minimal_analyze_cfg)

    def test_unknown_atlas_rejected(self, minimal_analyze_cfg: dict) -> None:
        minimal_analyze_cfg["target_generation"]["rois"]["fef"] = {
            "method": "atlas",
            "atlas": "not-an-atlas",
            "regions": "X",
        }
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(minimal_analyze_cfg)


class TestROINaming:
    def test_underscore_in_roi_name_rejected(self, minimal_analyze_cfg: dict) -> None:
        rois = minimal_analyze_cfg["target_generation"]["rois"]
        rois["ips_left"] = rois.pop("fef")
        minimal_analyze_cfg["stim_conditions"] = ["ips_left"]
        with pytest.raises(ValidationError, match="underscores"):
            PipelineConfig.model_validate(minimal_analyze_cfg)

    def test_hyphen_in_roi_name_accepted(self, minimal_analyze_cfg: dict) -> None:
        rois = minimal_analyze_cfg["target_generation"]["rois"]
        rois["ips-left"] = rois.pop("fef")
        minimal_analyze_cfg["stim_conditions"] = ["ips-left"]
        cfg = PipelineConfig.model_validate(minimal_analyze_cfg)
        assert "ips-left" in cfg.target_generation.rois


class TestStimConditionsConsistency:
    def test_condition_without_roi_rejected(self, minimal_analyze_cfg: dict) -> None:
        minimal_analyze_cfg["stim_conditions"] = ["fef", "orphan"]
        with pytest.raises(ValidationError, match="orphan"):
            PipelineConfig.model_validate(minimal_analyze_cfg)


class TestPathsConfig:
    def test_legacy_single_root_accepted(self, minimal_analyze_cfg: dict) -> None:
        cfg = PipelineConfig.model_validate(minimal_analyze_cfg)
        assert cfg.paths.simnibs_output is not None

    def test_split_paths_accepted(self, minimal_analyze_cfg: dict, tmp_path: Path) -> None:
        minimal_analyze_cfg["paths"] = {
            "simnibs_preps": str(tmp_path / "preps"),
            "simnibs_simu": str(tmp_path / "simu"),
            "results_dir": str(tmp_path / "results"),
        }
        cfg = PipelineConfig.model_validate(minimal_analyze_cfg)
        assert cfg.paths.simnibs_preps is not None

    def test_no_input_path_rejected(self, minimal_analyze_cfg: dict, tmp_path: Path) -> None:
        minimal_analyze_cfg["paths"] = {"results_dir": str(tmp_path / "results")}
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(minimal_analyze_cfg)

    def test_partial_split_rejected(self, minimal_analyze_cfg: dict, tmp_path: Path) -> None:
        """simnibs_preps alone is not enough — simnibs_simu is also required."""
        minimal_analyze_cfg["paths"] = {
            "simnibs_preps": str(tmp_path / "preps"),
            "results_dir": str(tmp_path / "results"),
        }
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(minimal_analyze_cfg)


class TestDefaults:
    def test_defaults_applied(self, minimal_analyze_cfg: dict) -> None:
        cfg = PipelineConfig.model_validate(minimal_analyze_cfg)
        assert cfg.space == "mni"
        assert cfg.running.if_exists == "skip"
        assert cfg.preprocessing.smooth_fwhm == 2.0
        assert cfg.analysis.clustering.specificity_threshold == 1.5

    def test_invalid_mode_rejected(self, minimal_analyze_cfg: dict) -> None:
        minimal_analyze_cfg["mode"] = ["teleportation"]
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(minimal_analyze_cfg)

    def test_negative_smooth_rejected(self, minimal_analyze_cfg: dict) -> None:
        minimal_analyze_cfg["preprocessing"] = {"smooth_fwhm": -1.0}
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(minimal_analyze_cfg)


class TestLoadAndValidate:
    def test_roundtrip_from_yaml(self, minimal_analyze_cfg: dict, tmp_path: Path) -> None:
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(minimal_analyze_cfg))
        cfg = load_and_validate(p)
        assert cfg.subjects == ["0001"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_and_validate(tmp_path / "nope.yaml")
