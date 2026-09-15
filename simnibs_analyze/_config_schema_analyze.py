"""
Pydantic models for pipeline config.yaml validation.

These models mirror the config.yaml structure exactly.
Usage (standalone check):
    python _config.py --config config.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# ROI definitions  (target_generation.rois)
# ---------------------------------------------------------------------------


class SphereROI(BaseModel):
    """ROI defined by a sphere centred on MNI coordinates."""

    method: Literal["sphere"]
    coords: Annotated[list[float], Field(min_length=3, max_length=3)]
    folder_pattern: str | None = None
    """Glob fragment used to find SimNIBS output folders for this ROI.
    If omitted, the ROI key name is used (e.g. 'fef' → 'simulation_simulation_fef_*').
    Set this when the folder name differs from the ROI key (e.g. ROI 'ips-left' but
    folders are named '…ips_left…' → folder_pattern: 'ips_left')."""


class AtlasROI(BaseModel):
    """ROI defined by one or more parcels from a brain atlas."""

    method: Literal["atlas"]
    atlas: Literal["harvard-oxford", "aal", "destrieux"]
    regions: str | list[str]
    folder_pattern: str | None = None
    """See SphereROI.folder_pattern."""


# Discriminated union: Pydantic inspects the `method` field to pick the right model.
ROIDef = Annotated[SphereROI | AtlasROI, Field(discriminator="method")]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


class TargetGenerationConfig(BaseModel):
    radius_mm: float = Field(default=10.0, gt=0)
    rois: dict[str, ROIDef]

    @field_validator("rois")
    @classmethod
    def _no_underscores_in_roi_names(cls, v: dict) -> dict:
        bad = [k for k in v if "_" in k]
        if bad:
            raise ValueError(
                f"ROI names must not contain underscores (use hyphens instead): {bad}"
            )
        return v


class PathsConfig(BaseModel):
    """Input and output locations.

    Key names match the visualisation schema so a single set of paths can be
    copied between an analyse config and a viz config unchanged.
    """

    sim_base: Path | None = None
    """Simulation / optimization results  →  {sub}/simulations/"""
    seg_base: Path | None = None
    """charm / segmentation results  →  {sub}/m2m_{sub}/"""
    out_root: Path | None = None
    """Output folder: CSVs, figures, statistics."""

    # ── Deprecated aliases, kept so existing configs keep working ──────────
    simnibs_simu: Path | None = None
    """Deprecated alias for sim_base."""
    simnibs_preps: Path | None = None
    """Deprecated alias for seg_base."""
    results_dir: Path | None = None
    """Deprecated alias for out_root."""
    simnibs_output: Path | None = None
    """Deprecated: one root holding both preps and simulations."""

    @model_validator(mode="after")
    def _apply_aliases(self) -> PathsConfig:
        """Fold the deprecated names into the canonical ones."""
        for canonical, legacy in (
            ("sim_base", "simnibs_simu"),
            ("seg_base", "simnibs_preps"),
            ("out_root", "results_dir"),
        ):
            if getattr(self, canonical) is None and getattr(self, legacy) is not None:
                object.__setattr__(self, canonical, getattr(self, legacy))
        return self

    @model_validator(mode="after")
    def _validate_input_paths(self) -> PathsConfig:
        if self.out_root is None:
            raise ValueError("'out_root' is required (deprecated alias: results_dir).")
        if self.sim_base is None and self.simnibs_output is None:
            raise ValueError(
                "Specify 'sim_base' + 'seg_base', or 'simnibs_output' (legacy)."
            )
        return self


class PreprocessingConfig(BaseModel):
    smooth_fwhm: float = Field(default=2.0, ge=0)
    outlier_method: Literal["iqr", "zscore"] = "iqr"
    portion: float | None = Field(default=None, gt=0, le=1)


class FeatureExtractionConfig(BaseModel):
    metrics: list[str] = ["mean", "median", "std", "min", "max"]


class ClusteringConfig(BaseModel):
    method: str = "mean"
    specificity_threshold: float = Field(default=1.5, gt=0)
    intensity_col: str = "mean"


class AnalysisConfig(BaseModel):
    metric: str = "mean"
    subject_col: str = "subject"
    condition_col: str = "condition"
    clustering: ClusteringConfig = ClusteringConfig()


class RunningConfig(BaseModel):
    if_exists: Literal["skip", "overwrite", "error"] = "skip"


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    subjects: list[str]
    stim_conditions: list[str]
    mode: list[Literal["simulation", "optimization"]]
    space: Literal["mni", "native"] = "mni"
    running: RunningConfig = RunningConfig()
    target_generation: TargetGenerationConfig
    paths: PathsConfig
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    feature_extraction: FeatureExtractionConfig = FeatureExtractionConfig()
    analysis: AnalysisConfig = AnalysisConfig()

    @model_validator(mode="after")
    def _stim_conditions_match_rois(self) -> PipelineConfig:
        """Every stim_condition must have a matching ROI key (used to find the mask file)."""
        roi_names = set(self.target_generation.rois)
        missing = [c for c in self.stim_conditions if c not in roi_names]
        if missing:
            raise ValueError(
                f"stim_conditions {missing} have no matching entry in target_generation.rois. "
                f"Available ROI keys: {sorted(roi_names)}"
            )
        return self


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def load_and_validate(config_path: Path) -> PipelineConfig:
    """Load a YAML config file and return a validated PipelineConfig."""
    import yaml

    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return PipelineConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# CLI — standalone validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Validate a pipeline config.yaml")
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "config.yaml"
    )
    args = parser.parse_args()

    try:
        cfg = load_and_validate(args.config)
        print(
            f"✓ Config valid — {len(cfg.subjects)} subject(s), "
            f"{len(cfg.target_generation.rois)} ROI(s), space={cfg.space}"
        )
    except Exception as e:  # noqa: BLE001 — CLI reports any validation failure
        print(f"✗ Invalid config:\n{e}", file=sys.stderr)
        sys.exit(1)
