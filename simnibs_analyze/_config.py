from __future__ import annotations

"""
Pydantic models for pipeline config.yaml validation.

These models mirror the config.yaml structure exactly.
Usage (standalone check):
    python _config.py --config config.yaml
"""

from pathlib import Path
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# ROI definitions  (target_generation.rois)
# ---------------------------------------------------------------------------


class VisualizationConfig(BaseModel):
    figure_type: Union[str, list[str]] = "3d"  # ou 'parallel', 'acs', '3d'
    lesion_mask_color: str = "magenta"
    lesion_mask_opacity: float = 0.4
    t1_opacity: float = 0.15
    mask_color: str = "cyan"
    mask_opacity: float = 0.3
    camera_position: Union[str, list[str]] = "xy"
    method_definition_center_acs: str = "roi"  # ou 'lesion'
    cmap: str = "hot"


class SphereROI(BaseModel):
    """ROI defined by a sphere centred on MNI coordinates."""

    method: Literal["sphere"]
    coords: Annotated[list[float], Field(min_length=3, max_length=3)]
    folder_pattern: Optional[str] = None
    """Glob fragment used to find SimNIBS output folders for this ROI.
    If omitted, the ROI key name is used (e.g. 'fef' → 'simulation_simulation_fef_*').
    Set this when the folder name differs from the ROI key (e.g. ROI 'ips-left' but
    folders are named '…ips_left…' → folder_pattern: 'ips_left')."""


class AtlasROI(BaseModel):
    """ROI defined by one or more parcels from a brain atlas."""

    method: Literal["atlas"]
    atlas: Literal["harvard-oxford", "aal", "destrieux"]
    regions: Union[str, list[str]]
    folder_pattern: Optional[str] = None
    """See SphereROI.folder_pattern."""


# Discriminated union: Pydantic inspects the `method` field to pick the right model.
ROIDef = Annotated[Union[SphereROI, AtlasROI], Field(discriminator="method")]


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
    # ── Nouvelles clés (préféré) ───────────────────────────────────────────
    simnibs_preps: Optional[Path] = None
    """Dossier des résultats charm/segmentation  →  {sub}/m2m_{sub}/"""
    simnibs_simu: Optional[Path] = None
    """Dossier des résultats de simulation/optimisation  →  {sub}/simulations/"""

    # ── Compatibilité ascendante (ancien pipeline, un seul dossier racine) ─
    simnibs_output: Optional[Path] = None
    """Deprecated : utiliser simnibs_preps + simnibs_simu."""

    # ── Sorties pipeline ──────────────────────────────────────────────────
    results_dir: Path
    """Dossier de sortie : CSVs, figures, statistiques."""

    # ── Templates ─────────────────────────────────────────────────────────
    mni_template: Optional[Path] = None
    mni_brain_mask: Optional[Path] = None

    lesion_masks_dir: Optional[Path] = None
    """Dossier des masques de lésion synthstroke (un dossier par sujet)."""

    @model_validator(mode="after")
    def _validate_input_paths(self) -> "PathsConfig":
        split = self.simnibs_preps is not None and self.simnibs_simu is not None
        legacy = self.simnibs_output is not None
        if not split and not legacy:
            raise ValueError(
                "Specify either 'simnibs_output' (legacy) OR both "
                "'simnibs_preps' + 'simnibs_simu'."
            )
        return self


class PreprocessingConfig(BaseModel):
    smooth_fwhm: float = Field(default=2.0, ge=0)
    outlier_method: Literal["iqr", "zscore"] = "iqr"
    portion: Optional[float] = Field(default=None, gt=0, le=1)


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
    visualisation: VisualizationConfig = VisualizationConfig()

    @model_validator(mode="after")
    def _stim_conditions_match_rois(self) -> "PipelineConfig":
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
    except Exception as e:
        print(f"✗ Invalid config:\n{e}", file=sys.stderr)
        sys.exit(1)
