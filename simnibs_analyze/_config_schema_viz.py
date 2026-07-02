"""
config_schema.py
----------------
Typed validation of the visualisation config (config-viz.yaml) with pydantic v2.

Usage
-----
    from config_schema import load_config
    cfg = load_config("config-viz_stimSD.yaml")   # raises on any malformed field

Design
------
- ``vols``   : a *named* registry of reusable layers (anat / roi / field).
              A figure references layers by name — this hides the raw NiiVue
              dict from the user, as the draft asked.
- ``figures``: a list of figure blocks; one block = one kind of figure.
- Cross-checks: every name referenced by a figure must exist in ``vols``;
  ``fields_scale.min < max``; a ROI has exactly one source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────
# Layers (the `vols` registry)
# ─────────────────────────────────────────────────────────────────────


class _LayerBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    colormap: str = "gray"
    opacity: float = Field(1.0, ge=0.0, le=1.0)


class AnatVol(_LayerBase):
    """Structural background layer."""

    kind: Literal["anat"]
    source: Literal["t1", "brain_mask", "label_prep", "lesion_native", "lesion_mni"] = (
        "t1"
    )
    render: Literal["fill", "contour"] = "fill"


class RoiVol(_LayerBase):
    """ROI layer — exactly one source: atlas+regions, coords (MNI), or file."""

    kind: Literal["roi"]
    # ROI : (x,y,z coord en mni) | atlas + list | file  (commençons simple)
    atlas: str | None = None
    regions: list[str] | None = None
    coords: list[float] | None = None  # MNI mm ; le script envoie en natif
    radius: float = 10.0
    file: str | None = None
    render: Literal["fill", "contour"] = "fill"  # 'contour' → TODO support SimnibsViz
    colormap: str = "blue"

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "RoiVol":
        n = sum(x is not None for x in (self.atlas, self.coords, self.file))
        if n != 1:
            raise ValueError(
                "roi vol needs exactly one source among {atlas, coords, file}, "
                f"got {n}."
            )
        if self.atlas is not None and not self.regions:
            raise ValueError("roi vol with `atlas` also requires `regions`.")
        if self.coords is not None and len(self.coords) != 3:
            raise ValueError("roi `coords` must be [x, y, z] in MNI mm.")
        return self


class FieldVol(_LayerBase):
    """E-field / current-density layer (continuous, colour-scaled)."""

    kind: Literal["field"]
    name: str = "magnE"  # e | E | j | J | magnE | magnJ
    colormap: str = "hot"
    opacity: float = Field(0.6, ge=0.0, le=1.0)
    # for later ! space: mni or native


VolSpec = Annotated[Union[AnatVol, RoiVol, FieldVol], Field(discriminator="kind")]


# ─────────────────────────────────────────────────────────────────────
# Figure blocks
# ─────────────────────────────────────────────────────────────────────


class FigureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal["2D", "3D"]
    vols: list[str]  # references into the top-level `vols` registry
    cohort: bool = False
    if_exists: Literal["overwrite", "skip", "error"] = (
        "overwrite"  # TODO enforce in viz
    )

    # -- 2D only --
    subtype: Literal["ortho", "parallel"] = "ortho"

    # Centre de coupe (ortho ET parallel) — au plus un des deux :
    #   cut_coords   : [x, y, z] MNI explicites (ortho: centre; parallel: projeté sur l'axe)
    #   cut_center_vol: nom d'un vol → CoM calculé au runtime (prioritaire sur cut_coords)
    cut_coords: list[float] | None = None
    cut_center_vol: str | None = None

    # Étendue (parallel uniquement) :
    #   half_width + spacing → coupes de (centre - half_width) à (centre + half_width)
    #   sans half_width      → n_cuts coupes auto-réparties par nilearn
    axis: Literal["x", "y", "z"] = "z"
    n_cuts: int = 7
    half_width: float | None = None  # demi-largeur du bloc (mm) ; parallel seulement
    spacing: float | None = None  # mm entre coupes ; nécessite half_width

    # -- contour overrides (per-figure, overrides vol-level render) --
    contour_vols: list[str] = Field(
        default_factory=list
    )  # vol names to render as contour in this figure

    # -- cohort montage --
    montage_ncols: int | None = None  # colonnes dans la grille cohorte (None = auto)
    montage_panel_h: float = (
        4.0  # hauteur d'un panel en pouces (la largeur est déduite de l'aspect ratio)
    )

    # -- 3D only --
    camera: list[float] = Field(
        default_factory=lambda: [225.0, 15.0]
    )  # azimuth, elevation

    @field_validator("camera")
    @classmethod
    def _camera_len(cls, v: list[float]) -> list[float]:
        if len(v) != 2:
            raise ValueError("camera must be [azimuth, elevation].")
        return v

    @model_validator(mode="after")
    def _validate_cut(self) -> "FigureConfig":
        # cut_coords et cut_center_vol sont mutuellement exclusifs
        if self.cut_coords is not None and self.cut_center_vol is not None:
            raise ValueError(
                "`cut_coords` and `cut_center_vol` are mutually exclusive."
            )
        if self.cut_coords is not None and len(self.cut_coords) != 3:
            raise ValueError("`cut_coords` must be [x, y, z] in MNI mm.")
        # half_width sans spacing n'a pas de sens (on ne sait pas combien de coupes)
        if (
            self.half_width is not None
            and self.spacing is None
            and self.subtype == "parallel"
        ):
            raise ValueError("`half_width` requires `spacing`.")
        return self


# ─────────────────────────────────────────────────────────────────────
# Top-level
# ─────────────────────────────────────────────────────────────────────


class SubjectsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ids: list[str]
    stim_pattern: str
    # TODO gérer les groupes de sujets left and right


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sim_base: Path
    seg_base: Path
    out_root: Path


class FieldsScale(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min: float
    max: float
    # for later ! symmetric scale
    # for later ! space: mni or native

    @model_validator(mode="after")
    def _ordered(self) -> "FieldsScale":
        if self.min >= self.max:
            raise ValueError(
                f"fields_scale: min ({self.min}) must be < max ({self.max})."
            )
        return self


class VizConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subjects: SubjectsConfig
    paths: PathsConfig
    fields_scale: FieldsScale
    vols: dict[str, VolSpec]
    figures: list[FigureConfig]

    @model_validator(mode="after")
    def _references_exist(self) -> "VizConfig":
        known = set(self.vols)
        for fig in self.figures:
            missing = [v for v in fig.vols if v not in known]
            if missing:
                raise ValueError(
                    f"figure '{fig.name}' references unknown vols {missing}; "
                    f"available: {sorted(known)}."
                )
            # a figure without any layer is almost surely a mistake
            if not fig.vols:
                raise ValueError(f"figure '{fig.name}' has an empty `vols` list.")
        return self

    def figure_has_field(self, fig: FigureConfig) -> bool:
        """True if any layer referenced by *fig* is a continuous field."""
        return any(isinstance(self.vols[v], FieldVol) for v in fig.vols)


def load_config(path: str | Path) -> VizConfig:
    """Load + validate a YAML config into a :class:`VizConfig`."""
    import yaml

    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return VizConfig.model_validate(raw)
