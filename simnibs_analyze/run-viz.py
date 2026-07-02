"""
run-viz.py
----------
Config-driven cohort visualisation for SimNIBS results.

Reads a validated YAML config (see config_schema.py) and, for each figure
block, generates one figure per subject — then composes the cohort montages
for the blocks flagged ``cohort: true`` (single shared e-field scale).

    python run-viz.py config-viz_stimSD.yaml
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

sys.path.insert(0, "/Users/hippolyte.dreyfus/Documents/simnibs-reader")
from simnibs_reader import SimulationResult, SegmentationResult
from simnibs_reader.nifti.efield import EField  # for later TODO REMOVE !!!

from simnibs_analyze.steps.viz import SimnibsViz
from simnibs_analyze._config_schema_viz import (
    VizConfig,
    FigureConfig,
    AnatVol,
    RoiVol,
    FieldVol,
    load_config,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers — MNI ↔ native warping (Conform2MNI_nonl, cf. debug_warp.py)
# ─────────────────────────────────────────────────────────────────────


@dataclass
class SubjectCtx:
    """Per-subject state reused across every figure block."""

    key: str
    sub_id: str
    sim: object
    seg: object
    out: Path
    warp_img: object
    warp_coords: np.ndarray  # (N, 3) MNI mm per subject voxel
    _layer_cache: dict = field(default_factory=dict)


def build_warp(seg) -> tuple[object, np.ndarray]:
    """Load Conform2MNI_nonl (subject grid, MNI-mm values) once per subject."""
    warp_img = nib.load(str(seg.path / "toMNI" / "Conform2MNI_nonl.nii.gz"))
    warp_coords = np.squeeze(warp_img.get_fdata()).reshape(-1, 3)
    return warp_img, warp_coords


def warp_mni_mask_to_native(mask_mni, warp_img, warp_coords) -> nib.Nifti1Image:
    """Pull an MNI binary mask onto the subject grid via the deformation field."""
    inv_aff = np.linalg.inv(mask_mni.affine)
    vox = nib.affines.apply_affine(inv_aff, warp_coords).T  # (3, N) voxel idx MNI
    warped = map_coordinates(mask_mni.get_fdata(), vox, order=0, cval=0)
    data = warped.reshape(warp_img.shape[:3]).astype(np.uint8)
    return nib.Nifti1Image(data, warp_img.affine)


def mni_to_native_coords(mni_xyz, warp_img, warp_coords) -> list[float]:
    """Nearest subject-space (mm) point to an MNI target, via the warp field."""
    dist = np.sum((warp_coords - np.asarray(mni_xyz)) ** 2, axis=1)
    closest_vox = np.unravel_index(np.argmin(dist), warp_img.shape[:3])
    return list(nib.affines.apply_affine(warp_img.affine, closest_vox).round(1))


# ─────────────────────────────────────────────────────────────────────
# Layer resolution — config vol spec → concrete NiiVue/nilearn dict
# ─────────────────────────────────────────────────────────────────────


def _field_efield(sim, name: str) -> EField:
    """Map a config field name to a reader e-field (native space)."""
    key = name.lower()
    if key in ("e", "magne"):
        return sim.magnE_native
    if key in ("j", "magnj"):
        return sim.magnJ_native
    # for later : E/J vector components, MNI space, normalE ...
    raise ValueError(f"champ '{name}' non géré (attendu: e/E/j/J).")


def resolve_layer(name: str, cfg: VizConfig, ctx: SubjectCtx) -> dict:
    """Turn a named vol spec into a dict {path, colormap, opacity[, cal_min/max]}.

    Cached per subject: the same ROI/atlas isn't warped twice.
    """
    if name in ctx._layer_cache:
        return dict(ctx._layer_cache[name])

    spec = cfg.vols[name]

    if isinstance(spec, AnatVol):
        if spec.source == "lesion_native":
            path = ctx.seg.lesion_native
            if path is None:
                raise FileNotFoundError(
                    f"{ctx.sub_id}: lesion_native introuvable (seg.lesion_native is None)"
                )
        elif spec.source == "lesion_mni":
            path = ctx.seg.lesion_mni
            if path is None:
                raise FileNotFoundError(
                    f"{ctx.sub_id}: lesion_mni introuvable (seg.lesion_mni is None)"
                )
        else:
            path = {
                "t1": ctx.seg.t1,
                "brain_mask": ctx.seg.path
                / "surfaces"
                / "cereb_mask.nii.gz",  # TODO confirmer le bon mask
                "label_prep": ctx.seg.tissue_labeling_upsampled,
            }[spec.source]
        layer = {"path": str(path), "colormap": spec.colormap, "opacity": spec.opacity}

    elif isinstance(spec, RoiVol):
        native_path = ctx.out / f"roi_{name}_native.nii.gz"
        if spec.atlas is not None:
            mask_mni = EField._from_atlas(spec.atlas, spec.regions)
            mask_native = warp_mni_mask_to_native(
                mask_mni, ctx.warp_img, ctx.warp_coords
            )
            nib.save(mask_native, str(native_path))
        elif spec.coords is not None:
            native_center = mni_to_native_coords(
                spec.coords, ctx.warp_img, ctx.warp_coords
            )
            roi = ctx.sim.magnE_native.get_roi(coords=native_center, radius=spec.radius)
            nib.save(roi.mask_img, str(native_path))
        else:  # file
            native_path = Path(
                spec.file
            )  # TODO: on suppose déjà natif ; fichier MNI = later
        # TODO (A1) : filtrer l'hémisphère controlatéral (x<0 / x>0 en MNI avant warp)
        # TODO (A2) : spec.render == "contour" → nécessite un support dans SimnibsViz
        layer = {
            "path": str(native_path),
            "colormap": spec.colormap,
            "opacity": spec.opacity,
        }

    elif isinstance(spec, FieldVol):
        efield = _field_efield(ctx.sim, spec.name)
        layer = {
            "path": str(efield.path),
            "colormap": spec.colormap,
            "opacity": spec.opacity,
            # échelle cohorte injectée → 2D (plot_anat vols) ET 3D (render_3d) cohérents
            "cal_min": cfg.fields_scale.min,
            "cal_max": cfg.fields_scale.max,
        }
        # TODO (B1) : meilleure saturation e-field — à raffiner dans SimnibsViz
    else:  # pragma: no cover
        raise TypeError(f"vol spec inconnu: {type(spec).__name__}")

    ctx._layer_cache[name] = layer
    return dict(layer)


# ─────────────────────────────────────────────────────────────────────
# Figure dispatch
# ─────────────────────────────────────────────────────────────────────


def _parallel_cut_coords(fig: FigureConfig):
    """Explicit cut positions if coord_min/max/spacing given, else n_cuts (int)."""
    if fig.coord_min is not None and fig.coord_max is not None and fig.spacing:
        return list(np.arange(fig.coord_min, fig.coord_max + 1e-6, fig.spacing))
    return fig.n_cuts  # nilearn auto-spreads across the whole brain


def render_figure(
    fig: FigureConfig, cfg: VizConfig, ctx: SubjectCtx, viz: SimnibsViz
) -> Path:
    """Produce one figure of block *fig* for subject *ctx*; return the PNG path."""
    vols = [resolve_layer(v, cfg, ctx) for v in fig.vols]
    out_png = ctx.out / f"{fig.name}.png"
    title = f"{ctx.sub_id} — {fig.name}"

    if fig.type == "3D":
        az, el = fig.camera
        # cohort panels are rendered WITHOUT their own colorbar (shared one in montage)
        return viz.render_3d(
            vols, out_png, azimuth=az, elevation=el, colorbar=not fig.cohort
        )

    # -- 2D --
    if fig.subtype == "ortho":
        center = (
            mni_to_native_coords(fig.cut_coords, ctx.warp_img, ctx.warp_coords)
            if fig.cut_coords is not None
            else None
        )
        viz.plot_anat(
            vols, cut_coords=center, display_mode="ortho", output=out_png, title=title
        )
    else:  # parallel (mosaic)
        viz.plot_anat(
            vols,
            cut_coords=_parallel_cut_coords(fig),
            display_mode=fig.axis,
            output=out_png,
            title=title,
        )
    return out_png


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def load_subjects(cfg: VizConfig) -> dict[str, tuple]:
    subjects = {}
    for sub_id in cfg.subjects.ids:
        seg = SegmentationResult(cfg.paths.seg_base / sub_id / f"m2m_{sub_id}")
        sim_parent = cfg.paths.sim_base / sub_id
        sim_dirs = sorted(sim_parent.rglob(f"simulation_*{cfg.subjects.stim_pattern}*"))
        if not sim_dirs:
            print(
                f"⚠ {sub_id}: aucun dossier '*{cfg.subjects.stim_pattern}*' dans {sim_parent}"
            )
            continue
        for sim_dir in sim_dirs:
            sim = SimulationResult(sim_dir)
            sim.set_segmentation(seg)
            subjects[f"{sub_id}_{sim_dir.name}"] = (sim, seg)
    if not subjects:
        raise SystemExit("Aucune simulation trouvée — vérifie paths / stim_pattern.")
    return subjects


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    subjects = load_subjects(cfg)

    viz = SimnibsViz(output_dir=cfg.paths.out_root)
    viz.set_scale(cfg.fields_scale.min, cfg.fields_scale.max)  # échelle manuelle fixe
    print(
        f"Cohort scale (manual): [{cfg.fields_scale.min}, {cfg.fields_scale.max}] V/m"
    )

    # collecteurs cohorte : {figure_name: [ {label, image}, ... ]}
    cohort_panels: dict[str, list[dict]] = {f.name: [] for f in cfg.figures if f.cohort}

    # ── Par sujet ──────────────────────────────────────────────────
    for key, (sim, seg) in subjects.items():
        sub_id = key.split("_")[0]
        print(f"\n{'='*60}\n  {key}\n{'='*60}")

        out = cfg.paths.out_root / key
        out.mkdir(parents=True, exist_ok=True)
        warp_img, warp_coords = build_warp(seg)
        ctx = SubjectCtx(key, sub_id, sim, seg, out, warp_img, warp_coords)

        for fig in cfg.figures:
            png = render_figure(fig, cfg, ctx, viz)
            if fig.cohort:
                cohort_panels[fig.name].append({"label": sub_id, "image": png})

    # ── Cohorte (une seule fois, tous sujets confondus) ────────────
    cohort_dir = cfg.paths.out_root / "_cohort"
    for fig in cfg.figures:
        if not fig.cohort:
            continue
        viz.plot_cohort_montage(
            cohort_panels[fig.name],
            cohort_dir / f"cohort_{fig.name}.png",
            title=f"{fig.name} — cohorte",
            add_colorbar=cfg.figure_has_field(fig),  # échelle unique ssi champ continu
            cbar_label="E-field (V/m)",
        )

    print(f"\n✓ All figures saved in {cfg.paths.out_root}")


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config-viz_stimSD.yaml"
    main(cfg_path)
