"""
run-viz.py
----------
Cohort visualisation script for SimNIBS results.
Generates anatomical + e-field figures per subject with cohort-locked scale.
"""

from pathlib import Path

import numpy as np
import nibabel as nib


import sys

sys.path.insert(0, "/Users/hippolyte.dreyfus/Documents/simnibs-reader")
from simnibs_reader import SimulationResult, SegmentationResult

from simnibs_analyze.steps.viz import SimnibsViz


# ── Config ───────────────────────────────────────────────────────────

SUBJECTS = ["0001", "0003", "0004"]
SIM_BASE = Path(
    "/Volumes/levy/valerocabre/stimSD/Data/derivatives/mri/2-simnibs-simu-left"
)
SEG_BASE = Path("/Volumes/levy/valerocabre/stimSD/Data/derivatives/mri/1-simnibs-preps")
OUT_ROOT = Path("/Volumes/levy/valerocabre/stimSD/Analysis/simnibs-figures")

CUT_COORDS = [-42, -68, 32]  # centre de la lésion / cible
ROI_RADIUS = 10.0
STIM_PATTERN = "AFFT"

# ── Helper ───────────────────────────────────────────────────────────


def center_of_mass(mask_img):
    """Centre of mass of a binary mask in world (mm) coordinates."""
    data = mask_img.get_fdata()
    ijk = np.array(np.where(data > 0))
    if ijk.size == 0:
        return None
    center_vox = ijk.mean(axis=1)
    center_world = nib.affines.apply_affine(mask_img.affine, center_vox)
    return list(center_world.round(1))


# ── 1. Load all subjects ────────────────────────────────────────────

subjects = {}
for sub_id in SUBJECTS:
    seg_dir = SEG_BASE / sub_id / f"m2m_{sub_id}"
    seg = SegmentationResult(seg_dir)

    sim_parent = SIM_BASE / sub_id
    sim_dirs = sorted(sim_parent.rglob(f"simulation_*{STIM_PATTERN}*"))
    if not sim_dirs:
        print(
            f"⚠ {sub_id}: aucun dossier matching '*{STIM_PATTERN}*' dans {sim_parent}"
        )
        continue

    for sim_dir in sim_dirs:
        sim = SimulationResult(sim_dir)
        sim.set_segmentation(seg)
        key = f"{sub_id}_{sim_dir.name}"
        subjects[key] = (sim, seg)

viz = SimnibsViz(output_dir=OUT_ROOT)


# ── 2. Compute cohort e-field scale ─────────────────────────────────

efields = [sim.magnE_native for sim, _ in subjects.values()]
vmin, vmax = viz.set_scale_from_cohort(efields)
print(f"Cohort scale: [{vmin:.4f}, {vmax:.4f}] V/m")


# ── 3. Per-subject figures ──────────────────────────────────────────

for sub_id, (sim, seg) in subjects.items():
    print(f"\n{'='*60}\n  {sub_id}\n{'='*60}")

    out = OUT_ROOT / sub_id
    efield = sim.magnE_native
    t1 = seg.t1  # Path
    # label_prep = seg.tissue_labeling_upsampled # Path (si besoin)
    # lesion = ...                               # Path ou mask nifti

    # ROI
    roi = efield.get_roi(coords=CUT_COORDS, radius_mm=ROI_RADIUS)
    # center = center_of_mass(roi.mask_img) or CUT_COORDS
    center = CUT_COORDS

    # ==============================================================
    # A. ANAT (native)
    # ==============================================================

    # --- A1. Scalp 3D : 3/4 left + 3/4 right ---------------------
    vols_anat = [{"path": str(t1), "colormap": "gray", "opacity": 1.0}]

    viz.render_3d(vols_anat, out / "A1_scalp_3d_34left.png", azimuth=225, elevation=15)
    viz.render_3d(vols_anat, out / "A1_scalp_3d_34right.png", azimuth=135, elevation=15)

    # --- A2. Scalp 2D ortho, centré sur lésion --------------------
    viz.plot_anat(
        t1, cut_coords=center, output=out / "A2_anat_ortho.png", title=f"{sub_id} — T1"
    )

    # si lésion disponible :
    # viz.plot_efield_roi(t1, efield=None, roi_mask=lesion,
    #                     cut_coords=center,
    #                     output=out / "A2_anat_lesion_ortho.png",
    #                     title=f"{sub_id} — T1 + lésion")

    # --- A3. Brain 3D : lésion + T1 fondu, vue postérieure --------
    # NiiVue gère la transparence T1 + overlay ROI
    vols_brain_lesion = [
        {"path": str(t1), "colormap": "gray", "opacity": 0.4},
        # {"path": str(lesion_path), "colormap": "red", "opacity": 0.6},
    ]
    viz.render_3d(
        vols_brain_lesion, out / "A3_brain_3d_lesion_post.png", azimuth=0, elevation=15
    )

    # ==============================================================
    # B. SIMNIBS (e-field, native)
    # ==============================================================

    # --- B1. E-field 2D ortho + ROI contour -----------------------
    viz.plot_efield(
        t1,
        efield.img,
        cut_coords=center,
        output=out / "B1_efield_ortho.png",
        title=f"{sub_id} — magnE",
    )

    viz.plot_efield_roi(
        t1,
        efield.img,
        roi.mask_img,
        cut_coords=center,
        output=out / "B1_efield_roi_ortho.png",
        title=f"{sub_id} — magnE + ROI",
    )

    # --- B2. E-field mosaic axial ---------------------------------
    viz.plot_mosaic(
        t1,
        efield=efield.img,
        roi_mask=roi.mask_img,
        n_cuts=7,
        display_mode="z",
        output=out / "B2_efield_mosaic_axial.png",
        title=f"{sub_id} — magnE axial slices",
    )

    # --- B3. E-field 3D : orientations occipitales ----------------
    vols_efield = [
        {"path": str(t1), "colormap": "gray", "opacity": 1.0},
        {"path": str(efield.path), "colormap": "hot", "opacity": 0.7},
    ]

    orientations = {
        "post": (0, 15),
        "34left": (225, 15),
        "34right": (135, 15),
        "top": (0, 90),
    }
    for name, (az, el) in orientations.items():
        viz.render_3d(
            vols_efield, out / f"B3_efield_3d_{name}.png", azimuth=az, elevation=el
        )


print(f"\n✓ All figures saved in {OUT_ROOT}")
