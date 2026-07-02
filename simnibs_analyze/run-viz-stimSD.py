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
from simnibs_reader.nifti.efield import EField  # for later TODO REMOVE !!!

from scipy.ndimage import map_coordinates


from simnibs_analyze.steps.viz import SimnibsViz


# ── Config ───────────────────────────────────────────────────────────
## TODO gerer les groupes de sujets left and right.

SUBJECTS = [
    "0001",
    "0003",
    # "0004",
    # "0005",
    # "0007",
    # "0008",
    # "0009",
    # "0012",
    # "0013",
    # "0015",
    # "0017",
    # "0021",
    # "0023",
    # "0031",
    # "0032",
    # "0034",
    # "0035",
    # "0036",
    # "0037",
    # "0043",
    # "0046",
    # "0048",
    # "0050",
    # "0052",
    # "0053",
    # "0054",
    # "0055",
    # "0056",
    # "0057",
    # "0058",
    # "0060",
    # "0061",
    # "0062",
    # "0063",
    # "0064",
    # "0065",
]
SIM_BASE = Path(
    "/Volumes/levy/valerocabre/stimSD/Data/derivatives/mri/2-simnibs-simu-left"
)
SEG_BASE = Path("/Volumes/levy/valerocabre/stimSD/Data/derivatives/mri/1-simnibs-preps")
OUT_ROOT = Path("/Volumes/levy/valerocabre/stimSD/Analysis/simnibs-figures")

MNI_CUT_COORDS = [56, 8, -10]  # centre de la lésion / cible
ROI_RADIUS = 10.0
STIM_PATTERN = "AFFT"

# ── Helper ───────────────────────────────────────────────────────────


def center_of_mass(mask_img):
    """Centre of mass of a binary mask in world (mm) coordinates."""
    data = np.squeeze(mask_img.get_fdata())  # robuste au 4e axe singleton SimNIBS
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

if not subjects:
    raise SystemExit("Aucune simulation trouvée — vérifie SIM_BASE / STIM_PATTERN.")

viz = SimnibsViz(output_dir=OUT_ROOT)


# ── 2. Compute cohort e-field scale ─────────────────────────────────

efields = [sim.magnE_native for sim, _ in subjects.values()]
# vmin, vmax = viz.set_scale_from_cohort(efields, alpha=0.1,lower_pct=50, upper_pct=85)
viz.set_scale(
    0.05, 0.4
)  # ← manuel : bornes fixes en V/m #TODO symetric scale for later


# ── 3. Per-subject figures ──────────────────────────────────────────
# collecteurs pour les montages cohorte (remplis dans la boucle)
panels_efield_3d = []  # portera l'échelle e-field → colorbar partagée
panels_brain_3d = []  # anat seul → pas de colorbar

for key, (sim, seg) in subjects.items():
    sub_id = key.split("_")[0]  # "0001_simulation_..." → "0001"

    print(f"\n{'='*60}\n  {key}\n{'='*60}")

    out = OUT_ROOT / key
    out.mkdir(parents=True, exist_ok=True)

    efield = sim.magnE_native
    t1 = seg.t1  # Path
    t1_brain = seg.path / "surfaces" / "cereb_mask.nii.gz"
    # label_prep = seg.tissue_labeling_upsampled  # Path (si besoin)
    # lesion = ...                                # Path ou mask nifti

    # ROI (sphère en espace sujet, centrée sur la cible)
    # roi = efield.get_roi(coords=CUT_COORDS, radius=ROI_RADIUS)
    # roi = efield.get_roi(atlas='harvard-oxford', region='Frontal Pole')
    # center = center_of_mass(roi.mask_img) or CUT_COORDS
    # center = MNI_CUT_COORDS

    ##### PROJECTIONS TO NATIVE

    # ── 1. Mask atlas en MNI ──
    ATL = [
        "Temporal Pole",
        "Superior Temporal Gyrus, anterior division",
        "Middle Temporal Gyrus, anterior division",
        "Inferior Temporal Gyrus, anterior division",
        "Temporal Fusiform Cortex, anterior division",
    ]
    mask_mni = EField._from_atlas("harvard-oxford", ATL)

    # ── 2. Warp MNI → natif (volume) ──
    warp_img = nib.load(
        str(seg.path / "toMNI" / "Conform2MNI_nonl.nii.gz")
    )  # ← pas MNI2Conform
    warp_data = np.squeeze(warp_img.get_fdata())  # (176, 256, 256, 3)

    inv_aff = np.linalg.inv(mask_mni.affine)
    coords = warp_data.reshape(-1, 3)
    vox = (inv_aff @ np.column_stack([coords, np.ones(len(coords))]).T)[:3]

    warped = map_coordinates(mask_mni.get_fdata(), vox, order=0, cval=0)
    mask_native = nib.Nifti1Image(
        warped.reshape(warp_data.shape[:3]).astype(np.uint8), warp_img.affine
    )

    # Écriture disque : NiiVue charge via un chemin, pas un objet en mémoire
    out.mkdir(parents=True, exist_ok=True)
    mask_native_path = out / "roi_native_mask.nii.gz"
    nib.save(mask_native, str(mask_native_path))

    # ── 3. Coords MNI → natif (même warp) ──
    mni_target = np.array(MNI_CUT_COORDS)
    dist = np.sum((coords - mni_target) ** 2, axis=1)
    closest_vox = np.unravel_index(np.argmin(dist), warp_data.shape[:3])
    center = list(nib.affines.apply_affine(warp_img.affine, closest_vox).round(1))
    print(f"MNI {MNI_CUT_COORDS} → Subject {center}")

    # ==============================================================
    # A. ANAT (native)
    # ==============================================================

    # --- A1. Scalp 3D : 3/4 left + 3/4 right ---------------------
    vols = [
        {"path": str(t1), "opacity": 0.5},
        {"path": str(mask_native_path), "opacity": 0.5},
    ]
    png_brain = viz.render_3d(
        vols, out / "A1_scalp_3d_34left.png", azimuth=225, elevation=15
    )
    panels_brain_3d.append({"label": sub_id, "image": png_brain})

    viz.render_3d(vols, out / "A1_scalp_3d_34right.png", azimuth=135, elevation=15)
    viz.plot_anat(
        vols,
        cut_coords=center,
        output=out / "A2_anat_ortho_brain.png",
        title=f"{key} — T1",
    )

    vols = [
        {"path": str(t1_brain), "colormap": "gray", "opacity": 0.5},
        {"path": str(mask_native_path), "colormap": "blue", "opacity": 0.5},
    ]
    viz.render_3d(vols, out / "A1_brain_3d_34left.png", azimuth=225, elevation=15)
    viz.render_3d(vols, out / "A1_brain_3d_34right.png", azimuth=135, elevation=15)

    # --- A2. Scalp 2D ortho, centré sur lésion --------------------
    viz.plot_anat(
        vols,
        cut_coords=center,
        output=out / "A2_anat_ortho_skull.png",
        title=f"{key} — T1",
    )
    # viz.plot_anat(t1, cut_coords=center, output=out / "A2_anat_ortho-old.png", title=f"{key} — T1")

    # ==============================================================
    # B. SIMNIBS (e-field, native)
    # ==============================================================

    # --- B1. E-field 2D ortho + ROI contour -----------------------
    vols = [
        {"path": str(t1_brain), "colormap": "gray", "opacity": 0.4},
        {"path": str(mask_native_path), "colormap": "blue", "opacity": 0.2},
        {"path": str(efield.path), "colormap": "linspecer", "opacity": 0.6},
    ]
    png_efield = viz.render_3d(
        vols, out / "B1_efield_3d.png", azimuth=225, elevation=15
    )
    panels_efield_3d.append({"label": sub_id, "image": png_efield})

    viz.plot_efield(
        t1,
        efield.img,
        cut_coords=center,
        output=out / "B1_efield_ortho.png",
        title=f"{key} — magnE",
    )

    viz.plot_efield_roi(
        t1,
        efield.img,
        mask_native,
        cut_coords=center,
        output=out / "B1_efield_roi_ortho.png",
        title=f"{key} — magnE + ROI",
    )

    # --- B2. E-field mosaic axial ---------------------------------
    viz.plot_mosaic(
        t1,
        efield=efield.img,
        roi_mask=mask_native,
        n_cuts=10,
        display_mode="z",
        output=out / "B2_efield_mosaic_axial.png",
        title=f"{key} — magnE axial slices",
    )

# ==============================================================
# C. Cohort figures  (une seule fois, tous sujets confondus)
# ==============================================================

cohort_dir = OUT_ROOT / "_cohort"

# e-field : UNE colorbar partagée, dérivée du scale cohorte verrouillé
viz.plot_cohort_montage(
    panels_efield_3d,
    cohort_dir / "cohort_efield_3d_34left.png",
    title="magnE — cohorte (3/4 left)",
    cbar_label="E-field (V/m)",
)

# anat : pas d'échelle → pas de colorbar
viz.plot_cohort_montage(
    panels_brain_3d,
    cohort_dir / "cohort_brain_3d_34left.png",
    title="Brain + ROI — cohorte (3/4 left)",
    add_colorbar=False,
)


print(f"\n✓ All figures saved in {OUT_ROOT}")
