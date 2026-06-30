"""
debug_warp.py
-------------
Diagnose why the MNI→native atlas warp produces an empty mask.

It inspects BOTH SimNIBS deformation fields (toMNI/MNI2Conform_nonl and
toMNI/Conform2MNI_nonl), figures out — empirically — which one is defined on
the subject grid and what space its values live in (MNI mm / subject mm /
voxel indices), then runs the warp with each candidate and reports which
yields a non-empty mask aligned with the T1.

Run:  python debug_warp.py
Edit the CONFIG block for your subject.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

# ── CONFIG ───────────────────────────────────────────────────────────
sys.path.insert(0, "/Users/hippolyte.dreyfus/Documents/simnibs-reader")
from simnibs_reader.nifti.efield import EField  # noqa: E402

SUB_ID = "0001"
SEG_DIR = (
    Path("/Volumes/levy/valerocabre/stimSD/Data/derivatives/mri/1-simnibs-preps")
    / SUB_ID
    / f"m2m_{SUB_ID}"
)
OUT_DIR = Path("/tmp/debug_warp") / SUB_ID
ATLAS, REGION = "harvard-oxford", "Frontal Pole"
# ─────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
np.set_printoptions(precision=1, suppress=True)


def rule(title: str) -> None:
    print(f"\n{'='*64}\n  {title}\n{'='*64}")


def world_bbox(img: nib.Nifti1Image) -> np.ndarray:
    """Min/max world (mm) coordinates spanned by the volume, shape (2, 3)."""
    shape = np.array(img.shape[:3])
    corners = np.array(
        [
            [i, j, k]
            for i in (0, shape[0] - 1)
            for j in (0, shape[1] - 1)
            for k in (0, shape[2] - 1)
        ],
        float,
    )
    world = nib.affines.apply_affine(img.affine, corners)
    return np.vstack([world.min(0), world.max(0)])


def describe(name: str, img: nib.Nifti1Image) -> None:
    print(f"\n[{name}]")
    print(f"  shape  : {img.shape}")
    print(f"  zooms  : {np.round(img.header.get_zooms()[:3], 2)}")
    bb = world_bbox(img)
    print(
        f"  world bbox (mm):  x[{bb[0,0]:.0f},{bb[1,0]:.0f}]  "
        f"y[{bb[0,1]:.0f},{bb[1,1]:.0f}]  z[{bb[0,2]:.0f},{bb[1,2]:.0f}]"
    )


def value_space_guess(vals: np.ndarray) -> str:
    """Heuristic: are these (N,3) values MNI mm, subject mm, or voxel indices?"""
    lo, hi = vals.min(0), vals.max(0)
    looks_voxel = (lo >= -2).all() and (hi <= 400).all() and (hi > 90).all()
    spans_negative = (lo < -20).any()
    if looks_voxel and not spans_negative:
        return "VOXEL indices (0..N) — do NOT apply inv(affine)"
    if spans_negative:
        return "WORLD mm (centred near 0) — apply inv(mask.affine) before sampling"
    return "ambiguous — inspect ranges manually"


def warp_mni_to_native(mask_mni, warp_img, *, values_are_mm: bool) -> nib.Nifti1Image:
    """Pull an MNI mask onto the warp field's grid.

    warp_data[v] gives, for each output voxel v, the location in the MNI mask
    to sample. If those locations are world mm, convert to mask voxel indices
    first; if they are already voxel indices, sample directly.
    """
    warp_data = np.squeeze(warp_img.get_fdata())
    coords = warp_data.reshape(-1, 3)
    if values_are_mm:
        vox = nib.affines.apply_affine(np.linalg.inv(mask_mni.affine), coords).T
    else:
        vox = coords.T
    warped = map_coordinates(mask_mni.get_fdata(), vox, order=0, cval=0)
    data = warped.reshape(warp_data.shape[:3]).astype(np.uint8)
    return nib.Nifti1Image(data, warp_img.affine)


def report_mask(name: str, mask: nib.Nifti1Image, t1: nib.Nifti1Image) -> None:
    data = np.squeeze(mask.get_fdata())
    n = int((data > 0).sum())
    print(f"\n  → {name}: {n} voxels non nuls", end="")
    if n == 0:
        print("  ❌ VIDE")
        return
    ijk = np.array(np.where(data > 0))
    com_world = nib.affines.apply_affine(mask.affine, ijk.mean(1))
    t1_bb = world_bbox(t1)
    inside = (com_world >= t1_bb[0]).all() and (com_world <= t1_bb[1]).all()
    flag = "✓ dans la bbox T1" if inside else "❌ HORS bbox T1 (mauvais espace)"
    print(f"  COM(mm)={np.round(com_world,1)}  {flag}")
    same_grid = (mask.shape[:3] == t1.shape[:3]) and np.allclose(
        mask.affine, t1.affine, atol=1e-3
    )
    print(f"     grille == T1 ? {same_grid}")


# ═════════════════════════════════════════════════════════════════════
rule("0 · Inputs")

t1_path = SEG_DIR / "T1.nii.gz"
assert t1_path.exists(), f"T1 introuvable: {t1_path}"
t1 = nib.load(str(t1_path))
describe("T1 (espace sujet de référence)", t1)
print("  T1 bbox ↑ = l'enveloppe où un masque sujet VALIDE doit tomber.")

mask_mni = EField._from_atlas(ATLAS, REGION)
describe(f"mask_mni  ({ATLAS} / {REGION})", mask_mni)
print(f"  voxels non nuls (atlas MNI): {int((mask_mni.get_fdata()>0).sum())}")
assert (mask_mni.get_fdata() > 0).any(), "Le masque atlas MNI est déjà vide !"


# ═════════════════════════════════════════════════════════════════════
rule("1 · Inspection des deux champs de déformation")

warp_files = {
    "MNI2Conform_nonl": SEG_DIR / "toMNI" / "MNI2Conform_nonl.nii.gz",
    "Conform2MNI_nonl": SEG_DIR / "toMNI" / "Conform2MNI_nonl.nii.gz",
}

warp_imgs: dict[str, nib.Nifti1Image] = {}
for name, p in warp_files.items():
    if not p.exists():
        print(f"\n[{name}]  ABSENT: {p}")
        continue
    img = nib.load(str(p))
    warp_imgs[name] = img
    describe(name, img)
    vals = np.squeeze(img.get_fdata()).reshape(-1, 3)
    print(f"  valeurs min/axe : {vals.min(0)}")
    print(f"  valeurs max/axe : {vals.max(0)}")
    print(f"  → interprétation : {value_space_guess(vals)}")
    on_subject_grid = img.shape[:3] == t1.shape[:3]
    print(
        f"  → grille == T1 ? {on_subject_grid}  "
        f"({'CANDIDAT pour MNI→sujet' if on_subject_grid else 'défini sur grille MNI'})"
    )


# ═════════════════════════════════════════════════════════════════════
rule("2 · Test du warp avec chaque champ")
print("Un masque sujet correct doit : être NON vide, COM dans la bbox T1.\n")

for name, warp_img in warp_imgs.items():
    vals = np.squeeze(warp_img.get_fdata()).reshape(-1, 3)
    mm = (vals.min(0) < -20).any()  # mm si valeurs négatives marquées
    print(f"\n--- {name} (values_are_mm={mm}) ---")
    try:
        mask_native = warp_mni_to_native(mask_mni, warp_img, values_are_mm=mm)
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ exception: {type(e).__name__}: {e}")
        continue
    report_mask(name, mask_native, t1)
    out_p = OUT_DIR / f"mask_native_via_{name}.nii.gz"
    nib.save(mask_native, str(out_p))
    print(f"     sauvegardé: {out_p}")


# ═════════════════════════════════════════════════════════════════════
rule("3 · Référence : SimNIBS nifti_transform (si dispo)")
try:
    from simnibs.utils.transformations import nifti_transform

    mni_mask_path = OUT_DIR / "mask_mni.nii.gz"
    nib.save(mask_mni, str(mni_mask_path))
    for warp_name in ("Conform2MNI_nonl", "MNI2Conform_nonl"):
        wp = warp_files[warp_name]
        if not wp.exists():
            continue
        out_p = OUT_DIR / f"mask_native_simnibs_{warp_name}.nii.gz"
        try:
            nifti_transform(
                str(mni_mask_path), str(wp), str(t1_path), str(out_p), order=0
            )
            m = nib.load(str(out_p))
            report_mask(f"nifti_transform[{warp_name}]", m, t1)
            print(f"     sauvegardé: {out_p}")
        except Exception as e:  # noqa: BLE001
            print(f"  [{warp_name}] ❌ {type(e).__name__}: {e}")
except ImportError:
    print(
        "simnibs non importable ici — section ignorée "
        "(c'est pourtant la voie robuste : voir get_roi_mni)."
    )

rule("Conclusion")
print(
    "Le bon champ est celui dont (a) la grille == T1 et (b) le warp donne un\n"
    "masque non vide avec COM dans la bbox T1. Compare la sortie de la section 2\n"
    "à celle de la section 3 : elles doivent pointer vers le MÊME fichier.\n"
    "Branche ensuite ce fichier (et le flag values_are_mm) dans run-viz.py."
)
