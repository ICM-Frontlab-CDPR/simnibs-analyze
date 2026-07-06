"""
SS_orig_toseg_folder.py
-----------------------
Copie les masques de lésion dans ``_lesions/`` de chaque head-model SimNIBS.

Priorité de source (par sujet) :
  1. Lésion originale normalisée (espace MNI) :
         <orig_base>/sub-<id>/Lesion_normalisee/*.nii.gz
     → copiée telle quelle comme ``T1_brain_lesion_mni.nii.gz``
     → warpée en espace natif comme ``T1_brain_lesion.nii.gz``
       (via Conform2MNI_nonl.nii.gz du head-model SimNIBS)

  2. Fallback — sortie SynthStroke :
         <ss_base>/sub-<id>/T1_brain_lesion*.nii.gz
     → copiée telle quelle (native + mni déjà présents)

Exemples
--------
    python SS_orig_toseg_folder.py --dry-run
    python SS_orig_toseg_folder.py --overwrite --verify
    python SS_orig_toseg_folder.py --subjects 0001 0002 sub-0008
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

# ── Défauts (dataset hemianotACS) — surchargeables en CLI ────────────
DEFAULT_ORIG_BASE = Path(
    "/Volumes/levy/raw/valerocabre/hemianotACS/Data/derivatives/mri/"
    "0-lesion-masks-original-clean"
)
DEFAULT_SS_BASE = Path(
    "/Volumes/levy/raw/valerocabre/hemianotACS/Data/derivatives/mri/"
    "0-lesion-synthstroke-masks-SS"
)
DEFAULT_DEST_BASE = Path(
    "/Volumes/levy/raw/valerocabre/hemianotACS/Data/derivatives/mri/"
    "1-simnibs-preps-maskSS"
)
LESIONS_SUBDIR = "_lesions"
ORIG_SUBDIR = "Lesion_normalisee"  # sous-dossier dans orig_base/sub-<id>/
SS_PATTERN = "T1_brain_lesion*.nii.gz"

# noms cibles dans _lesions/ (attendus par le reader)
NAME_NATIVE = "T1_brain_lesion.nii.gz"
NAME_MNI = "T1_brain_lesion_mni.nii.gz"


def norm_id(raw: str) -> str:
    """'sub-0001' | '0001' → '0001' (ID tel qu'utilisé dans les dossiers m2m)."""
    return raw.strip().removeprefix("sub-")


# ── Warp MNI mask → native ────────────────────────────────────────────


def _warp_mni_to_native(mask_mni_path: Path, m2m_dir: Path) -> nib.Nifti1Image:
    """Pull an MNI binary mask onto the subject grid via the deformation field."""
    warp_path = m2m_dir / "toMNI" / "Conform2MNI_nonl.nii.gz"
    if not warp_path.exists():
        raise FileNotFoundError(f"Warp introuvable : {warp_path}")
    warp_img = nib.load(str(warp_path))
    warp_coords = np.squeeze(warp_img.get_fdata()).reshape(-1, 3)

    mask_mni = nib.load(str(mask_mni_path))
    inv_aff = np.linalg.inv(mask_mni.affine)
    vox = nib.affines.apply_affine(inv_aff, warp_coords).T  # (3, N)
    warped = map_coordinates(mask_mni.get_fdata(), vox, order=0, cval=0)
    data = warped.reshape(warp_img.shape[:3]).astype(np.uint8)
    return nib.Nifti1Image(data, warp_img.affine)


# ── Helpers ───────────────────────────────────────────────────────────


def verify_nifti(path: Path) -> str | None:
    try:
        img = nib.load(str(path))
        _ = img.shape
        return None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def _write_provenance(dest_dir: Path, source_path: Path, source_type: str) -> None:
    stamp = datetime.now().isoformat(timespec="seconds")
    (dest_dir / ".copied_from.txt").write_text(
        f"{stamp}\nsource_type: {source_type}\npath: {source_path}\n"
    )


def discover_subjects(
    orig_base: Path, ss_base: Path, explicit: list[str] | None
) -> list[str]:
    if explicit:
        return [norm_id(s) for s in explicit]
    ids: set[str] = set()
    for base in (orig_base, ss_base):
        if base.is_dir():
            ids |= {norm_id(p.name) for p in base.glob("sub-*") if p.is_dir()}
    return sorted(ids)


# ── Traitement par sujet ──────────────────────────────────────────────


def process_subject(
    sub_id: str,
    orig_base: Path,
    ss_base: Path,
    dest_base: Path,
    *,
    overwrite: bool,
    dry_run: bool,
    verify: bool,
) -> tuple[int, int, list[str], str]:
    copied, skipped, errors = 0, 0, []
    m2m_dir = dest_base / sub_id / f"m2m_{sub_id}"
    dest_dir = m2m_dir / LESIONS_SUBDIR

    if not m2m_dir.is_dir():
        return 0, 0, [f"{sub_id}: m2m absent → {m2m_dir}"], "none"

    # ── Source 1 : lésion originale normalisée (MNI) ──────────────────
    orig_dir = orig_base / f"sub-{sub_id}" / ORIG_SUBDIR
    orig_files = sorted(orig_dir.glob("*.nii.gz")) if orig_dir.is_dir() else []

    if orig_files:
        mni_src = orig_files[0]
        print(f"    [orig] source MNI : {mni_src.name}")
        dst_mni = dest_dir / NAME_MNI
        dst_nat = dest_dir / NAME_NATIVE

        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)

        # MNI : copie directe
        if dst_mni.exists() and not overwrite:
            print(f"    ↷ existe déjà (skip) : {NAME_MNI}")
            skipped += 1
        elif dry_run:
            print(f"    [dry-run] copierait {mni_src.name} → {dst_mni}")
            copied += 1
        else:
            try:
                shutil.copy2(mni_src, dst_mni)
                if verify and (err := verify_nifti(dst_mni)):
                    errors.append(f"{sub_id}/{NAME_MNI}: {err}")
                else:
                    print(f"    ✓ {NAME_MNI}  ({dst_mni.stat().st_size / 1e6:.1f} Mo)")
                    copied += 1
            except Exception as e:  # noqa: BLE001
                errors.append(f"{sub_id}/{NAME_MNI}: {type(e).__name__}: {e}")

        # Native : warp MNI → natif
        if dst_nat.exists() and not overwrite:
            print(f"    ↷ existe déjà (skip) : {NAME_NATIVE}")
            skipped += 1
        elif dry_run:
            print(f"    [dry-run] warperait {mni_src.name} → {dst_nat}")
            copied += 1
        else:
            try:
                nat_img = _warp_mni_to_native(mni_src, m2m_dir)
                nib.save(nat_img, str(dst_nat))
                if verify and (err := verify_nifti(dst_nat)):
                    errors.append(f"{sub_id}/{NAME_NATIVE}: {err}")
                else:
                    print(
                        f"    ✓ {NAME_NATIVE}  ({dst_nat.stat().st_size / 1e6:.1f} Mo)"
                    )
                    copied += 1
            except Exception as e:  # noqa: BLE001
                errors.append(f"{sub_id}/{NAME_NATIVE} (warp): {type(e).__name__}: {e}")

        if not dry_run and copied:
            _write_provenance(dest_dir, mni_src, "orig-normalized")

        return copied, skipped, errors, "orig"

    # ── Source 2 : fallback SynthStroke ──────────────────────────────
    ss_dir = ss_base / f"sub-{sub_id}"
    if not ss_dir.is_dir():
        return 0, 0, [f"{sub_id}: aucune source (orig absent, SS absent)"], "none"

    ss_files = sorted(ss_dir.glob(SS_PATTERN))
    if not ss_files:
        return 0, 0, [f"{sub_id}: aucun fichier '{SS_PATTERN}' dans {ss_dir}"], "none"

    print(f"    [fallback SS] {len(ss_files)} fichier(s) depuis {ss_dir.name}")

    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)

    for src in ss_files:
        dst = dest_dir / src.name
        if dst.exists() and not overwrite:
            print(f"    ↷ existe déjà (skip) : {dst.name}")
            skipped += 1
            continue
        if dry_run:
            print(f"    [dry-run] copierait {src.name} → {dst}")
            copied += 1
            continue
        try:
            shutil.copy2(src, dst)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{sub_id}/{src.name}: {type(e).__name__}: {e}")
            continue
        if verify and (err := verify_nifti(dst)):
            errors.append(f"{sub_id}/{src.name}: NIfTI invalide — {err}")
            continue
        print(f"    ✓ {src.name}  ({src.stat().st_size / 1e6:.1f} Mo)")
        copied += 1

    if not dry_run and copied:
        _write_provenance(dest_dir, ss_dir, "synthstroke")

    return copied, skipped, errors, "ss"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--orig-base",
        type=Path,
        default=DEFAULT_ORIG_BASE,
        help="dossier des lésions originales normalisées (priorité 1)",
    )
    ap.add_argument(
        "--ss-base",
        type=Path,
        default=DEFAULT_SS_BASE,
        help="dossier des sorties SynthStroke (fallback)",
    )
    ap.add_argument("--dest-base", type=Path, default=DEFAULT_DEST_BASE)
    ap.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="IDs explicites ('0001' ou 'sub-0001'); sinon auto-découverte",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="écraser les fichiers déjà présents (défaut: skip)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="afficher ce qui serait fait, sans rien copier",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="charger chaque NIfTI produit pour valider (nibabel)",
    )
    args = ap.parse_args()

    print("=" * 66)
    print("SS_orig_toseg_folder")
    print("=" * 66)
    print(f"orig_base   : {args.orig_base}")
    print(f"ss_base     : {args.ss_base}")
    print(f"dest_base   : {args.dest_base}")
    print(
        f"mode        : {'DRY-RUN' if args.dry_run else 'COPY'}"
        f"{' +overwrite' if args.overwrite else ''}"
        f"{' +verify' if args.verify else ''}"
    )

    if not args.dest_base.is_dir():
        raise SystemExit(f"[erreur] dest_base introuvable : {args.dest_base}")

    subject_ids = discover_subjects(args.orig_base, args.ss_base, args.subjects)
    if not subject_ids:
        raise SystemExit("[erreur] aucun sujet trouvé dans orig_base ni ss_base")
    print(f"sujets      : {len(subject_ids)} → {subject_ids}")

    total_copied, total_skipped = 0, 0
    all_errors: list[str] = []
    source_log: dict[str, list[str]] = {"orig": [], "ss": [], "none": []}

    for sub_id in subject_ids:
        print(f"\n── {sub_id} " + "─" * 50)
        c, s, errs, src_type = process_subject(
            sub_id,
            args.orig_base,
            args.ss_base,
            args.dest_base,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            verify=args.verify,
        )
        total_copied += c
        total_skipped += s
        all_errors.extend(errs)
        source_log[src_type].append(sub_id)

    print("\n" + "=" * 66)
    print(
        f"RÉSUMÉ : {total_copied} copié(s), {total_skipped} sauté(s), "
        f"{len(all_errors)} erreur(s)."
    )
    if source_log["orig"]:
        print(f"  orig (priorité)        : {source_log['orig']}")
    if source_log["ss"]:
        print(f"  synthstroke (fallback) : {source_log['ss']}")
    if source_log["none"]:
        print(f"  aucune source          : {source_log['none']}")
    if all_errors:
        print("\nErreurs :")
        for e in all_errors:
            print(f"  ✗ {e}")
    print("=" * 66)

    if all_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
