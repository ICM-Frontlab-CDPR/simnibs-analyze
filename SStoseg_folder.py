"""
synthstroke2seg_folder.py
-------------------------
Copie les masques de lésion SynthStroke dans un sous-dossier ``_lesions/`` de
chaque head-model SimNIBS (``m2m_<id>``), pour que le reader puisse les linker.

    source :  <source_base>/sub-<id>/*.nii.gz
    dest   :  <dest_base>/<id>/m2m_<id>/_lesions/*.nii.gz

Subtilité : l'ID a un préfixe ``sub-`` côté source mais pas côté m2m.

Exemples
--------
    # prévisualiser (rien n'est copié) :
    python synthstroke2seg_folder.py --dry-run

    # copier pour de vrai :
    python synthstroke2seg_folder.py

    # forcer l'écrasement + vérifier que les NIfTI se chargent :
    python synthstroke2seg_folder.py --overwrite --verify

    # un sous-ensemble de sujets :
    python synthstroke2seg_folder.py --subjects 0001 0002 sub-0008
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# ── Défauts (dataset hemianotACS) — surchargeables en CLI ────────────
DEFAULT_SOURCE_BASE = Path(
    "/network/iss/levy/raw/valerocabre/hemianotACS/Data/derivatives/mri/"
    "0-lesion-synthstroke-masks-SS"
)
DEFAULT_DEST_BASE = Path(
    "/network/iss/levy/raw/valerocabre/hemianotACS/Data/derivatives/mri/"
    "1-simnibs-preps-maskSS"
)
LESIONS_SUBDIR = "_lesions"
DEFAULT_PATTERN = "T1_brain_lesion*.nii.gz"  # native + mni uniquement


def norm_id(raw: str) -> str:
    """'sub-0001' | '0001' → '0001' (ID tel qu'utilisé dans les dossiers m2m)."""
    return raw.strip().removeprefix("sub-")


@dataclass
class SubjectPlan:
    sub_id: str  # '0001'
    source_dir: Path  # .../0-lesion-.../sub-0001
    m2m_dir: Path  # .../1-.../0001/m2m_0001
    dest_dir: Path  # .../m2m_0001/_lesions
    files: list[Path]  # fichiers à copier


def discover_subjects(source_base: Path, explicit: list[str] | None) -> list[str]:
    """IDs à traiter : liste explicite, sinon auto-découverte des ``sub-*``."""
    if explicit:
        return [norm_id(s) for s in explicit]
    found = sorted(norm_id(p.name) for p in source_base.glob("sub-*") if p.is_dir())
    return found


def build_plan(
    sub_id: str,
    source_base: Path,
    dest_base: Path,
    pattern: str,
) -> SubjectPlan | str:
    """Construit le plan de copie d'un sujet, ou renvoie une string d'erreur."""
    source_dir = source_base / f"sub-{sub_id}"
    m2m_dir = dest_base / sub_id / f"m2m_{sub_id}"
    dest_dir = m2m_dir / LESIONS_SUBDIR

    if not source_dir.is_dir():
        return f"{sub_id}: source absente → {source_dir}"
    if not m2m_dir.is_dir():
        return f"{sub_id}: m2m absent → {m2m_dir}"

    files = sorted(source_dir.glob(pattern))
    if not files:
        return f"{sub_id}: aucun fichier '{pattern}' dans {source_dir}"

    return SubjectPlan(sub_id, source_dir, m2m_dir, dest_dir, files)


def verify_nifti(path: Path) -> str | None:
    """Charge le NIfTI pour valider (header + shape). Renvoie une erreur ou None."""
    try:
        import nibabel as nib

        img = nib.load(str(path))
        _ = img.shape  # force la lecture du header
        return None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def copy_subject(
    plan: SubjectPlan,
    *,
    overwrite: bool,
    dry_run: bool,
    verify: bool,
) -> tuple[int, int, list[str]]:
    """Copie les fichiers d'un sujet. Renvoie (copiés, sautés, erreurs)."""
    copied, skipped, errors = 0, 0, []

    if not dry_run:
        plan.dest_dir.mkdir(parents=True, exist_ok=True)

    for src in plan.files:
        dst = plan.dest_dir / src.name
        if dst.exists() and not overwrite:
            print(f"    ↷ existe déjà (skip) : {dst.name}")
            skipped += 1
            continue

        if dry_run:
            print(f"    [dry-run] copierait {src.name} → {dst}")
            copied += 1
            continue

        try:
            shutil.copy2(src, dst)  # copy2 préserve les métadonnées
        except Exception as e:  # noqa: BLE001
            errors.append(f"{plan.sub_id}/{src.name}: {type(e).__name__}: {e}")
            continue

        if verify:
            err = verify_nifti(dst)
            if err is not None:
                errors.append(f"{plan.sub_id}/{src.name}: NIfTI invalide — {err}")
                continue

        print(f"    ✓ {src.name}  ({src.stat().st_size / 1e6:.1f} Mo)")
        copied += 1

    # provenance : trace d'où viennent les fichiers (utile 6 mois plus tard)
    if not dry_run and copied:
        stamp = datetime.now().isoformat(timespec="seconds")
        (plan.dest_dir / ".copied_from.txt").write_text(
            f"{stamp}\nsource: {plan.source_dir}\n"
            + "\n".join(f.name for f in plan.files)
            + "\n"
        )

    return copied, skipped, errors


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source-base", type=Path, default=DEFAULT_SOURCE_BASE)
    ap.add_argument("--dest-base", type=Path, default=DEFAULT_DEST_BASE)
    ap.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"glob des masques à copier (défaut: {DEFAULT_PATTERN})",
    )
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
        help="charger chaque NIfTI copié pour valider (nibabel requis)",
    )
    args = ap.parse_args()

    print("=" * 66)
    print("synthstroke2seg_folder")
    print("=" * 66)
    print(f"source_base : {args.source_base}")
    print(f"dest_base   : {args.dest_base}")
    print(f"pattern     : {args.pattern}")
    print(
        f"mode        : {'DRY-RUN' if args.dry_run else 'COPY'}"
        f"{' +overwrite' if args.overwrite else ''}"
        f"{' +verify' if args.verify else ''}"
    )

    if not args.source_base.is_dir():
        raise SystemExit(f"[erreur] source_base introuvable : {args.source_base}")
    if not args.dest_base.is_dir():
        raise SystemExit(f"[erreur] dest_base introuvable : {args.dest_base}")

    subject_ids = discover_subjects(args.source_base, args.subjects)
    if not subject_ids:
        raise SystemExit(f"[erreur] aucun sujet 'sub-*' dans {args.source_base}")
    print(f"sujets      : {len(subject_ids)} → {subject_ids}")

    total_copied, total_skipped = 0, 0
    problems: list[str] = []
    all_errors: list[str] = []

    for sub_id in subject_ids:
        print(f"\n── {sub_id} " + "─" * 50)
        plan = build_plan(sub_id, args.source_base, args.dest_base, args.pattern)
        if isinstance(plan, str):  # erreur de plan
            print(f"    ⚠ {plan}")
            problems.append(plan)
            continue

        print(f"    source : {plan.source_dir}")
        print(f"    dest   : {plan.dest_dir}")
        c, s, errs = copy_subject(
            plan, overwrite=args.overwrite, dry_run=args.dry_run, verify=args.verify
        )
        total_copied += c
        total_skipped += s
        all_errors.extend(errs)

    # ── Récapitulatif ────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print(
        f"RÉSUMÉ : {total_copied} copié(s), {total_skipped} sauté(s), "
        f"{len(problems)} sujet(s) sans copie, {len(all_errors)} erreur(s)."
    )
    if problems:
        print("\nSujets ignorés :")
        for p in problems:
            print(f"  ⚠ {p}")
    if all_errors:
        print("\nErreurs :")
        for e in all_errors:
            print(f"  ✗ {e}")
    print("=" * 66)

    # code de sortie non nul si des erreurs réelles (pas les simples skips)
    if all_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
