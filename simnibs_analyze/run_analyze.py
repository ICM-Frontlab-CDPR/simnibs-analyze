"""
run_analyze.py
--------------
SimNIBS e-field analysis pipeline, driven by a validated PipelineConfig and
built entirely on **simnibs-reader** (no direct nibabel/nilearn plumbing here).

Per (subject × condition × mode) it:
  1. locates the simulation folder via the reader,
  2. extracts the target ROI (sphere / atlas / mask) on the chosen space,
  3. post-processes (smooth + outlier removal),
  4. computes intra-ROI stats + extra-ROI (complement) stats + intra/extra ratio,
  5. appends one row matching all_features_space-<space>.csv.

Then it runs the inter/intra-subject analysis + clustering (steps in
``steps/analysis.py``) and writes the summary CSVs.

    simnibs-analyze --config docs/examples/config/config-analyze_htacs.yaml
    simnibs-analyze --config ... --skip-features   # reuse existing CSV
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import simnibs_reader as snr
from simnibs_reader.nifti.stats import compute_ratio

from simnibs_analyze._config_schema_analyze import (
    PipelineConfig,
    load_and_validate,
)
from simnibs_analyze._logging import get_logger
from simnibs_analyze.steps.analysis import Analysis

logger = get_logger(__name__)

SPACE_SUFFIX = {"mni": "space-mni", "native": "space-native"}


# ─────────────────────────────────────────────────────────────────────
# Path helpers (kept here so the reader stays generic)
# ─────────────────────────────────────────────────────────────────────


def _simu_root(cfg: PipelineConfig) -> Path:
    """Root folder holding <subject>/simulations/ (new or legacy layout)."""
    if cfg.paths.sim_base is not None:
        return cfg.paths.sim_base
    return cfg.paths.simnibs_output  # legacy single-root


def _preps_root(cfg: PipelineConfig) -> Path | None:
    """Root folder holding <subject>/m2m_<subject>/."""
    if cfg.paths.seg_base is not None:
        return cfg.paths.seg_base
    return cfg.paths.simnibs_output


def features_csv_path(cfg: PipelineConfig) -> Path:
    return cfg.paths.out_root / f"all_features_{SPACE_SUFFIX[cfg.space]}.csv"


def find_simulation_dir(
    simu_root: Path, subject: str, condition: str, mode: str
) -> Path | None:
    """Locate the simulation/optimization folder for one (subject, condition, mode).

    Folder names look like ``simulation_simulation_<cond>_<study>_<hash>`` or
    ``optimization_...``.

    ROI keys cannot contain underscores (they clash with the output filename
    convention), while SimNIBS folders often do — ``ips-left`` as a key,
    ``ips_left`` on disk. Both spellings are tried, so the two never have to be
    declared separately.

    Parameters
    ----------
    simu_root : Path
        Root holding ``<subject>/simulations/``.
    subject, condition, mode : str
        Subject ID, ROI/condition key, and ``simulation`` or ``optimization``.

    Returns
    -------
    Path or None
        First matching folder, or None when the subject or condition is absent.
    """
    subj_dir = simu_root / subject
    if not subj_dir.is_dir():
        return None

    # '-' and '_' are interchangeable in folder names
    tokens = dict.fromkeys([condition, condition.replace("-", "_")])
    for token in tokens:
        # be permissive: match the mode prefix + condition token anywhere after it
        matches = sorted(subj_dir.rglob(f"{mode}_*{token}*"))
        if matches:
            return matches[0]
    return None


# ─────────────────────────────────────────────────────────────────────
# ROI extraction (delegates entirely to the reader)
# ─────────────────────────────────────────────────────────────────────


def _get_efield(sim, space: str):
    """Return the magnE EField for the requested space via the reader."""
    return sim.magnE if space == "mni" else sim.magnE_native


def _extract_roi(efield, roi_def, radius_mm: float):
    """Build a reader ROI from a config ROI definition (sphere or atlas).

    Config coordinates are always MNI, whatever space the analysis runs in.
    Saying so explicitly lets the reader warp them onto the subject grid when
    the e-field is in native space; without it the sphere landed outside the
    head and produced an empty mask.
    """
    if roi_def.method == "sphere":
        return efield.get_roi(
            coords=list(roi_def.coords), radius=radius_mm, coords_space="mni"
        )
    # atlas — atlases are defined in MNI, the reader warps when needed
    return efield.get_roi(atlas=roi_def.atlas, region=roi_def.regions)


def process_subject_condition(
    cfg: PipelineConfig,
    subject: str,
    condition: str,
    mode: str,
) -> dict | None:
    """One CSV row for (subject, condition, mode), or None if unavailable."""
    simu_root = _simu_root(cfg)
    roi_def = cfg.target_generation.rois[condition]
    sim_dir = find_simulation_dir(simu_root, subject, condition, mode)
    if sim_dir is None:
        logger.warning(f"{subject}/{condition}/{mode}: no simulation folder — skipped")
        return None

    try:
        sim = snr.simulation(str(sim_dir))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"{subject}/{condition}/{mode}: reader error — {e}")
        return None

    # Attach the segmentation in BOTH spaces. It resolves the brain mask for
    # complement() (MNI or native) and carries the deformation field needed to
    # warp MNI targets onto the subject grid.
    preps = _preps_root(cfg)
    m2m = preps / subject / f"m2m_{subject}" if preps else None
    if m2m and m2m.is_dir():
        sim.set_segmentation(snr.segmentation(str(m2m)))
    else:
        logger.warning(
            f"{subject}: no m2m folder under {preps} — "
            "extra-ROI stats and MNI→native warping will be unavailable"
        )

    try:
        efield = _get_efield(sim, cfg.space)
        roi = _extract_roi(efield, roi_def, cfg.target_generation.radius_mm)

        pp = cfg.preprocessing
        roi = roi.postprocess(
            smooth_fwhm=pp.smooth_fwhm,
            outlier_method=("zscore" if pp.outlier_method == "zscore" else "iqr"),
            portion=pp.portion,
        )
        intra = roi.stats()

        # extra-ROI (brain minus ROI) — needs a brain mask or attached seg
        extra = None
        try:
            extra = roi.complement().stats()
        except Exception as e:  # noqa: BLE001
            logger.info(f"{subject}/{condition}/{mode}: complement unavailable ({e})")

    except Exception as e:  # noqa: BLE001
        logger.warning(f"{subject}/{condition}/{mode}: extraction failed — {e}")
        return None

    row: dict = {
        "efield_path": str(efield.path),
        "roi_path": "",  # reserved (mask not persisted here) — kept for CSV parity
        "subject": subject,
        "condition": f"{condition}_{mode}",
        "space": cfg.space,
    }
    # intra stats (align with CSV columns: mean..max, n_voxels)
    for k in cfg.feature_extraction.metrics:
        row[k] = intra.get(k)
    row["n_voxels"] = intra.get("n_voxels")

    # extra stats + ratio
    if extra is not None:
        for k in cfg.feature_extraction.metrics:
            row[f"extra_{k}"] = extra.get(k)
        row["extra_n_voxels"] = extra.get("n_voxels")
        row["efield_ratio_mean"] = (
            compute_ratio([intra["mean"]], [extra["mean"]], method="mean")
            if extra.get("mean") not in (None, 0)
            else float("nan")
        )

    return row


# ─────────────────────────────────────────────────────────────────────
# Steps
# ─────────────────────────────────────────────────────────────────────


def build_features(cfg: PipelineConfig) -> Path:
    """Step 1+2: extract ROI features for the whole cohort → features CSV."""
    logger.info(f"Feature extraction in {cfg.space.upper()} space")
    rows: list[dict] = []
    ok = fail = 0

    for subject in cfg.subjects:
        for condition in cfg.stim_conditions:
            for mode in cfg.mode:
                row = process_subject_condition(cfg, subject, condition, mode)
                if row is not None:
                    rows.append(row)
                    ok += 1
                else:
                    fail += 1

    if not rows:
        raise SystemExit(
            "Aucune feature extraite — vérifie paths / conditions / space."
        )

    out = features_csv_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info(f"✓ {len(rows)} rows → {out}  (ok={ok}, skipped={fail})")
    return out


def run_analysis(cfg: PipelineConfig, features_csv: Path) -> None:
    """Step 3: inter/intra-subject summaries + clustering."""
    df = pd.read_csv(features_csv)
    logger.info(f"Analysis on {len(df)} rows from {features_csv.name}")

    a = cfg.analysis
    results_dir = cfg.paths.out_root
    tag = SPACE_SUFFIX[cfg.space]

    # inter-subject summary
    try:
        inter = Analysis(df).inter_subject_summary(
            metric=a.metric, condition_col=a.condition_col
        )
        inter.to_csv(results_dir / f"inter_subject_{tag}.csv", index=False)
        logger.info("✓ inter-subject summary")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"inter-subject summary failed: {e}")

    # intra-subject simulation vs optimization (only if both modes present)
    conditions = set(df[a.condition_col].unique())
    for cond in cfg.stim_conditions:
        sim_c, opt_c = f"{cond}_simulation", f"{cond}_optimization"
        if sim_c not in conditions or opt_c not in conditions:
            continue
        try:
            diff = Analysis(
                df[df[a.condition_col].isin([sim_c, opt_c])]
            ).intra_subject_diff(
                metric=a.metric,
                subject_col=a.subject_col,
                condition_col=a.condition_col,
                cond_a=sim_c,
                cond_b=opt_c,
            )
            diff.to_csv(results_dir / f"intra_subject_{cond}_{tag}.csv", index=False)
            logger.info(f"✓ intra-subject diff: {cond}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"intra-subject diff {cond}: {e}")

    # clustering on efield_ratio_<method>
    cl = a.clustering
    ratio_col = f"efield_ratio_{cl.method}"
    if ratio_col in df.columns:
        try:
            clustered = Analysis(df).assign_clusters(
                method=cl.method,
                specificity_threshold=cl.specificity_threshold,
                intensity_col=cl.intensity_col,
            )
            clustered.to_csv(results_dir / f"clusters_{tag}.csv", index=False)
            logger.info(f"✓ clusters ({clustered['cluster'].value_counts().to_dict()})")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"clustering failed: {e}")
    else:
        logger.warning(f"'{ratio_col}' absent — clustering skipped")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="simnibs-analyze",
        description="SimNIBS e-field analysis pipeline",
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument(
        "--skip-features",
        action="store_true",
        help="reuse the existing all_features CSV",
    )
    ap.add_argument("--skip-analysis", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_and_validate(args.config)
    logger.info(
        f"Subjects={len(cfg.subjects)} conditions={cfg.stim_conditions} "
        f"modes={cfg.mode} space={cfg.space}"
    )
    cfg.paths.out_root.mkdir(parents=True, exist_ok=True)

    # Step 1+2 — features
    if args.skip_features:
        features_csv = features_csv_path(cfg)
        if not features_csv.exists():
            logger.error(f"--skip-features but CSV missing: {features_csv}")
            return 1
        logger.info(f"Reusing {features_csv}")
    else:
        features_csv = build_features(cfg)

    # Step 3 — analysis
    if not args.skip_analysis:
        run_analysis(cfg, features_csv)

    logger.info("Pipeline complete.")
    return 0


def cli(argv: Iterable[str] | None = None) -> int:
    """Entry point for the ``simnibs-analyze`` command."""
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(cli())
