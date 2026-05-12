#!/usr/bin/env python3
"""
Main SimNIBS e-field analysis pipeline.
Orchestrates target generation, preprocessing, feature extraction, analysis and visualisation.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Dict, List

# Allow running as `python run.py` from within simnibs_analyze/
# (has no effect when imported as part of the installed package)
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    __package__ = "simnibs_analyze"  # noqa: A001

import nibabel as nib
import pandas as pd

from simnibs_analyze.steps._0_anatomical_preparer import AnatomicalPreparer
from simnibs_analyze.steps._1_preprocessing import Preprocessor
from simnibs_analyze.steps._2_features_extraction import FeatureExtractor
from simnibs_analyze.steps._3_analysis import Analysis
from simnibs_analyze.steps._4_viz import Visualizer
from simnibs_analyze._pipeline_io import (
    SPACE_MNI,
    SPACE_NATIVE,
    check_output,
    find_efield_files,
    find_simulation_dirs,
    get_analysis_dir,
    get_clusters_csv_path,
    get_features_csv_path,
    get_inter_subject_summary_csv_path,
    get_intra_subject_diff_csv_path,
    get_preproc_dir,
    get_preproc_paths,
    get_roi_mask_path,
    get_subject_paths,
    get_subject_paths_from_config,
    load_config,
    save_dataframe,
    save_nifti,
    save_rows,
    space_tag,
)
from simnibs_analyze._logging import get_logger
from simnibs_analyze._config import PipelineConfig

logger = get_logger(__name__)


def process_subject_condition(
    subject: str,
    condition: str,
    mode: str,
    config: PipelineConfig,
    skip_preprocessing: bool = False,
    space: str = SPACE_MNI,
    if_exists: str = "overwrite",
) -> List[Dict]:
    """
    Preprocess and extract features from all e-fields for a subject/condition/mode.

    Parameters
    ----------
    space : str
        ``'mni'`` (default) or ``'native'`` — working space for e-fields and ROI masks.

    Returns
    -------
    List[Dict]
        Extracted feature rows (one per e-field file found).
    """
    results: List[Dict] = []
    subject_paths = get_subject_paths_from_config(config.paths, subject)
    subject_dir = subject_paths["subject_dir"]

    if not subject_dir.exists():
        logger.warning(f"Subject directory not found: {subject_dir}")
        return results

    simulation_dirs = find_simulation_dirs(
        subject_dir,
        condition,
        mode,
        folder_pattern=config.target_generation.rois[condition].folder_pattern,
    )
    if not simulation_dirs:
        logger.warning(f"No simulation found for {subject}/{condition}/{mode}")
        return results

    try:
        from simnibs_analyze._pipeline_io import get_simu_root
        roi_mask_path = get_roi_mask_path(
            get_simu_root(config.paths), condition, space=space, subject=subject
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return results

    for sim_dir in simulation_dirs:
        for efield_path in find_efield_files(sim_dir, mode, space=space):
            preproc_dir = get_preproc_dir(sim_dir, mode, space=space)
            base_name = efield_path.stem.replace(".nii", "")
            paths = get_preproc_paths(preproc_dir, base_name, condition)

            preproc_kwargs = dict(
                smooth_fwhm=config.preprocessing.smooth_fwhm,
                outlier_method=config.preprocessing.outlier_method,
                portion=config.preprocessing.portion,
            )

            # ── Preprocessing INTRA-ROI ──────────────────────────────────
            if skip_preprocessing:
                if not paths["intra_cleaned"].exists():
                    logger.warning(
                        f"Preprocessed file not found, skipping: {paths['intra_cleaned']}"
                    )
                    continue
                logger.info(f"Using existing file: {paths['intra_cleaned'].name}")
            else:
                if paths["intra_cleaned"].exists() and paths["intra_masked"].exists():
                    if if_exists == "skip":
                        logger.info(
                            f"Already preprocessed, skipping: {paths['intra_cleaned'].name}"
                        )
                        pass  # will fall through to feature extraction below
                    elif if_exists == "error":
                        logger.error(
                            f"Output exists (if_exists='error'): {paths['intra_cleaned'].name}"
                        )
                        return []
                    else:
                        try:
                            preproc = Preprocessor(**preproc_kwargs).run(
                                efield_path, roi_mask_path
                            )
                            save_nifti(preproc.masked_img, paths["intra_masked"])
                            save_nifti(preproc.cleaned_img, paths["intra_cleaned"])
                            logger.info(
                                f"✓ Intra-ROI preprocessing: {paths['intra_cleaned'].name}"
                            )
                        except Exception as e:
                            logger.error(
                                f"✗ Intra-ROI preprocessing failed ({efield_path.name}): {e}"
                            )
                            continue
                else:
                    try:
                        preproc = Preprocessor(**preproc_kwargs).run(
                            efield_path, roi_mask_path
                        )
                        save_nifti(preproc.masked_img, paths["intra_masked"])
                        save_nifti(preproc.cleaned_img, paths["intra_cleaned"])
                        logger.info(
                            f"✓ Intra-ROI preprocessing: {paths['intra_cleaned'].name}"
                        )
                    except Exception as e:
                        logger.error(
                            f"✗ Intra-ROI preprocessing failed ({efield_path.name}): {e}"
                        )
                        continue

            # ── Preprocessing EXTRA-ROI ──────────────────────────────────
            if skip_preprocessing:
                if not paths["extra_cleaned"].exists():
                    logger.warning(
                        f"Extra preprocessed file not found, skipping: {paths['extra_cleaned']}"
                    )
                    continue
            else:
                if paths["extra_cleaned"].exists() and paths["extra_masked"].exists():
                    if if_exists == "skip":
                        logger.info(
                            f"Already preprocessed, skipping: {paths['extra_cleaned'].name}"
                        )
                        pass
                    elif if_exists == "error":
                        logger.error(
                            f"Output exists (if_exists='error'): {paths['extra_cleaned'].name}"
                        )
                        return []
                    else:
                        try:
                            extra_mask = Preprocessor.build_extra_mask(roi_mask_path)
                            extra_masked_img = (
                                Preprocessor(**preproc_kwargs)
                                .run(efield_path, extra_mask)
                                .masked_img
                            )
                            save_nifti(extra_masked_img, paths["extra_masked"])
                            save_nifti(
                                extra_masked_img, paths["extra_cleaned"]
                            )  # cleaned = masked
                            logger.info(
                                f"✓ Extra-ROI preprocessing: {paths['extra_masked'].name}"
                            )
                        except Exception as e:
                            logger.error(
                                f"✗ Extra-ROI preprocessing failed ({efield_path.name}): {e}"
                            )
                            continue
                else:
                    try:
                        extra_mask = Preprocessor.build_extra_mask(roi_mask_path)
                        extra_masked_img = (
                            Preprocessor(**preproc_kwargs)
                            .run(efield_path, extra_mask)
                            .masked_img
                        )
                        save_nifti(extra_masked_img, paths["extra_masked"])
                        save_nifti(
                            extra_masked_img, paths["extra_cleaned"]
                        )  # cleaned = masked
                        logger.info(
                            f"✓ Extra-ROI preprocessing: {paths['extra_masked'].name}"
                        )
                    except Exception as e:
                        logger.error(
                            f"✗ Extra-ROI preprocessing failed ({efield_path.name}): {e}"
                        )
                        continue

            # ── Feature extraction ───────────────────────────────────────
            try:
                row_intra = (
                    FeatureExtractor()
                    .run(
                        paths["intra_cleaned"],
                        roi_path=None,
                        subject=subject,
                        condition=f"{condition}_{mode}",
                    )
                    .row
                )
                row_extra = (
                    FeatureExtractor()
                    .run(
                        paths["extra_cleaned"],
                        roi_path=None,
                        subject=None,
                        condition=None,
                    )
                    .row
                )

                # Merge: intra columns without prefix, extra columns with extra_ prefix
                row = {**row_intra}
                for k in ["mean", "median", "std", "min", "max", "n_voxels"]:
                    if k in row_extra:
                        row[f"extra_{k}"] = row_extra[k]
                row["space"] = space

                # Ratio computed from cleaned values
                intra_mean = row.get("mean", 0.0)
                extra_mean = row.get("extra_mean", 1e-10)
                row["efield_ratio_mean"] = intra_mean / max(float(extra_mean), 1e-10)

                logger.info(
                    f"✓ Features : intra_mean={intra_mean:.6f} | "
                    f"extra n_voxels={row_extra.get('n_voxels','?')} mean={row_extra.get('mean', 'MISSING')!r} | "
                    f"extra_mean={extra_mean:.6e} | "
                    f"ratio={row['efield_ratio_mean']:.4f}"
                )
                results.append(row)
            except Exception as e:
                logger.error(
                    f"✗ Feature extraction failed ({subject}/{condition}/{mode}): {e}"
                )

    return results


def run_analysis(
    features_csv: Path, config: PipelineConfig, space: str, if_exists: str = "overwrite"
) -> None:
    """Inter/intra-subject analysis and simulation vs optimisation scatter plot."""
    logger.step("INTER/INTRA-SUBJECT ANALYSIS")

    results_dir = config.paths.results_dir
    analysis_dir = get_analysis_dir(results_dir, space)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    metric = config.analysis.metric
    subject_col = config.analysis.subject_col
    condition_col = config.analysis.condition_col

    df = pd.read_csv(features_csv)
    logger.info(f"Loaded: {len(df)} rows from {features_csv}")

    # Inter-subject
    inter = Analysis(df).inter_subject_summary(
        metric=metric, condition_col=condition_col
    )
    inter_csv = get_inter_subject_summary_csv_path(results_dir, space)
    if check_output(inter_csv, if_exists):
        save_dataframe(inter, inter_csv, index=False)
        logger.info(f"✓ Inter-subject summary: {inter_csv}")

    # Intra-subject (simulation / optimization pairs)
    conditions = set(df[condition_col].unique())
    for cond in config.stim_conditions:
        sim_cond, opt_cond = f"{cond}_simulation", f"{cond}_optimization"
        if sim_cond not in conditions or opt_cond not in conditions:
            continue
        df_cond = df[df[condition_col].isin([sim_cond, opt_cond])]
        try:
            diff_df = Analysis(df_cond).intra_subject_diff(
                metric=metric,
                subject_col=subject_col,
                condition_col=condition_col,
                cond_a=sim_cond,
                cond_b=opt_cond,
            )
            diff_csv = get_intra_subject_diff_csv_path(results_dir, space, cond)
            if check_output(diff_csv, if_exists):
                save_dataframe(diff_df, diff_csv, index=False)
                logger.info(f"✓ Intra-subject diff: {diff_csv}")
        except Exception as e:
            logger.warning(f"Intra-subject analysis not possible for {cond}: {e}")

    # Clustering
    cl_params = config.analysis.clustering
    cl_method = cl_params.method
    cl_threshold = cl_params.specificity_threshold
    cl_intensity_col = cl_params.intensity_col
    ratio_col = f"efield_ratio_{cl_method}"
    if ratio_col in df.columns:
        try:
            clustered_df = Analysis(df).assign_clusters(
                method=cl_method,
                specificity_threshold=cl_threshold,
                intensity_col=cl_intensity_col,
            )
            # clusters.csv retains all original columns (efield_path, subject,
            # condition, stats…) + cluster — the link to e-fields is therefore direct.
            clusters_csv = get_clusters_csv_path(results_dir, space)
            if check_output(clusters_csv, if_exists):
                save_dataframe(clustered_df, clusters_csv, index=False)
                logger.info(f"✓ Clusters saved: {clusters_csv}")
            dist = clustered_df["cluster"].value_counts().to_dict()
            logger.info(f"  Distribution: {dist}")
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
    else:
        logger.warning(
            f"Column '{ratio_col}' not found in {features_csv.name} — clustering skipped. "
            "Make sure compute_efield_ratio is called during feature extraction."
        )

    # Scatter simulation vs optimization
    Visualizer(analysis_dir, if_exists=if_exists).plot_simulation_vs_optimization(
        df,
        metric=metric,
        subject_col=subject_col,
        condition_col=condition_col,
        output_tag=space,
    )
    logger.info("✓ Simulation vs optimisation scatter plot created")
    logger.step("ANALYSIS COMPLETE")


def run_viz(config: PipelineConfig, space: str, if_exists: str = "overwrite") -> None:
    """Collect paths (I/O) then generate all visualisations."""
    logger.step("GENERATING VISUALISATIONS")

    simnibs_output = config.paths.simnibs_output
    results_dir = config.paths.results_dir
    subjects: List[str] = config.subjects
    conditions: List[str] = config.stim_conditions
    modes: List[str] = config.mode

    viz = Visualizer(
        output_dir=results_dir,
        cmap="hot",
        threshold_percentile=50.0,
        bins=50,
        camera_position="xy",
        if_exists=if_exists,
    )

    # ── Masques ROI ─────────────────────────────────────────────────────
    mni_target_dir = simnibs_output / "mni_target"
    mask_paths = sorted(mni_target_dir.glob("*_mask_space-mni.nii.gz"))
    if space == SPACE_MNI and mask_paths:
        mni_template = (
            nib.load(str(config.paths.mni_template))
            if config.paths.mni_template
            else None
        )
        mask_imgs = [nib.load(str(p)) for p in mask_paths]
        roi_names = [p.name.replace("_mask_space-mni.nii.gz", "") for p in mask_paths]
        viz.visualize_roi_masks(mask_imgs, roi_names, mni_template)
        logger.info("✓ ROI masks visualised")

    # ── 3D e-field figures ────────────────────────────────────────────────────────
    # Brain backgrounds per subject (produced by AnatomicalPreparer.run())
    mni_brain_bg_by_subject: Dict[str, Path] = {}
    subject_brain_bg_by_subject: Dict[str, Path] = {}
    for subject in subjects:
        subject_paths = get_subject_paths_from_config(config.paths, subject)
        mni_bg = subject_paths["subject_target_dir"] / "T1_MNI_brain.nii.gz"
        subj_bg = subject_paths["subject_target_dir"] / "T1_subject_brain.nii.gz"
        if mni_bg.exists():
            mni_brain_bg_by_subject[subject] = mni_bg
        if subj_bg.exists():
            subject_brain_bg_by_subject[subject] = subj_bg

    file_info: Dict = {}
    for mode in modes:
        for condition in conditions:
            for subject in subjects:
                subject_paths = get_subject_paths_from_config(config.paths, subject)
                sim_dirs = find_simulation_dirs(
                    subject_paths["subject_dir"],
                    condition,
                    mode,
                    folder_pattern=config.target_generation.rois[
                        condition
                    ].folder_pattern,
                )
                if not sim_dirs:
                    continue
                efields = find_efield_files(sim_dirs[0], mode, space=space)
                if not efields:
                    continue
                file_info.setdefault((condition, mode), []).append(
                    (subject, efields[0])
                )
    if file_info:
        brain_bgs = (
            mni_brain_bg_by_subject
            if space == SPACE_MNI
            else subject_brain_bg_by_subject
        )
        viz.efields_figures(
            file_info, t1_brain_by_subject=brain_bgs or None, space=space
        )
        logger.info(f"✓ 3D e-field figures generated ({space.upper()})")
    else:
        logger.warning(f"No e-fields found for space={space}, figures skipped")

    # ── Histogrammes preprocessing ───────────────────────────────────────
    intra_data: Dict = {}
    extra_data: Dict = {}
    for subject in subjects:
        for mode in modes:
            for condition in conditions:
                subject_paths = get_subject_paths_from_config(config.paths, subject)
                sim_dirs = find_simulation_dirs(
                    subject_paths["subject_dir"],
                    condition,
                    mode,
                    folder_pattern=config.target_generation.rois[
                        condition
                    ].folder_pattern,
                )
                if not sim_dirs:
                    continue
                preproc_dir = get_preproc_dir(sim_dirs[0], mode, space=space)
                for efield_path in find_efield_files(sim_dirs[0], mode, space=space):
                    base_name = efield_path.stem.replace(".nii", "")
                    p = get_preproc_paths(preproc_dir, base_name, condition)
                    if p["intra_masked"].exists() and p["intra_cleaned"].exists():
                        intra_data.setdefault(subject, []).append(
                            (condition, mode, p["intra_masked"], p["intra_cleaned"])
                        )
                    if p["extra_masked"].exists() and p["extra_cleaned"].exists():
                        extra_data.setdefault(subject, []).append(
                            (condition, mode, p["extra_masked"], p["extra_cleaned"])
                        )
    if intra_data:
        viz.efields_histograms(intra_data, region="intra", space=space)
        logger.info("✓ Intra-ROI histograms generated")
    if extra_data:
        viz.efields_histograms(extra_data, region="extra", space=space)
        logger.info("✓ Extra-ROI histograms generated")
    logger.step("VISUALISATIONS COMPLETE")


def main(
    config_path: Path,
    skip_target_generation: bool = False,
    skip_preprocessing: bool = False,
    skip_features: bool = False,
    skip_analysis: bool = False,
    skip_viz: bool = False,
) -> int:
    """Main pipeline entry point."""
    logger.step("STARTING E-FIELD ANALYSIS PIPELINE")
    logger.info(f"Config: {config_path}")

    config: PipelineConfig = load_config(config_path)
    space = config.space

    logger.info(f"Subjects   : {config.subjects}")
    logger.info(f"Conditions : {config.stim_conditions}")
    logger.info(f"Modes      : {config.mode}")
    logger.info(f"Space      : {space}")

    results_dir = config.paths.results_dir
    from simnibs_analyze._pipeline_io import get_simu_root
    simu_root = get_simu_root(config.paths)

    if_exists = config.running.if_exists
    logger.info(f"if_exists = '{if_exists}'")

    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 0: Setup targets (once, independent of subjects) ────────────
    rois = config.target_generation.rois
    mni_target_dir = simu_root / "mni_target"
    gen = AnatomicalPreparer(
        reference_img_path=config.paths.mni_template,
        radius_mm=config.target_generation.radius_mm,
        mni_brain_mask_path=config.paths.mni_brain_mask,
    )

    if not skip_target_generation:
        logger.step("STEP 0: ROI MASK GENERATION (MNI)")
        existing_masks = all(
            (mni_target_dir / f"{roi}_mask_space-mni.nii.gz").exists() for roi in rois
        )
        if existing_masks and if_exists == "skip":
            logger.info(f"✓ ROI masks already present, skipping: {mni_target_dir}")
        elif existing_masks and if_exists == "error":
            logger.error(
                f"ROI masks already present in {mni_target_dir} (if_exists='error')"
            )
            return 1
        else:
            try:
                gen.setup_mni_rois(rois, mni_target_dir, if_exists=if_exists)
            except Exception as e:
                logger.error(f"✗ Target generation failed: {e}")
                return 1
    else:
        logger.info("ROI mask generation skipped")

    # ── Steps 1+2: Preprocessing + Feature extraction ─────────────────────────
    analysis_dir = get_analysis_dir(results_dir, space)
    features_csv = get_features_csv_path(results_dir, space)

    if skip_features:
        if not features_csv.exists():
            logger.error(
                f"all_features_{space_tag(space)}.csv not found: {features_csv}"
            )
            return 1
        logger.info(f"Feature extraction skipped — using {features_csv}")
    else:
        all_features: List[Dict] = []
        stats = {"total": 0, "success": 0, "failed": 0}

        logger.info(f"Computing in {space.upper()} space")

        for subject in config.subjects:
            logger.step(f"SUBJECT: {subject}")
            subject_paths = get_subject_paths_from_config(config.paths, subject)
            m2m_dir = subject_paths["m2m_dir"]
            subject_target_dir = subject_paths["subject_target_dir"]

            if space == SPACE_NATIVE and skip_target_generation:
                missing = [
                    roi
                    for roi in rois
                    if not (
                        subject_target_dir / f"{roi}_mask_space-native.nii.gz"
                    ).exists()
                ]
                if missing:
                    logger.warning(
                        f"Missing native-space ROI masks for {subject} ({missing}) with --skip-target-generation. "
                        "Subject skipped to avoid ambiguous outputs."
                    )
                    continue

            if not skip_target_generation and m2m_dir.exists():
                # Always generate T1 skull-stripped in both spaces
                gen.run(m2m_dir, subject_target_dir, if_exists=if_exists)

                # If working in native space, generate native-space ROI masks
                if space == SPACE_NATIVE:
                    logger.info("Generating native-space ROI masks...")
                    try:
                        gen.create_subject_roi_from_mni(
                            m2m_dir, subject_target_dir, if_exists=if_exists
                        )
                        logger.info(f"✓ Native-space ROI masks generated for {subject}")
                    except Exception as e:
                        logger.warning(f"Native-space ROI generation failed: {e}")
                        logger.warning(
                            f"Subject {subject} skipped to avoid mixing spaces"
                        )
                        continue
            elif space == SPACE_NATIVE and not m2m_dir.exists():
                logger.warning(
                    f"m2m not found for {subject}: {m2m_dir}. Subject skipped in native space."
                )
                continue

            for condition in config.stim_conditions:
                for mode in config.mode:
                    logger.info(f"--- {subject} / {condition} / {mode} ---")
                    rows = process_subject_condition(
                        subject,
                        condition,
                        mode,
                        config,
                        skip_preprocessing=skip_preprocessing,
                        space=space,
                        if_exists=if_exists,
                    )
                    stats["total"] += 1
                    if rows:
                        stats["success"] += 1
                        all_features.extend(rows)
                    else:
                        stats["failed"] += 1

        if not all_features:
            logger.warning("No features extracted!")
            return 1

        analysis_dir.mkdir(parents=True, exist_ok=True)
        if features_csv.exists() and if_exists == "error":
            logger.error(f"{features_csv.name} already exists (if_exists='error')")
            return 1
        save_rows(all_features, features_csv)
        logger.info(f"✓ {len(all_features)} features saved → {features_csv}")
        logger.info(
            f"  Successes: {stats['success']}/{stats['total']}, failures: {stats['failed']}"
        )

    # ── Step 3: Analysis ──────────────────────────────────────────────────
    if not skip_analysis:
        run_analysis(features_csv, config, space=space, if_exists=if_exists)
    else:
        logger.info("Analysis skipped")

    # ── Step 4: Visualisations ────────────────────────────────────────────────
    if not skip_viz:
        run_viz(config, space=space, if_exists=if_exists)
    else:
        logger.info("Visualisations skipped")

    logger.step("PIPELINE COMPLETE")
    return 0


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="SimNIBS e-field analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                                                    # full run
  python run.py --skip-preprocessing                              # reuse existing preprocessed files
  python run.py --skip-preprocessing --skip-features              # reuse all_features.csv
  python run.py --skip-preprocessing --skip-features --skip-analysis  # viz only
  python run.py --config my_config.yaml
        """,
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "config.yaml"
    )
    parser.add_argument("--skip-target-generation", action="store_true")
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--skip-viz", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    _args = _parse_args()
    raise SystemExit(
        main(
            config_path=_args.config,
            skip_target_generation=_args.skip_target_generation,
            skip_preprocessing=_args.skip_preprocessing,
            skip_features=_args.skip_features,
            skip_analysis=_args.skip_analysis,
            skip_viz=_args.skip_viz,
        )
    )
