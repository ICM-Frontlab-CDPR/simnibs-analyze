#!/usr/bin/env python3
"""
Pipeline principal d'analyse des e-fields SimNIBS.
Orchestre target generation, preprocessing, feature extraction, analysis et visualisation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import pandas as pd

from _0_anatomical_preparer import AnatomicalPreparer
from _1_preprocessing import Preprocessor
from _2_features_extraction import FeatureExtractor
from _3_analysis import Analysis
from _4_viz import Visualizer
from _pipeline_io import (
    find_efield_files,
    find_simulation_dirs,
    get_roi_mask_path,
    load_config,
    save_nifti,
    save_rows,
)
from _logging import get_logger

logger = get_logger(__name__)




def process_subject_condition(
    subject: str,
    condition: str,
    mode: str,
    config: Dict,
    skip_preprocessing: bool = False,
) -> List[Dict]:
    """
    Préprocesse et extrait les features de tous les e-fields pour un sujet/condition/mode.

    Returns
    -------
    List[Dict]
        Lignes de features extraites (une par fichier e-field trouvé).
    """
    results: List[Dict] = []
    simnibs_output = Path(config["paths"]["simnibs_output"])
    subject_dir = simnibs_output / subject

    if not subject_dir.exists():
        logger.warning(f"Répertoire sujet introuvable : {subject_dir}")
        return results

    simulation_dirs = find_simulation_dirs(subject_dir, condition, mode)
    if not simulation_dirs:
        logger.warning(f"Aucune simulation trouvée pour {subject}/{condition}/{mode}")
        return results

    try:
        roi_mask_path = get_roi_mask_path(simnibs_output, condition)
    except FileNotFoundError as e:
        logger.error(str(e))
        return results

    preproc_params = config.get("preprocessing", {})

    for sim_dir in simulation_dirs:
        for efield_path in find_efield_files(sim_dir, mode):
            if mode == "optimization":
                preproc_dir = sim_dir / "simulation_with_optimal_montage" / "mni_volumes"
            else:
                preproc_dir = sim_dir / "mni_volumes"

            base_name = efield_path.stem.replace(".nii", "")
            cleaned_path = preproc_dir / f"{base_name}_roi_cleaned.nii.gz"
            masked_path = preproc_dir / f"{base_name}_roi_masked.nii.gz"

            # ── Preprocessing ───────────────────────────────────────────
            if skip_preprocessing:
                if not cleaned_path.exists():
                    logger.warning(f"Fichier preprocessed introuvable, skip : {cleaned_path}")
                    continue
                logger.info(f"Utilisation fichier existant : {cleaned_path.name}")
            else:
                if cleaned_path.exists() and masked_path.exists():
                    logger.info(f"Déjà preprocessé, skip : {cleaned_path.name}")
                else:
                    try:
                        preproc = Preprocessor(
                            smooth_fwhm=preproc_params.get("smooth_fwhm", 2.0),
                            outlier_method=preproc_params.get("outlier_method", "iqr"),
                            portion=preproc_params.get("portion", None),
                        ).run(efield_path, roi_mask_path)
                        save_nifti(preproc.masked_img, masked_path)
                        save_nifti(preproc.cleaned_img, cleaned_path)
                        logger.info(f"✓ Preprocessing : {cleaned_path.name}")
                    except Exception as e:
                        logger.error(f"✗ Preprocessing échoué ({efield_path.name}) : {e}")
                        continue

            # ── Feature extraction ──────────────────────────────────────
            try:
                row = FeatureExtractor().run(
                    cleaned_path,
                    roi_path=None,
                    subject=subject,
                    condition=f"{condition}_{mode}",
                ).row
                logger.info(f"✓ Features : mean={row.get('mean', 'N/A'):.4f}")
                results.append(row)
            except Exception as e:
                logger.error(f"✗ Feature extraction échouée ({cleaned_path.name}) : {e}")

    return results


def run_analysis(features_csv: Path, config: Dict) -> None:
    """Analyse inter/intra-sujets et scatter plot simulation vs optimization."""
    logger.step("ANALYSE INTER/INTRA-SUJETS")

    results_dir = Path(config["paths"]["results_dir"])
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    ap = config.get("analysis", {})
    metric = ap.get("metric", "mean")
    subject_col = ap.get("subject_col", "subject")
    condition_col = ap.get("condition_col", "condition")

    df = pd.read_csv(features_csv)
    logger.info(f"Chargement : {len(df)} lignes depuis {features_csv}")

    # Inter-sujet
    inter = Analysis(df).inter_subject_summary(metric=metric, condition_col=condition_col)
    inter_csv = analysis_dir / "inter_subject_summary.csv"
    inter.to_csv(inter_csv, index=False)
    logger.info(f"✓ Résumé inter-sujet : {inter_csv}")

    # Intra-sujet (paires simulation / optimization)
    conditions = set(df[condition_col].unique())
    for cond in config["stim_conditions"]:
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
            diff_csv = analysis_dir / f"intra_subject_diff_{cond}.csv"
            diff_df.to_csv(diff_csv, index=False)
            logger.info(f"✓ Diff intra-sujet : {diff_csv}")
        except Exception as e:
            logger.warning(f"Analyse intra-sujet impossible pour {cond} : {e}")

    # Scatter simulation vs optimization
    Visualizer(analysis_dir).plot_simulation_vs_optimization(
        df, metric=metric, subject_col=subject_col, condition_col=condition_col,
    )
    logger.info("✓ Scatter simulation vs optimization créé")
    logger.step("ANALYSE TERMINÉE")


def run_viz(config: Dict) -> None:
    """Collecte les chemins (IO) puis génère toutes les visualisations."""
    logger.step("GÉNÉRATION DES VISUALISATIONS")

    simnibs_output = Path(config["paths"]["simnibs_output"])
    results_dir = Path(config["paths"]["results_dir"])
    subjects: List[str] = config["subjects"]
    conditions: List[str] = config["stim_conditions"]
    modes: List[str] = config["mode"]

    viz = Visualizer(output_dir=results_dir, cmap="hot", threshold_percentile=50.0,
                     bins=50, camera_position="xy")

    # ── Masques ROI ─────────────────────────────────────────────────────
    mni_target_dir = simnibs_output / "mni_target"
    mask_paths = sorted(mni_target_dir.glob("*_mask.nii.gz"))
    if mask_paths:
        mni_tpl_path = config.get("paths", {}).get("mni_template")
        mni_template = nib.load(str(mni_tpl_path)) if mni_tpl_path else None
        mask_imgs = [nib.load(str(p)) for p in mask_paths]
        roi_names = [p.stem.replace("_mask", "") for p in mask_paths]
        viz.visualize_roi_masks(mask_imgs, roi_names, mni_template)
        logger.info("✓ Masques ROI visualisés")

    # ── Figures 3D e-fields ─────────────────────────────────────────────
    # Brain backgrounds par sujet (produits par AnatomicalPreparer.run())
    mni_brain_bg_by_subject: Dict[str, Path] = {}
    subject_brain_bg_by_subject: Dict[str, Path] = {}
    for subject in subjects:
        mni_bg = simnibs_output / subject / "subject_target" / "T1_MNI_brain.nii.gz"
        subj_bg = simnibs_output / subject / "subject_target" / "T1_subject_brain.nii.gz"
        if mni_bg.exists():
            mni_brain_bg_by_subject[subject] = mni_bg
        if subj_bg.exists():
            subject_brain_bg_by_subject[subject] = subj_bg

    for space in ["mni", "subject"]:
        file_info: Dict = {}
        for mode in modes:
            for condition in conditions:
                for subject in subjects:
                    sim_dirs = find_simulation_dirs(simnibs_output / subject, condition, mode)
                    if not sim_dirs:
                        continue
                    efields = find_efield_files(sim_dirs[0], mode, space=space)
                    if not efields:
                        continue
                    file_info.setdefault((condition, mode), []).append((subject, efields[0]))
        if not file_info:
            logger.warning(f"Aucun e-field trouvé pour space={space}, figures skippées")
            continue
        brain_bgs = mni_brain_bg_by_subject if space == "mni" else subject_brain_bg_by_subject
        viz.efields_figures(file_info, t1_brain_by_subject=brain_bgs or None, space=space)
        logger.info(f"✓ Figures 3D e-fields générées ({space.upper()})")    

    # ── Histogrammes preprocessing ───────────────────────────────────────
    preproc_data: Dict = {}
    for subject in subjects:
        for mode in modes:
            for condition in conditions:
                sim_dirs = find_simulation_dirs(simnibs_output / subject, condition, mode)
                if not sim_dirs:
                    continue
                base = (
                    sim_dirs[0] / "simulation_with_optimal_montage" / "mni_volumes"
                    if mode == "optimization"
                    else sim_dirs[0] / "mni_volumes"
                )
                masked = list(base.glob("*_roi_masked.nii.gz"))
                cleaned = list(base.glob("*_roi_cleaned.nii.gz"))
                if not masked or not cleaned:
                    continue
                preproc_data.setdefault(subject, []).append(
                    (condition, mode, masked[0], cleaned[0])
                )
    viz.efields_histograms(preproc_data)
    logger.info("✓ Histogrammes générés")
    logger.step("VISUALISATIONS TERMINÉES")


def main(
    config_path: Path,
    skip_target_generation: bool = False,
    skip_preprocessing: bool = False,
    skip_features: bool = False,
    skip_analysis: bool = False,
    skip_viz: bool = False,
) -> int:
    """Point d'entrée principal du pipeline."""
    logger.step("DÉMARRAGE DU PIPELINE D'ANALYSE E-FIELD")
    logger.info(f"Config : {config_path}")

    config = load_config(config_path)
    logger.info(f"Sujets     : {config['subjects']}")
    logger.info(f"Conditions : {config['stim_conditions']}")
    logger.info(f"Modes      : {config['mode']}")

    results_dir = Path(config["paths"]["results_dir"])
    simnibs_output = Path(config["paths"]["simnibs_output"])

    if results_dir.exists() and any(results_dir.iterdir()):
        logger.warning(f"Le dossier de sortie existe déjà : {results_dir}")
        response = input("Continuer et écraser les fichiers existants ? (o/N) : ")
        if response.lower() not in ("o", "oui", "y", "yes"):
            logger.info("Annulé par l'utilisateur")
            return 0

    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Étape 0 : Setup targets (une fois, indépendant des sujets) ────────
    rois: Dict = config.get("rois", {})
    mni_target_dir = simnibs_output / "mni_target"
    ref = config.get("paths", {}).get("mni_template")
    mni_brain_mask = config.get("paths", {}).get("mni_brain_mask")
    radius_mm = config.get("target_generation", {}).get("radius_mm", 10.0)
    gen = AnatomicalPreparer(
        reference_img_path=Path(ref) if ref else None,
        radius_mm=radius_mm,
        mni_brain_mask_path=Path(mni_brain_mask) if mni_brain_mask else None,
    )

    if not skip_target_generation:
        logger.step("ÉTAPE 0 : GÉNÉRATION DES MASQUES ROI MNI")
        if all((mni_target_dir / f"{roi}_mask.nii.gz").exists() for roi in rois):
            logger.info(f"✓ Masques ROI déjà présents dans {mni_target_dir}")
        else:
            try:
                gen.setup(rois, mni_target_dir)
            except Exception as e:
                logger.error(f"✗ Target generation échouée : {e}")
                return 1
    else:
        logger.info("Génération des masques ROI skippée")

    # ── Étapes 1+2 : Preprocessing + Feature extraction ─────────────────
    analysis_dir = results_dir / "analysis"
    features_csv = analysis_dir / "all_features.csv"

    if skip_features:
        if not features_csv.exists():
            logger.error(f"all_features.csv introuvable : {features_csv}")
            return 1
        logger.info(f"Feature extraction skippée — utilisation de {features_csv}")
    else:
        all_features: List[Dict] = []
        stats = {"total": 0, "success": 0, "failed": 0}

        for subject in config["subjects"]:
            logger.step(f"SUJET : {subject}")
            m2m_dir = simnibs_output / subject / f"m2m_{subject}"
            if not skip_target_generation and m2m_dir.exists():
                subject_target_dir = simnibs_output / subject / "subject_target"
                gen.run(m2m_dir, subject_target_dir)
            for condition in config["stim_conditions"]:
                for mode in config["mode"]:
                    logger.info(f"--- {subject} / {condition} / {mode} ---")
                    rows = process_subject_condition(
                        subject, condition, mode, config,
                        skip_preprocessing=skip_preprocessing,
                    )
                    stats["total"] += 1
                    if rows:
                        stats["success"] += 1
                        all_features.extend(rows)
                    else:
                        stats["failed"] += 1

        if not all_features:
            logger.warning("Aucune feature extraite !")
            return 1

        analysis_dir.mkdir(parents=True, exist_ok=True)
        save_rows(all_features, features_csv)
        logger.info(f"✓ {len(all_features)} features sauvegardées → {features_csv}")
        logger.info(f"  Succès : {stats['success']}/{stats['total']}, échecs : {stats['failed']}")

    # ── Étape 3 : Analyse ────────────────────────────────────────────────
    if not skip_analysis:
        run_analysis(features_csv, config)
    else:
        logger.info("Analyse skippée")

    # ── Étape 4 : Visualisations ─────────────────────────────────────────
    if not skip_viz:
        run_viz(config)
    else:
        logger.info("Visualisations skippées")

    logger.step("PIPELINE TERMINÉ")
    return 0


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Pipeline d'analyse des e-fields SimNIBS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python run.py                                                    # run complet
  python run.py --skip-preprocessing                              # réutilise les fichiers preprocessés
  python run.py --skip-preprocessing --skip-features              # réutilise all_features.csv
  python run.py --skip-preprocessing --skip-features --skip-analysis  # viz seulement
  python run.py --config mon_config.yaml
        """,
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--skip-target-generation", action="store_true")
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--skip-viz", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    _args = _parse_args()
    raise SystemExit(main(
        config_path=_args.config,
        skip_target_generation=_args.skip_target_generation,
        skip_preprocessing=_args.skip_preprocessing,
        skip_features=_args.skip_features,
        skip_analysis=_args.skip_analysis,
        skip_viz=_args.skip_viz,
    ))
