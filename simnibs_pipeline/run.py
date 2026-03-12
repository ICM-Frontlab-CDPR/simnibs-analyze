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
    SPACE_MNI,
    SPACE_NATIVE,
    ROI_METHOD_SPHERE,
    ROI_METHOD_ATLAS,
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
    load_config,
    method_tag,
    normalize_roi_method,
    normalize_space,
    space_tag,
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
    space: str = SPACE_MNI,
) -> List[Dict]:
    """
    Préprocesse et extrait les features de tous les e-fields pour un sujet/condition/mode.

    Parameters
    ----------
    space : str
        ``'mni'`` (défaut) ou ``'native'`` — espace de travail pour les efields et ROI masks.

    Returns
    -------
    List[Dict]
        Lignes de features extraites (une par fichier e-field trouvé).
    """
    results: List[Dict] = []
    simnibs_output = Path(config["paths"]["simnibs_output"])
    subject_paths = get_subject_paths(simnibs_output, subject)
    subject_dir = subject_paths["subject_dir"]

    if not subject_dir.exists():
        logger.warning(f"Répertoire sujet introuvable : {subject_dir}")
        return results

    simulation_dirs = find_simulation_dirs(subject_dir, condition, mode)
    if not simulation_dirs:
        logger.warning(f"Aucune simulation trouvée pour {subject}/{condition}/{mode}")
        return results

    tg_params = config.get("target_generation", {})
    roi_method = normalize_roi_method(tg_params.get("roi_method", ROI_METHOD_SPHERE))
    atlas_name = tg_params.get("atlas_name") if roi_method == ROI_METHOD_ATLAS else None

    try:
        roi_mask_path = get_roi_mask_path(
            simnibs_output, condition, space=space, subject=subject,
            method=roi_method, atlas_name=atlas_name,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return results

    preproc_params = config.get("preprocessing", {})

    for sim_dir in simulation_dirs:
        for efield_path in find_efield_files(sim_dir, mode, space=space):
            preproc_dir = get_preproc_dir(sim_dir, mode, space=space)
            base_name = efield_path.stem.replace(".nii", "")
            paths = get_preproc_paths(preproc_dir, base_name)

            preproc_kwargs = dict(
                smooth_fwhm=preproc_params.get("smooth_fwhm", 2.0),
                outlier_method=preproc_params.get("outlier_method", "iqr"),
                portion=preproc_params.get("portion", None),
            )

            # ── Preprocessing INTRA-ROI ──────────────────────────────────
            if skip_preprocessing:
                if not paths["intra_cleaned"].exists():
                    logger.warning(f"Fichier preprocessed introuvable, skip : {paths['intra_cleaned']}")
                    continue
                logger.info(f"Utilisation fichier existant : {paths['intra_cleaned'].name}")
            else:
                if paths["intra_cleaned"].exists() and paths["intra_masked"].exists():
                    logger.info(f"Déjà preprocessé, skip : {paths['intra_cleaned'].name}")
                else:
                    try:
                        preproc = Preprocessor(**preproc_kwargs).run(efield_path, roi_mask_path)
                        save_nifti(preproc.masked_img, paths["intra_masked"])
                        save_nifti(preproc.cleaned_img, paths["intra_cleaned"])
                        logger.info(f"✓ Preprocessing intra-ROI : {paths['intra_cleaned'].name}")
                    except Exception as e:
                        logger.error(f"✗ Preprocessing intra-ROI échoué ({efield_path.name}) : {e}")
                        continue

            # ── Preprocessing EXTRA-ROI ──────────────────────────────────
            if skip_preprocessing:
                if not paths["extra_cleaned"].exists():
                    logger.warning(f"Fichier extra preprocessed introuvable, skip : {paths['extra_cleaned']}")
                    continue
            else:
                if paths["extra_cleaned"].exists() and paths["extra_masked"].exists():
                    logger.info(f"Déjà preprocessé, skip : {paths['extra_cleaned'].name}")
                else:
                    try:
                        extra_mask = Preprocessor.build_extra_mask(roi_mask_path)
                        extra_masked_img = Preprocessor(**preproc_kwargs).run(efield_path, extra_mask).masked_img
                        save_nifti(extra_masked_img, paths["extra_masked"])
                        save_nifti(extra_masked_img, paths["extra_cleaned"])  # cleaned = masked
                        logger.info(f"✓ Preprocessing extra-ROI : {paths['extra_masked'].name}")
                    except Exception as e:
                        logger.error(f"✗ Preprocessing extra-ROI échoué ({efield_path.name}) : {e}")
                        continue

            # ── Feature extraction ───────────────────────────────────────
            try:
                row_intra = FeatureExtractor().run(
                    paths["intra_cleaned"],
                    roi_path=None,
                    subject=subject,
                    condition=f"{condition}_{mode}",
                ).row
                row_extra = FeatureExtractor().run(
                    paths["extra_cleaned"],
                    roi_path=None,
                    subject=None,
                    condition=None,
                ).row

                # Fusion : colonnes intra sans préfixe, extra avec préfixe extra_
                row = {**row_intra}
                for k in ["mean", "median", "std", "min", "max", "n_voxels"]:
                    if k in row_extra:
                        row[f"extra_{k}"] = row_extra[k]
                row["space"] = space

                # Ratio calculé depuis les valeurs nettoyées
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
                logger.error(f"✗ Feature extraction échouée ({subject}/{condition}/{mode}) : {e}")

    return results


def run_analysis(features_csv: Path, config: Dict, space: str) -> None:
    """Analyse inter/intra-sujets et scatter plot simulation vs optimization."""
    logger.step("ANALYSE INTER/INTRA-SUJETS")

    results_dir = Path(config["paths"]["results_dir"])
    analysis_dir = get_analysis_dir(results_dir, space)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    ap = config.get("analysis", {})
    metric = ap.get("metric", "mean")
    subject_col = ap.get("subject_col", "subject")
    condition_col = ap.get("condition_col", "condition")

    df = pd.read_csv(features_csv)
    logger.info(f"Chargement : {len(df)} lignes depuis {features_csv}")

    # Inter-sujet
    inter = Analysis(df).inter_subject_summary(metric=metric, condition_col=condition_col)
    inter_csv = get_inter_subject_summary_csv_path(results_dir, space)
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
            diff_csv = get_intra_subject_diff_csv_path(results_dir, space, cond)
            diff_df.to_csv(diff_csv, index=False)
            logger.info(f"✓ Diff intra-sujet : {diff_csv}")
        except Exception as e:
            logger.warning(f"Analyse intra-sujet impossible pour {cond} : {e}")

    # Clustering
    cl_params = ap.get("clustering", {})
    cl_method = cl_params.get("method", "mean")
    cl_threshold = float(cl_params.get("specificity_threshold", 1.5))
    cl_intensity_col = cl_params.get("intensity_col", "mean")
    ratio_col = f"efield_ratio_{cl_method}"
    if ratio_col in df.columns:
        try:
            clustered_df = Analysis(df).assign_clusters(
                method=cl_method,
                specificity_threshold=cl_threshold,
                intensity_col=cl_intensity_col,
            )
            # clusters.csv conserve toutes les colonnes originales (efield_path, subject,
            # condition, stats…) + cluster — le lien avec les e-fields est donc direct.
            clusters_csv = get_clusters_csv_path(results_dir, space)
            clustered_df.to_csv(clusters_csv, index=False)
            logger.info(f"✓ Clusters sauvegardés : {clusters_csv}")
            dist = clustered_df["cluster"].value_counts().to_dict()
            logger.info(f"  Distribution : {dist}")
        except Exception as e:
            logger.warning(f"Clustering échoué : {e}")
    else:
        logger.warning(
            f"Colonne '{ratio_col}' absente de {features_csv.name} — clustering ignoré. "
            "Assurez-vous que compute_efield_ratio est appelé lors de l'extraction."
        )

    # Scatter simulation vs optimization
    Visualizer(analysis_dir).plot_simulation_vs_optimization(
        df, metric=metric, subject_col=subject_col, condition_col=condition_col,
        output_tag=space,
    )
    logger.info("✓ Scatter simulation vs optimization créé")
    logger.step("ANALYSE TERMINÉE")


def run_viz(config: Dict, space: str) -> None:
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
    # Glob accepts masks from any method (sphere, atlas, …)
    mask_paths = sorted(mni_target_dir.glob("*_method-*_mask_space-mni.nii.gz"))
    if space == SPACE_MNI and mask_paths:
        mni_tpl_path = config.get("paths", {}).get("mni_template")
        mni_template = nib.load(str(mni_tpl_path)) if mni_tpl_path else None
        mask_imgs = [nib.load(str(p)) for p in mask_paths]
        roi_names = [
            p.name[: p.name.rfind("_method-")].replace("_mask_space-mni.nii.gz", "")
            for p in mask_paths
        ]
        viz.visualize_roi_masks(mask_imgs, roi_names, mni_template)
        logger.info("✓ Masques ROI visualisés")

    # ── Figures 3D e-fields ─────────────────────────────────────────────
    # Brain backgrounds par sujet (produits par AnatomicalPreparer.run())
    mni_brain_bg_by_subject: Dict[str, Path] = {}
    subject_brain_bg_by_subject: Dict[str, Path] = {}
    for subject in subjects:
        subject_paths = get_subject_paths(simnibs_output, subject)
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
                subject_paths = get_subject_paths(simnibs_output, subject)
                sim_dirs = find_simulation_dirs(subject_paths["subject_dir"], condition, mode)
                if not sim_dirs:
                    continue
                efields = find_efield_files(sim_dirs[0], mode, space=space)
                if not efields:
                    continue
                file_info.setdefault((condition, mode), []).append((subject, efields[0]))
    if file_info:
        brain_bgs = mni_brain_bg_by_subject if space == SPACE_MNI else subject_brain_bg_by_subject
        viz.efields_figures(file_info, t1_brain_by_subject=brain_bgs or None, space=space)
        logger.info(f"✓ Figures 3D e-fields générées ({space.upper()})")
    else:
        logger.warning(f"Aucun e-field trouvé pour space={space}, figures skippées")

    # ── Histogrammes preprocessing ───────────────────────────────────────
    intra_data: Dict = {}
    extra_data: Dict = {}
    for subject in subjects:
        for mode in modes:
            for condition in conditions:
                subject_paths = get_subject_paths(simnibs_output, subject)
                sim_dirs = find_simulation_dirs(subject_paths["subject_dir"], condition, mode)
                if not sim_dirs:
                    continue
                preproc_dir = get_preproc_dir(sim_dirs[0], mode, space=space)
                for efield_path in find_efield_files(sim_dirs[0], mode, space=space):
                    base_name = efield_path.stem.replace(".nii", "")
                    p = get_preproc_paths(preproc_dir, base_name)
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
        logger.info("✓ Histogrammes intra-ROI générés")
    if extra_data:
        viz.efields_histograms(extra_data, region="extra", space=space)
        logger.info("✓ Histogrammes extra-ROI générés")
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
    try:
        space = normalize_space(config.get("space", SPACE_MNI))
    except ValueError as e:
        logger.error(str(e))
        return 1

    logger.info(f"Sujets     : {config['subjects']}")
    logger.info(f"Conditions : {config['stim_conditions']}")
    logger.info(f"Modes      : {config['mode']}")
    logger.info(f"Espace     : {space}")

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
    roi_method = normalize_roi_method(
        config.get("target_generation", {}).get("roi_method", ROI_METHOD_SPHERE)
    )
    atlas_name = (
        config.get("target_generation", {}).get("atlas_name")
        if roi_method == ROI_METHOD_ATLAS
        else None
    )
    m_tag = method_tag(roi_method, atlas_name)
    gen = AnatomicalPreparer(
        reference_img_path=Path(ref) if ref else None,
        radius_mm=radius_mm,
        mni_brain_mask_path=Path(mni_brain_mask) if mni_brain_mask else None,
    )

    if not skip_target_generation:
        logger.step("ÉTAPE 0 : GÉNÉRATION DES MASQUES ROI MNI")
        if all(
            (mni_target_dir / f"{roi}_{m_tag}_mask_space-mni.nii.gz").exists()
            for roi in rois
        ):
            logger.info(f"✓ Masques ROI déjà présents dans {mni_target_dir}")
        else:
            try:
                if roi_method == ROI_METHOD_ATLAS:
                    gen.setup_from_atlas(atlas_name, rois, mni_target_dir)
                else:
                    gen.setup(rois, mni_target_dir)
            except Exception as e:
                logger.error(f"✗ Target generation échouée : {e}")
                return 1
    else:
        logger.info("Génération des masques ROI skippée")

    # ── Étapes 1+2 : Preprocessing + Feature extraction ─────────────────
    analysis_dir = get_analysis_dir(results_dir, space)
    features_csv = get_features_csv_path(results_dir, space)

    if skip_features:
        if not features_csv.exists():
            logger.error(f"all_features_{space_tag(space)}.csv introuvable : {features_csv}")
            return 1
        logger.info(f"Feature extraction skippée — utilisation de {features_csv}")
    else:
        all_features: List[Dict] = []
        stats = {"total": 0, "success": 0, "failed": 0}
        
        logger.info(f"Computing in {space.upper()} space")

        for subject in config["subjects"]:
            logger.step(f"SUJET : {subject}")
            subject_paths = get_subject_paths(simnibs_output, subject)
            m2m_dir = subject_paths["m2m_dir"]
            subject_target_dir = subject_paths["subject_target_dir"]

            if space == SPACE_NATIVE and skip_target_generation:
                missing = [
                    roi for roi in rois
                    if not (subject_target_dir / f"{roi}_{m_tag}_mask_space-native.nii.gz").exists()
                ]
                if missing:
                    logger.warning(
                        f"Masques ROI native-space manquants pour {subject} ({missing}) avec --skip-target-generation. "
                        "Sujet ignoré pour éviter des outputs ambigus."
                    )
                    continue
            
            if not skip_target_generation and m2m_dir.exists():
                # Always generate T1 skull-stripped in both spaces
                gen.run(m2m_dir, subject_target_dir)
                
                # If working in native space, generate native-space ROI masks
                if space == SPACE_NATIVE:
                    logger.info("Generating native-space ROI masks...")
                    try:
                        gen.create_subject_roi_from_mni(m2m_dir, subject_target_dir)
                        logger.info(f"✓ Native-space ROI masks generated for {subject}")
                    except Exception as e:
                        logger.warning(f"Native-space ROI generation failed: {e}")
                        logger.warning(f"Subject {subject} ignoré pour éviter un mélange d'espaces")
                        continue
            elif space == SPACE_NATIVE and not m2m_dir.exists():
                logger.warning(
                    f"m2m introuvable pour {subject}: {m2m_dir}. Sujet ignoré en espace native."
                )
                continue
            
            for condition in config["stim_conditions"]:
                for mode in config["mode"]:
                    logger.info(f"--- {subject} / {condition} / {mode} ---")
                    rows = process_subject_condition(
                        subject, condition, mode, config,
                        skip_preprocessing=skip_preprocessing,
                        space=space,
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
        run_analysis(features_csv, config, space=space)
    else:
        logger.info("Analyse skippée")

    # ── Étape 4 : Visualisations ─────────────────────────────────────────
    if not skip_viz:
        run_viz(config, space=space)
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
