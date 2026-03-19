#!/usr/bin/env python3
"""
Script de test pour le module de visualisation.
Génère une grille d'images d'e-fields pour chaque sujet.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import logging
import matplotlib.pyplot as plt
import numpy as np

from viz.plots import load_nifti
from viz import SubjectVisualizer, GroupVisualizer
from file_io import find_preprocessed_efield, find_raw_efield

try:
    from nilearn import plotting

    HAS_NILEARN = True
except ImportError:
    HAS_NILEARN = False
    logging.warning("nilearn not available, 3D rendering will be limited")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Charge le fichier de configuration YAML"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def plot_efield_grid(
    config: dict,
    roi: str,
    mode: str,
    output_path: Path,
    max_subjects: int | None = None,
) -> None:
    """
    Crée une grille d'images d'e-fields pour chaque sujet avec rendu 3D.
    Utilise les e-fields bruts de simnibs-config-healthyV2.

    Parameters
    ----------
    config : dict
        Configuration du pipeline
    roi : str
        ROI à visualiser
    mode : str
        Mode (simulation ou optimization)
    output_path : Path
        Chemin de sauvegarde de la figure
    max_subjects : int, optional
        Nombre maximum de sujets à afficher
    """
    logger.info(f"Création de la grille d'e-fields pour ROI={roi}, mode={mode}")

    simnibs_output = Path(config["paths"]["simnibs_output"])
    subjects = config["subjects"][:max_subjects] if max_subjects else config["subjects"]

    # Collecter les e-fields disponibles
    efield_paths = []
    valid_subjects = []

    for subject in subjects:
        efield_path = find_raw_efield(simnibs_output, subject, roi, mode)

        if efield_path and efield_path.exists():
            efield_paths.append(efield_path)
            valid_subjects.append(subject)
            logger.info(f"✓ Trouvé: {subject}")
        else:
            logger.warning(f"✗ Non trouvé: {subject} - {roi} - {mode}")

    if not efield_paths:
        logger.error("Aucun e-field trouvé!")
        return

    logger.info(f"Total de {len(efield_paths)} e-fields trouvés")

    # Calculer le layout de la grille
    n_subjects = len(efield_paths)
    n_cols = min(6, n_subjects)  # Maximum 6 colonnes
    n_rows = int(np.ceil(n_subjects / n_cols))

    # Déterminer le vmax commun pour une échelle cohérente
    logger.info("Chargement des données pour trouver vmax...")
    vmax = 0
    for path in efield_paths:
        data, _ = load_nifti(path)
        vmax = max(vmax, np.nanmax(data))
    logger.info(f"Vmax commun: {vmax:.3f} V/m")

    # Créer la figure
    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))

    # Rendre chaque e-field
    for idx, (efield_path, subject) in enumerate(zip(efield_paths, valid_subjects)):
        ax = plt.subplot(n_rows, n_cols, idx + 1)

        if HAS_NILEARN:
            # Utiliser nilearn pour un rendu 3D propre
            try:
                # Vue axiale (depuis le haut)
                display = plotting.plot_stat_map(
                    str(efield_path),
                    display_mode="z",
                    cut_coords=1,
                    colorbar=False,
                    cmap="hot",
                    vmax=vmax,
                    axes=ax,
                    title=f"Sub {subject}",
                    annotate=False,
                )
            except Exception as e:
                logger.warning(f"Erreur nilearn pour {subject}: {e}")
                # Fallback: afficher juste le texte
                ax.text(
                    0.5,
                    0.5,
                    f"Error\n{subject}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.axis("off")
        else:
            # Fallback sans nilearn: afficher juste le nom
            ax.text(
                0.5,
                0.5,
                f"Subject\n{subject}\n(install nilearn)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")

    # Masquer les axes vides
    for idx in range(len(efield_paths), n_rows * n_cols):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        ax.axis("off")

    # Ajouter une colorbar commune
    if HAS_NILEARN:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        cmap = plt.cm.hot
        norm = Normalize(vmin=0, vmax=vmax)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Colorbar
        fig.subplots_adjust(bottom=0.1)
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("E-field (V/m)", fontsize=12)

    # Titre général
    fig.suptitle(
        f"E-field Grid - {roi.upper()} - {mode.capitalize()}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()

    # Sauvegarder
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"✓ Figure sauvegardée: {output_path}")
    plt.close(fig)


def test_subject_visualizations(config: dict, subject: str, roi: str) -> None:
    """
    Test des visualisations niveau sujet.

    Parameters
    ----------
    config : dict
        Configuration du pipeline
    subject : str
        ID du sujet à visualiser
    roi : str
        ROI à visualiser
    """
    logger.info(f"Test visualisations niveau SUJET pour {subject} - {roi}")

    output_dir = Path(config["paths"]["output_dir"])
    viz_output = output_dir / "visualizations" / "subject_level"
    viz_output.mkdir(parents=True, exist_ok=True)

    subject_viz = SubjectVisualizer(dpi=150)

    # 1. Comparaison e-fields stim vs opti
    stim_path = find_preprocessed_efield(output_dir, subject, roi, "simulation")
    opti_path = find_preprocessed_efield(output_dir, subject, roi, "optimization")

    if stim_path and opti_path:
        logger.info("→ Création comparaison stim vs opti...")
        try:
            subject_viz.efield_comparison(
                stim_efield_path=stim_path,
                opti_efield_path=opti_path,
                subject_id=subject,
                roi_name=roi,
                output_path=viz_output / f"sub-{subject}_{roi}_comparison.png",
            )
            logger.info("  ✓ Comparaison créée")
        except Exception as e:
            logger.error(f"  ✗ Erreur: {e}")
    else:
        logger.warning("  ✗ E-fields manquants")


def test_group_visualizations(config: dict, roi: str) -> None:
    """
    Test des visualisations niveau groupe.

    Parameters
    ----------
    config : dict
        Configuration du pipeline
    roi : str
        ROI à visualiser
    """
    logger.info(f"Test visualisations niveau GROUPE pour {roi}")

    output_dir = Path(config["paths"]["output_dir"])
    viz_output = output_dir / "visualizations" / "group_level"
    viz_output.mkdir(parents=True, exist_ok=True)

    group_viz = GroupVisualizer(dpi=150)

    # 1. ROI definition
    roi_mask_path = Path(config["paths"]["roi_masks"]) / f"{roi}_mask.nii.gz"

    if roi_mask_path.exists():
        logger.info("→ Création définition ROI...")
        try:
            group_viz.roi_definition(
                roi_mask_path=roi_mask_path,
                roi_name=roi.upper(),
                output_path=viz_output / f"roi_{roi}_definition.png",
            )
            logger.info("  ✓ Définition ROI créée")
        except Exception as e:
            logger.error(f"  ✗ Erreur: {e}")
    else:
        logger.warning(f"  ✗ Masque ROI non trouvé: {roi_mask_path}")

    # 2. Comparaison groupe stim vs opti
    features_csv = output_dir / "all_features.csv"

    if features_csv.exists():
        logger.info("→ Création comparaison groupe stim vs opti...")
        try:
            group_viz.stim_vs_opti_comparison(
                features_csv=features_csv,
                roi_name=roi,
                metric="mean",
                output_path=viz_output / f"group_{roi}_stim_vs_opti.png",
            )
            logger.info("  ✓ Comparaison groupe créée")
        except Exception as e:
            logger.error(f"  ✗ Erreur: {e}")
    else:
        logger.warning(f"  ✗ CSV features non trouvé: {features_csv}")


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Test du module de visualisation")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Chemin vers le fichier de configuration",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default="fef",
        choices=["fef", "ips_left", "ips_right"],
        help="ROI à visualiser",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="simulation",
        choices=["simulation", "optimization"],
        help="Mode à visualiser pour la grille",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Nombre maximum de sujets pour la grille",
    )
    parser.add_argument(
        "--test-subject",
        type=str,
        default=None,
        help="Tester les visualisations niveau sujet pour un sujet spécifique",
    )
    parser.add_argument(
        "--test-group",
        action="store_true",
        help="Tester les visualisations niveau groupe",
    )

    args = parser.parse_args()

    # Charger la configuration
    logger.info("=" * 60)
    logger.info("TEST MODULE DE VISUALISATION")
    logger.info("=" * 60)
    logger.info(f"Configuration: {args.config}")

    config = load_config(args.config)
    output_dir = Path(config["paths"]["output_dir"])

    # Créer le répertoire de visualisation
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: Grille d'e-fields
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: GRILLE D'E-FIELDS")
    logger.info("=" * 60)
    plot_efield_grid(
        config=config,
        roi=args.roi,
        mode=args.mode,
        output_path=viz_dir / f"grid_{args.roi}_{args.mode}.png",
        max_subjects=args.max_subjects,
    )

    # Test 2: Visualisations niveau sujet (optionnel)
    if args.test_subject:
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: VISUALISATIONS NIVEAU SUJET")
        logger.info("=" * 60)
        test_subject_visualizations(config, args.test_subject, args.roi)

    # Test 3: Visualisations niveau groupe (optionnel)
    if args.test_group:
        logger.info("\n" + "=" * 60)
        logger.info("TEST 3: VISUALISATIONS NIVEAU GROUPE")
        logger.info("=" * 60)
        test_group_visualizations(config, args.roi)

    logger.info("\n" + "=" * 60)
    logger.info("✓ TESTS TERMINÉS")
    logger.info("=" * 60)
    logger.info(f"Visualisations sauvegardées dans: {viz_dir}")


if __name__ == "__main__":
    main()
