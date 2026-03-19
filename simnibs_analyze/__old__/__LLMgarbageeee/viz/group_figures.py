"""
Figures niveau GROUPE - analyses sur plusieurs sujets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .plots import (
    load_nifti,
    extract_slice,
    plot_roi_overlay,
    plot_boxplot_comparison,
    plot_paired_data,
)


class GroupVisualizer:
    """
    Génère des figures pour les visualisations niveau groupe.
    """

    def __init__(self, dpi: int = 150):
        """
        Initialise le visualiseur groupe.

        Parameters
        ----------
        dpi : int
            Résolution des figures
        """
        self.dpi = dpi

    def roi_definition(
        self,
        roi_mask_path: Path,
        roi_name: str,
        template_path: Optional[Path] = None,
        slices: Optional[list] = None,
        output_path: Optional[Path] = None,
    ) -> Figure:
        """
        Figure de définition d'une ROI dans l'espace MNI (NIVEAU GROUPE).
        Cette visualisation est au niveau groupe car la ROI est la même pour tous.

        Parameters
        ----------
        roi_mask_path : Path
            Chemin vers le masque ROI
        roi_name : str
            Nom de la ROI
        template_path : Path, optional
            Chemin vers le template MNI
        slices : list of int, optional
            Indices des coupes
        output_path : Path, optional
            Chemin de sauvegarde

        Returns
        -------
        fig : Figure
            Figure matplotlib
        """
        # Charger la ROI
        roi_data, _ = load_nifti(roi_mask_path)

        # Charger le template si fourni
        template_data = None
        if template_path:
            template_data, _ = load_nifti(template_path)

        # Coupes par défaut (sagittal, coronal, axial)
        if slices is None:
            slices = [
                roi_data.shape[0] // 2,
                roi_data.shape[1] // 2,
                roi_data.shape[2] // 2,
            ]

        # Créer la figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes_names = ["Sagittal", "Coronal", "Axial"]

        for ax, axis, slice_idx, name in zip(axes, [0, 1, 2], slices, axes_names):
            roi_slice = extract_slice(roi_data, axis=axis, slice_idx=slice_idx)

            background_slice = None
            if template_data is not None:
                background_slice = extract_slice(
                    template_data, axis=axis, slice_idx=slice_idx
                )

            plot_roi_overlay(
                ax,
                roi_slice,
                background_slice=background_slice,
                title=f"{name} - Slice {slice_idx}",
            )

        fig.suptitle(
            f"ROI Definition - {roi_name} (MNI Space)", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

        return fig

    def stim_vs_opti_comparison(
        self,
        features_csv: Path,
        roi_name: str,
        metric: str = "mean",
        output_path: Optional[Path] = None,
    ) -> Figure:
        """
        Figure de comparaison stimulation vs optimisation (NIVEAU GROUPE).

        Parameters
        ----------
        features_csv : Path
            Chemin vers le CSV de features
        roi_name : str
            Nom de la ROI
        metric : str
            Métrique à afficher (mean, median, etc.)
        output_path : Path, optional
            Chemin de sauvegarde

        Returns
        -------
        fig : Figure
            Figure matplotlib
        """
        # Charger les données
        df = pd.read_csv(features_csv)

        # Filtrer pour la ROI spécifique
        df_roi = df[df["condition"].str.contains(roi_name, na=False)].copy()

        # Extraire type (simulation/optimization)
        df_roi["type"] = df_roi["condition"].apply(
            lambda x: "optimization" if "optimization" in x else "simulation"
        )

        # Préparer les données
        stim_data = df_roi[df_roi["type"] == "simulation"][metric].values
        opti_data = df_roi[df_roi["type"] == "optimization"][metric].values

        # Créer la figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Box plot comparatif
        plot_boxplot_comparison(
            axes[0],
            {"Stimulation": stim_data, "Optimization": opti_data},
            title="Group Comparison",
            ylabel=f"{metric.capitalize()} E-field (V/m)",
        )

        # Paired plot
        # Pivot pour avoir simulation et optimization côte à côte
        pivot = df_roi.pivot_table(index="subject", columns="type", values=metric)

        # Filtrer les sujets qui ont les deux
        pivot = pivot.dropna()

        if (
            not pivot.empty
            and "simulation" in pivot.columns
            and "optimization" in pivot.columns
        ):
            plot_paired_data(
                axes[1],
                pivot["simulation"].values,
                pivot["optimization"].values,
                "Stimulation",
                "Optimization",
                title="Paired Subject Data",
            )

        fig.suptitle(
            f"Group Level Analysis - {roi_name}", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

        return fig

    def multi_roi_comparison(
        self,
        features_csv: Path,
        roi_names: list,
        metric: str = "mean",
        output_path: Optional[Path] = None,
    ) -> Figure:
        """
        Figure de comparaison entre plusieurs ROIs (NIVEAU GROUPE).

        Parameters
        ----------
        features_csv : Path
            Chemin vers le CSV de features
        roi_names : list
            Liste des noms de ROI à comparer
        metric : str
            Métrique à afficher
        output_path : Path, optional
            Chemin de sauvegarde

        Returns
        -------
        fig : Figure
            Figure matplotlib
        """
        # Charger les données
        df = pd.read_csv(features_csv)

        # Créer la figure
        n_rois = len(roi_names)
        fig, axes = plt.subplots(1, n_rois, figsize=(6 * n_rois, 5), sharey=True)
        if n_rois == 1:
            axes = [axes]

        for ax, roi_name in zip(axes, roi_names):
            # Filtrer pour la ROI
            df_roi = df[df["condition"].str.contains(roi_name, na=False)].copy()

            if df_roi.empty:
                continue

            # Extraire type
            df_roi["type"] = df_roi["condition"].apply(
                lambda x: "optimization" if "optimization" in x else "simulation"
            )

            # Préparer les données
            stim_data = df_roi[df_roi["type"] == "simulation"][metric].values
            opti_data = df_roi[df_roi["type"] == "optimization"][metric].values

            # Box plot
            plot_boxplot_comparison(
                ax,
                {"Stim": stim_data, "Opti": opti_data},
                title=roi_name,
                ylabel=(
                    f"{metric.capitalize()} E-field (V/m)"
                    if roi_name == roi_names[0]
                    else ""
                ),
            )

        fig.suptitle("Multi-ROI Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

        return fig
