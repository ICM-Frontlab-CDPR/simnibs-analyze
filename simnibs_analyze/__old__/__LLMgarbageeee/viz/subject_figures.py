"""
Figures niveau SUJET - assemblent plusieurs plots pour un sujet donné.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .plots import (
    load_nifti,
    extract_slice,
    plot_segmentation_overlay,
    plot_efield_slice,
    plot_efield_difference,
    plot_histogram,
)


class SubjectVisualizer:
    """
    Génère des figures pour les visualisations niveau sujet.
    """

    def __init__(self, dpi: int = 150):
        """
        Initialise le visualiseur sujet.

        Parameters
        ----------
        dpi : int
            Résolution des figures
        """
        self.dpi = dpi

    def segmentation_check(
        self,
        t1_path: Path,
        segmentation_path: Path,
        subject_id: str,
        slices: Optional[List[int]] = None,
        output_path: Optional[Path] = None,
    ) -> Figure:
        """
        Figure de vérification de la segmentation (NIVEAU SUJET).

        Parameters
        ----------
        t1_path : Path
            Chemin vers le T1
        segmentation_path : Path
            Chemin vers la segmentation
        subject_id : str
            ID du sujet
        slices : list of int, optional
            Indices des coupes axiales
        output_path : Path, optional
            Chemin de sauvegarde

        Returns
        -------
        fig : Figure
            Figure matplotlib
        """
        # Charger les données
        t1_data, _ = load_nifti(t1_path)
        seg_data, _ = load_nifti(segmentation_path)

        # Coupes par défaut
        if slices is None:
            slices = [
                t1_data.shape[2] // 3,
                t1_data.shape[2] // 2,
                2 * t1_data.shape[2] // 3,
            ]

        # Créer la figure
        fig, axes = plt.subplots(1, len(slices), figsize=(6 * len(slices), 5))
        if len(slices) == 1:
            axes = [axes]

        for ax, slice_idx in zip(axes, slices):
            t1_slice = extract_slice(t1_data, axis=2, slice_idx=slice_idx)
            seg_slice = extract_slice(seg_data, axis=2, slice_idx=slice_idx)

            plot_segmentation_overlay(
                ax, t1_slice, seg_slice, title=f"Slice {slice_idx}"
            )

        fig.suptitle(
            f"Segmentation Check - Subject {subject_id}", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

        return fig

    def efield_comparison(
        self,
        stim_efield_path: Path,
        opti_efield_path: Path,
        subject_id: str,
        roi_name: str,
        slice_idx: Optional[int] = None,
        output_path: Optional[Path] = None,
    ) -> Figure:
        """
        Figure de comparaison des e-fields stim vs opti (NIVEAU SUJET).

        Parameters
        ----------
        stim_efield_path : Path
            Chemin vers l'e-field de stimulation
        opti_efield_path : Path
            Chemin vers l'e-field d'optimisation
        subject_id : str
            ID du sujet
        roi_name : str
            Nom de la ROI
        slice_idx : int, optional
            Index de la coupe axiale
        output_path : Path, optional
            Chemin de sauvegarde

        Returns
        -------
        fig : Figure
            Figure matplotlib
        """
        # Charger les e-fields
        stim_data, _ = load_nifti(stim_efield_path)
        opti_data, _ = load_nifti(opti_efield_path)

        # Déterminer le vmax commun
        vmax = max(np.nanmax(stim_data), np.nanmax(opti_data))

        # Extraire les coupes
        stim_slice = extract_slice(stim_data, axis=2, slice_idx=slice_idx)
        opti_slice = extract_slice(opti_data, axis=2, slice_idx=slice_idx)
        diff_slice = opti_slice - stim_slice

        # Créer la figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plots
        plot_efield_slice(axes[0], stim_slice, title="Stimulation", vmax=vmax)
        plot_efield_slice(axes[1], opti_slice, title="Optimization", vmax=vmax)
        plot_efield_difference(axes[2], diff_slice, title="Difference (Opti - Stim)")

        fig.suptitle(
            f"E-field Comparison - {subject_id} - {roi_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

        return fig

    def preprocessing_effect(
        self,
        raw_efield_path: Path,
        preprocessed_efield_path: Path,
        subject_id: str,
        roi_name: str,
        bins: int = 50,
        output_path: Optional[Path] = None,
    ) -> Figure:
        """
        Figure montrant l'effet du preprocessing (NIVEAU SUJET).

        Parameters
        ----------
        raw_efield_path : Path
            Chemin vers l'e-field brut
        preprocessed_efield_path : Path
            Chemin vers l'e-field préprocessé
        subject_id : str
            ID du sujet
        roi_name : str
            Nom de la ROI
        bins : int
            Nombre de bins pour les histogrammes
        output_path : Path, optional
            Chemin de sauvegarde

        Returns
        -------
        fig : Figure
            Figure matplotlib
        """
        # Charger les données
        raw_data, _ = load_nifti(raw_efield_path)
        prep_data, _ = load_nifti(preprocessed_efield_path)

        # Filtrer les valeurs valides
        raw_values = raw_data[np.isfinite(raw_data) & (raw_data != 0)]
        prep_values = prep_data[np.isfinite(prep_data) & (prep_data != 0)]

        # Créer la figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogrammes
        plot_histogram(
            axes[0], raw_values, title="Raw E-field", bins=bins, color="blue"
        )
        plot_histogram(
            axes[1],
            prep_values,
            title="Preprocessed E-field",
            bins=bins,
            color="orange",
        )

        fig.suptitle(
            f"Preprocessing Effect - {subject_id} - {roi_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

        return fig
