"""
Module de visualisation pour le pipeline d'analyse SimNIBS.

Architecture:
- plots.py: Fonctions atomiques de plotting (niveau bas)
- subject_figures.py: Figures niveau sujet (un seul sujet)
- group_figures.py: Figures niveau groupe (plusieurs sujets)
"""

from .plots import (
    load_nifti,
    extract_slice,
    plot_segmentation_overlay,
    plot_efield_slice,
    plot_efield_difference,
    plot_roi_overlay,
    plot_histogram,
    plot_boxplot_comparison,
    plot_paired_data,
)

from .subject_figures import SubjectVisualizer
from .group_figures import GroupVisualizer


__all__ = [
    # Utilitaires
    'load_nifti',
    'extract_slice',
    # Plots atomiques
    'plot_segmentation_overlay',
    'plot_efield_slice',
    'plot_efield_difference',
    'plot_roi_overlay',
    'plot_histogram',
    'plot_boxplot_comparison',
    'plot_paired_data',
    # Visualiseurs
    'SubjectVisualizer',
    'GroupVisualizer',
]
