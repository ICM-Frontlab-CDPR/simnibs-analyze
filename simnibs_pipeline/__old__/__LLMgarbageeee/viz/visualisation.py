"""
Guide d'utilisation du module de visualisation.

ARCHITECTURE
============

Le module est organisé en 3 niveaux :

1. plots.py - PLOTS ATOMIQUES (niveau bas)
   - Fonctions qui tracent sur un axes matplotlib fourni
   - Réutilisables et composables
   - Exemples : plot_efield_slice(), plot_histogram(), etc.

2. subject_figures.py - FIGURES NIVEAU SUJET
   - Visualisations spécifiques à UN SEUL sujet
   - Créent des figures complètes avec plusieurs subplots
   - SubjectVisualizer.segmentation_check()
   - SubjectVisualizer.efield_comparison()
   - SubjectVisualizer.preprocessing_effect()

3. group_figures.py - FIGURES NIVEAU GROUPE
   - Visualisations sur PLUSIEURS sujets (analyses de groupe)
   - GroupVisualizer.roi_definition() - ROI commune à tous
   - GroupVisualizer.stim_vs_opti_comparison() - comparaison groupe
   - GroupVisualizer.multi_roi_comparison() - comparaison multi-ROI

USAGE
=====

# Niveau SUJET
from viz import SubjectVisualizer

subject_viz = SubjectVisualizer(dpi=150)

# Check segmentation d'un sujet
subject_viz.segmentation_check(
    t1_path=Path("sub-001/anat/T1.nii.gz"),
    segmentation_path=Path("sub-001/simnibs/m2m_001/final_tissues.nii.gz"),
    subject_id="001",
    output_path=Path("outputs/sub-001_segmentation.png")
)

# Comparaison e-fields d'un sujet
subject_viz.efield_comparison(
    stim_efield_path=Path("sub-001/stim_efield.nii.gz"),
    opti_efield_path=Path("sub-001/opti_efield.nii.gz"),
    subject_id="001",
    roi_name="fef",
    output_path=Path("outputs/sub-001_efield_comparison.png")
)

# Niveau GROUPE
from viz import GroupVisualizer

group_viz = GroupVisualizer(dpi=150)

# Visualiser la ROI (même pour tous les sujets)
group_viz.roi_definition(
    roi_mask_path=Path("templates/fef_mask.nii.gz"),
    roi_name="FEF",
    template_path=Path("templates/MNI152_T1_1mm.nii.gz"),
    output_path=Path("outputs/roi_fef_definition.png")
)

# Comparaison groupe stim vs opti
group_viz.stim_vs_opti_comparison(
    features_csv=Path("outputs/all_features.csv"),
    roi_name="fef",
    metric="mean",
    output_path=Path("outputs/group_fef_comparison.png")
)

# Plots ATOMIQUES (pour usage avancé)
from viz import plot_efield_slice, plot_histogram
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plot_efield_slice(ax, efield_data, title="Custom E-field")
plt.savefig("custom_plot.png")

PRINCIPES
=========

1. Séparation des responsabilités
   - Plots atomiques : font UNE chose, sur un axes donné
   - Figures : assemblent plusieurs plots, créent la figure complète

2. Niveau sujet vs groupe
   - Sujet : données individuelles (segmentation, e-field d'un sujet)
   - Groupe : données agrégées ou communes à tous (ROI, statistiques)

3. Composition
   - Les figures utilisent les plots atomiques
   - Les plots atomiques sont réutilisables
   - Facilite la création de nouvelles visualisations personnalisées
"""

# Import des visualiseurs pour accès rapide
from .subject_figures import SubjectVisualizer
from .group_figures import GroupVisualizer

__all__ = ['SubjectVisualizer', 'GroupVisualizer']
