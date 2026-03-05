# Module de Visualisation SimNIBS

Architecture modulaire pour les visualisations du pipeline d'analyse SimNIBS.

## 📁 Structure

```
viz/
├── __init__.py              # API publique du module
├── plots.py                 # Plots atomiques (niveau bas)
├── subject_figures.py       # Figures niveau SUJET
├── group_figures.py         # Figures niveau GROUPE
├── visualisation.py         # Documentation et exemples
└── README.md               # Ce fichier
```

## 🏗️ Architecture en 3 niveaux

### 1. **Plots Atomiques** (`plots.py`)
Fonctions bas niveau qui tracent sur un `axes` matplotlib fourni.

**Caractéristiques:**
- ✅ Prennent un `ax` en paramètre
- ✅ Font UNE seule chose
- ✅ Réutilisables et composables
- ✅ Retournent l'axes modifié

**Exemples:**
```python
from viz import plot_efield_slice, plot_histogram

fig, ax = plt.subplots()
plot_efield_slice(ax, efield_data, title="E-field", vmax=1.0)
```

**Fonctions disponibles:**
- `plot_segmentation_overlay()` - Overlay segmentation sur T1
- `plot_efield_slice()` - Coupe d'e-field avec colormap
- `plot_efield_difference()` - Différence entre deux e-fields
- `plot_roi_overlay()` - ROI sur template
- `plot_histogram()` - Histogramme avec statistiques
- `plot_boxplot_comparison()` - Boxplot comparatif
- `plot_paired_data()` - Données paired (lignes connectées)

### 2. **Figures Niveau SUJET** (`subject_figures.py`)
Visualisations pour **UN SEUL sujet**.

**Classe:** `SubjectVisualizer`

**Méthodes:**
```python
from viz import SubjectVisualizer

viz = SubjectVisualizer(dpi=150)

# 1. Vérification de la segmentation
viz.segmentation_check(
    t1_path=Path("sub-001/T1.nii.gz"),
    segmentation_path=Path("sub-001/segmentation.nii.gz"),
    subject_id="001",
    output_path=Path("outputs/sub-001_seg_check.png")
)

# 2. Comparaison e-fields stim vs opti
viz.efield_comparison(
    stim_efield_path=Path("sub-001/stim_efield.nii.gz"),
    opti_efield_path=Path("sub-001/opti_efield.nii.gz"),
    subject_id="001",
    roi_name="fef",
    output_path=Path("outputs/sub-001_efield_comparison.png")
)

# 3. Effet du préprocessing
viz.preprocessing_effect(
    raw_efield_path=Path("sub-001/raw_efield.nii.gz"),
    preprocessed_efield_path=Path("sub-001/prep_efield.nii.gz"),
    subject_id="001",
    roi_name="fef",
    output_path=Path("outputs/sub-001_preprocessing.png")
)
```

### 3. **Figures Niveau GROUPE** (`group_figures.py`)
Visualisations sur **PLUSIEURS sujets** (analyses de groupe).

**Classe:** `GroupVisualizer`

**Méthodes:**
```python
from viz import GroupVisualizer

viz = GroupVisualizer(dpi=150)

# 1. Définition ROI (commune à tous les sujets)
viz.roi_definition(
    roi_mask_path=Path("templates/fef_mask.nii.gz"),
    roi_name="FEF",
    template_path=Path("templates/MNI152_T1_1mm.nii.gz"),
    output_path=Path("outputs/roi_fef_definition.png")
)

# 2. Comparaison groupe stim vs opti
viz.stim_vs_opti_comparison(
    features_csv=Path("outputs/all_features.csv"),
    roi_name="fef",
    metric="mean",
    output_path=Path("outputs/group_fef_comparison.png")
)

# 3. Comparaison multi-ROI
viz.multi_roi_comparison(
    features_csv=Path("outputs/all_features.csv"),
    roi_names=["fef", "ips_left", "ips_right"],
    metric="mean",
    output_path=Path("outputs/multi_roi_comparison.png")
)
```

## 🎯 Principes de Design

### 1. Séparation des responsabilités
- **Plots atomiques** : Tracent sur un axes, font UNE chose
- **Figures** : Assemblent plusieurs plots, créent la mise en page complète

### 2. Niveau Sujet vs Groupe
- **Sujet** : Données individuelles (segmentation, e-field spécifique)
- **Groupe** : Données agrégées ou communes (ROI, statistiques, comparaisons)

### 3. Composition
Les figures utilisent les plots atomiques, ce qui permet:
- ✅ Réutilisation du code
- ✅ Flexibilité pour créer de nouvelles visualisations
- ✅ Maintenance facilitée
- ✅ Tests unitaires plus simples

## 🔧 Usage Avancé

### Créer une visualisation personnalisée

```python
from viz import plots
import matplotlib.pyplot as plt

# Charger les données
efield_data, _ = plots.load_nifti(Path("efield.nii.gz"))
roi_data, _ = plots.load_nifti(Path("roi.nii.gz"))

# Créer une figure personnalisée
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Utiliser les plots atomiques
slice_efield = plots.extract_slice(efield_data, axis=2, slice_idx=50)
slice_roi = plots.extract_slice(roi_data, axis=2, slice_idx=50)

plots.plot_efield_slice(axes[0], slice_efield, title="E-field")
plots.plot_roi_overlay(axes[1], slice_roi, title="ROI")

plt.savefig("custom_viz.png", dpi=150)
```

### Combiner plusieurs visualisations

```python
from viz import SubjectVisualizer, GroupVisualizer

subject_viz = SubjectVisualizer(dpi=150)
group_viz = GroupVisualizer(dpi=150)

# Générer toutes les visualisations pour un sujet
for subject_id in subjects:
    subject_viz.segmentation_check(...)
    subject_viz.efield_comparison(...)
    subject_viz.preprocessing_effect(...)

# Puis les analyses de groupe
group_viz.roi_definition(...)
group_viz.stim_vs_opti_comparison(...)
group_viz.multi_roi_comparison(...)
```

## 📊 Visualisations Disponibles

### Niveau SUJET
| Méthode | Description | Output |
|---------|-------------|--------|
| `segmentation_check()` | Overlay segmentation sur T1 | 3 coupes axiales |
| `efield_comparison()` | Stim vs Opti vs Diff | 3 panels |
| `preprocessing_effect()` | Histogrammes raw vs preprocessed | 2 histogrammes |

### Niveau GROUPE
| Méthode | Description | Output |
|---------|-------------|--------|
| `roi_definition()` | ROI dans l'espace MNI | 3 vues orthogonales |
| `stim_vs_opti_comparison()` | Comparaison groupe | Boxplot + paired plot |
| `multi_roi_comparison()` | Plusieurs ROIs | Multiple boxplots |

## 🚀 Extension

Pour ajouter une nouvelle visualisation:

1. **Plot atomique** → Ajouter dans `plots.py`
2. **Figure sujet** → Ajouter méthode dans `SubjectVisualizer`
3. **Figure groupe** → Ajouter méthode dans `GroupVisualizer`
4. **Exposer** → Mettre à jour `__init__.py` si nécessaire

## 📝 Conventions

- Toutes les fonctions de plot prennent un `ax` en premier paramètre
- Toutes les méthodes de figure peuvent sauvegarder avec `output_path`
- Les couleurs sont cohérentes: `hot` pour e-field, `RdBu_r` pour différences
- Les titres sont descriptifs et incluent l'info contextuelle
- Les axes ont toujours des labels clairs

## 🔍 Voir aussi

- `visualisation.py` - Documentation détaillée et exemples
- `../analysis.py` - Analyses statistiques
- `../preprocessing.py` - Préprocessing des e-fields
