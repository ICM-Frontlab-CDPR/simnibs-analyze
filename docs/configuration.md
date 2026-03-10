# TODO : generer depuis le module ""_io_pipeline"

# Référence du fichier de configuration

Le pipeline est piloté par un fichier `config.yaml`. Ce fichier est passé à `run.py` via l'argument `--config` (par défaut : `simnibs_pipeline/config.yaml`).

```bash
python run.py --config mon_config.yaml
```

---

## Structure complète annotée

```yaml
# Identifiants des sujets à traiter
subjects: ["0001", "0002", "0004"]

# Noms des conditions de stimulation (correspondent aux dossiers SimNIBS)
stim_conditions: [fef, ips_left, ips_right]

# Modes à traiter
mode: [simulation, optimization]

# Coordonnées MNI [x, y, z] en mm des ROI cibles
rois:
  fef: [28, -8, 54]
  ips_left: [-25, -60, 52]
  ips_right: [25, -60, 52]

# Paramètres de génération des masques sphériques ROI
target_generation:
  radius_mm: 10.0        # Rayon de la sphère ROI en mm

# Chemins absolus vers les données et fichiers de référence
paths:
  simnibs_output: /chemin/vers/derivatives/simnibs   # Dossiers sujets SimNIBS
  results_dir: /chemin/vers/results                  # Dossier de sortie
  mni_template: /chemin/vers/MNI152_T1_1mm.nii.gz
  mni_brain_mask: /chemin/vers/MNI152_T1_1mm_brain_mask.nii.gz

# Paramètres de préprocessing des e-fields
preprocessing:
  smooth_fwhm: 2.0          # FWHM du filtre gaussien (mm) ; 0 pour désactiver
  outlier_method: iqr       # Méthode de suppression des outliers : "iqr" ou null
  portion: null             # Fraction haute à conserver (ex: 0.9) ; null = tout garder

# Métriques calculées lors de l'extraction de features
feature_extraction:
  metrics: [mean, median, std, min, max]

# Paramètres d'analyse statistique
analysis:
  metric: mean              # Métrique principale pour les résumés
  subject_col: subject      # Nom de la colonne sujet dans all_features.csv
  condition_col: condition  # Nom de la colonne condition dans all_features.csv
  clustering:
    method: mean                # Colonne efield_ratio_<method> utilisée pour le clustering
    specificity_threshold: 1.5  # Seuil ratio intra/extra pour "focalisé" vs "diffus"
    intensity_col: mean         # Colonne numérique pour classer intensité haute/basse
```

---

## Description des sections

### `subjects`

Liste des identifiants sujets. Chaque sujet doit avoir un dossier correspondant sous `paths.simnibs_output/`.

### `stim_conditions`

Noms des conditions de stimulation. Ils doivent correspondre aux noms de dossiers générés par SimNIBS (ex : `fef`, `ips_left`).

### `mode`

- `simulation` : e-field simulé avec les paramètres nominaux
- `optimization` : e-field issu de l'optimisation des électrodes

### `rois`

Coordonnées MNI des centres des ROI. Le pipeline génère automatiquement un masque sphérique (`radius_mm`) autour de chaque point et le transforme en espace sujet via ANTsPy.

### `target_generation.radius_mm`

Rayon de la sphère ROI en mm. Une valeur de 10 mm est recommandée pour les cibles corticales typiques.

### `paths`

| Clé               | Description                                                 |
| ------------------ | ----------------------------------------------------------- |
| `simnibs_output` | Racine des dossiers sujets produits par SimNIBS             |
| `results_dir`    | Dossier où seront écrits CSVs, figures et analyses        |
| `mni_template`   | Image de référence MNI (pour la génération des masques) |
| `mni_brain_mask` | Masque cerveau MNI (utilisé pour exclure le fond)          |

### `preprocessing`

| Paramètre         | Valeurs                | Description                                                                             |
| ------------------ | ---------------------- | --------------------------------------------------------------------------------------- |
| `smooth_fwhm`    | float ≥ 0             | Lissage gaussien avant masquage ;`0` = pas de lissage                                 |
| `outlier_method` | `"iqr"` ou `null`  | `"iqr"` supprime les voxels en dehors de [Q1−1.5×IQR, Q3+1.5×IQR]                  |
| `portion`        | float 0–1 ou `null` | Conserve uniquement les `portion` % de voxels les plus forts ; `null` = tout garder |

### `feature_extraction.metrics`

Métriques calculées par `FeatureExtractor` sur les voxels de la ROI nettoyée. Valeurs disponibles : `mean`, `median`, `std`, `min`, `max`. Le résultat est sauvegardé dans `results_dir/analysis/all_features.csv`.

### `analysis`

- **`metric`** : métrique utilisée pour les résumés inter/intra-sujets (doit être dans `feature_extraction.metrics`)
- **`clustering.specificity_threshold`** : si `efield_ratio_mean` > seuil → cluster "focalisé", sinon "diffus"
- **`clustering.intensity_col`** : colonne pour séparer intensité forte/faible (médiane de la distribution)

---

## Utilisation en ligne de commande

```bash
# Run complet
python run.py --config config.yaml

# Réutiliser le preprocessing existant
python run.py --skip-preprocessing

# Réutiliser all_features.csv (skip preprocessing + features)
python run.py --skip-preprocessing --skip-features

# Visualisations uniquement
python run.py --skip-preprocessing --skip-features --skip-analysis

# Regénérer seulement l'analyse et les figures
python run.py --skip-preprocessing --skip-features
```

---

## Fichiers produits

```
results_dir/
├── analysis/
│   ├── all_features.csv              # Features brutes par sujet/condition/mode
│   ├── inter_subject_summary.csv     # Résumé inter-sujets
│   ├── intra_subject_diff_<cond>.csv # Différences simulation vs optimization
│   └── clusters.csv                  # Clustering focalisation/intensité
└── figures/
    ├── roi_masks.png
    ├── efields_histograms_intra_*.png
    ├── efields_histograms_extra_*.png
    └── simulation_vs_optimization_*.png
```
