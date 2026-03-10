# Structure des fichiers produits par le pipeline

Le pipeline écrit ses sorties dans deux arbres distincts :

1. **`paths.simnibs_output/`** — fichiers ajoutés dans l'arborescence SimNIBS existante (masques, preprocessing)
2. **`paths.results_dir/`** — résultats d'analyse et figures

---

## 1. Ajouts dans `simnibs_output/`

### Dossier partagé (une fois, tous sujets)

```
simnibs_output/
└── mni_target/
    ├── fef_mask.nii.gz          # Masque sphérique ROI FEF en espace MNI
    ├── ips_left_mask.nii.gz     # Masque sphérique ROI IPS gauche
    └── ips_right_mask.nii.gz   # Masque sphérique ROI IPS droit
```

Généré par `AnatomicalPreparer.setup()` — une sphère de `radius_mm` autour des coordonnées MNI définies dans `config.rois`.

---

### Par sujet

```
simnibs_output/
└── <subject>/
    ├── subject_target/
    │   ├── T1_MNI_brain.nii.gz          # T1 du sujet recalé en espace MNI (fond pour les figures)
    │   ├── T1_subject_brain.nii.gz      # T1 du sujet en espace natif (fond pour les figures)
    │   ├── fef_mask.nii.gz              # Masque ROI transformé en espace sujet
    │   ├── ips_left_mask.nii.gz
    │   └── ips_right_mask.nii.gz
    │
    ├── simulations/
    │   └── simulation_simulation_<condition>_<hash>/
    │       └── mni_volumes/
    │           ├── <stem>_scalar_MNI_magnE.nii.gz      # E-field brut (MNI)
    │           ├── <stem>_roi_masked.nii.gz             # E-field masqué intra-ROI
    │           ├── <stem>_roi_cleaned.nii.gz            # E-field nettoyé intra-ROI (outliers supprimés)
    │           ├── <stem>_extra_roi_masked.nii.gz       # E-field masqué extra-ROI
    │           └── <stem>_extra_roi_cleaned.nii.gz      # E-field extra-ROI (= masked, pas de filtrage)
    │
    └── optimizations/
        └── optimization_optimization_<condition>_<hash>/
            └── simulation_with_optimal_montage/
                └── mni_volumes/
                    ├── <stem>_scalar_MNI_magnE.nii.gz
                    ├── <stem>_roi_masked.nii.gz
                    ├── <stem>_roi_cleaned.nii.gz
                    ├── <stem>_extra_roi_masked.nii.gz
                    └── <stem>_extra_roi_cleaned.nii.gz
```

> `<stem>` = nom de base du fichier e-field SimNIBS (ex: `ernie_TMS_1-0001_Magstim_70mm_Fig8_nii`).

#### Convention de nommage des fichiers preprocessés

| Suffixe                | Contenu                                                                     |
| ---------------------- | --------------------------------------------------------------------------- |
| `_roi_masked`        | E-field multiplié par le masque ROI intra (voxels hors ROI = 0)            |
| `_roi_cleaned`       | Idem + suppression des outliers IQR sur les voxels non nuls                 |
| `_extra_roi_masked`  | E-field multiplié par le masque complément ROI (tout le cerveau sauf ROI) |
| `_extra_roi_cleaned` | Identique à `_extra_roi_masked` (pas de filtrage pour l'extra-ROI)       |

---

## 2. Arborescence `results_dir/`

```
results_dir/
├── analysis/
│   ├── all_features.csv                    # Features brutes — une ligne par sujet×condition×mode
│   ├── inter_subject_summary.csv           # Résumé inter-sujets (moyenne ± std par condition)
│   ├── intra_subject_diff_fef.csv          # Diff intra-sujet simulation vs optimization (FEF)
│   ├── intra_subject_diff_ips_left.csv     # Idem IPS gauche
│   ├── intra_subject_diff_ips_right.csv    # Idem IPS droit
│   └── clusters.csv                        # Toutes colonnes de all_features + colonne `cluster`
│
├── simu/
│   ├── efields_3d_fef_simulation_mni_xy.png        # Figure 3D e-field (all subjects, MNI)
│   ├── efields_3d_fef_optimization_mni_xy.png
│   ├── efields_3d_ips_left_simulation_mni_xy.png
│   └── ...                                          # Une figure par (ROI × mode × espace × caméra)
│
├── preprocess/
│   ├── efields_histograms_<subject>_intra.png   # Histogrammes avant/après preprocessing intra-ROI
│   └── efields_histograms_<subject>_extra.png   # Histogrammes extra-ROI
│
├── targets/
│   ├── fef_mask_visualization.png           # Masque ROI FEF sur fond MNI
│   ├── ips_left_mask_visualization.png
│   ├── ips_right_mask_visualization.png
│   └── all_masks_combined.png               # Vue combinée tous masques
│
└── figures/
    └── simulation_vs_optimization.png        # Scatter simulation×optimization par ROI
```

---

## 3. Détail de `all_features.csv`

Chaque ligne correspond à un e-field traité (sujet × condition × mode).

| Colonne                                         | Description                                              |
| ----------------------------------------------- | -------------------------------------------------------- |
| `subject`                                     | Identifiant sujet                                        |
| `condition`                                   | `<roi>_simulation` ou `<roi>_optimization`           |
| `mean`, `median`, `std`, `min`, `max` | Métriques intra-ROI (voxels nettoyés)                  |
| `n_voxels`                                    | Nombre de voxels intra-ROI non nuls après preprocessing |
| `extra_mean`, `extra_median`, …            | Mêmes métriques pour l'extra-ROI                       |
| `extra_n_voxels`                              | Nombre de voxels extra-ROI non nuls                      |
| `efield_ratio_mean`                           | `mean / extra_mean` — ratio de focalisation           |

---

## 4. Détail de `clusters.csv`

Reprend toutes les colonnes de `all_features.csv` plus :

| Colonne     | Valeurs possibles                                                    | Description                            |
| ----------- | -------------------------------------------------------------------- | -------------------------------------- |
| `cluster` | `focused_high`, `focused_low`, `diffuse_high`, `diffuse_low` | Combinaison focalisation × intensité |

- **focused** : `efield_ratio_mean` > `analysis.clustering.specificity_threshold`
- **high/low** : `mean` au-dessus/en-dessous de la médiane de la distribution
