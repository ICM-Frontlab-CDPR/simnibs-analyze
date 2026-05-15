# simnibs-analyze

Post-processing and analysis pipeline for SimNIBS e-field outputs. 

Built to facilitate the analysis of simnibs simulations in the context of non-invasive brain stimulation studies (TMS/tDCS).


## What it does

Starting from SimNIBS outputs, the pipeline covers the full analysis workflow:

- **Target definition** — generate ROI masks in MNI and subject space from MNI coordinates or atlas parcels (sphere, atlas-based)
- **E-field preparation** — coregister, skull-strip, smooth, and mask NIfTI volumes; intra/extra-ROI decomposition
- **E-field analysis** — extract scalar features (mean, max, percentiles, focality ratio) per subject and condition
- **Single-subject optimisation assessment** — evaluate how well a given montage targets the intended ROI
- **Simulation robustness** — assess sensitivity of the e-field distribution to input variability
- **Stimulation method comparison** — contrast montages or stimulation parameters across conditions
- **Group-level analysis** — inter-subject summary statistics, condition comparisons, and effect-size reporting
- **Visualisation** — 2D slice overlays, 3D surface rendering, histograms, and group bar plots

## Installation

```bash
# TODO: publication sur PyPI
pip install simnibs-analyze
```


## **Prerequisite Data (Input structure from simnibs):**

You need to have already run:

- simnibs-simulation or/ andsimnibs-optimization folder
- simnibs-m2m folder

## Quick start:

- prepare a config file : use examples from (add link)
- then run:
  simnibs-analyze --config="pathToYourConfig.yaml"


## Other parameters of the pipeline:



## Documentation

For more detailed usecases, and if you need to run particular function of the pipeline:

| Ressource                                        | Description                                              |
| ------------------------------------------------ | -------------------------------------------------------- |
| [Documentation API](docs/api/simnibs_pipeline.html) | Classes et fonctions (généré par pdoc)                |
| [Référence config.yaml](docs/configuration.md)    | Toutes les clés du fichier de configuration             |
| [Structure des outputs](docs/output_structure.md)   | Fichiers générés dans simnibs_output/ et results_dir/ |

<!-- ## Citation

If you use this pipeline in your research, please cite it via the `CITATION.cff` file included in this repository.

The pipeline also depends on several open-source tools. Their references are printed automatically at the start of each run. For a machine-readable BibTeX summary, install [duecredit](https://github.com/duecredit/duecredit) (optional) and run:

```bash
pip install "simnibs-pipeline[citations]"  # installs duecredit
DUECREDIT_ENABLE=1 simnibs-analyze --config config.yaml
python -m duecredit summary --format bibtex
```

Key tools to cite: SimNIBS, ANTsPy, nilearn, nibabel, NumPy, pandas, matplotlib, PyVista. -->
