# simnibs-pipeline

Pipeline d'analyse des e-fields SimNIBS : préprocessing des volumes NIfTI, extraction de features intra/extra-ROI, analyse inter/intra-sujets et visualisations. Conçu pour les études de stimulation cérébrale non-invasive (TMS/tDCS) avec registration MNI via ANTsPy.

## Installation

```bash
# TODO: publication sur PyPI
pip install simnibs-pipeline
```

En attendant, cloner le dépôt et installer les dépendances manuellement :

```bash
git clone <repo>
cd simnibs-pipeline
pip install -e .
```

## Documentation

| Ressource                                        | Description                                  |
| ------------------------------------------------ | -------------------------------------------- |
| [Documentation API](docs/api/simnibs_pipeline.html) | Classes et fonctions (généré par pdoc)    |
| [Référence config.yaml](docs/configuration.md)    | Toutes les clés du fichier de configuration |
| [Structure des outputs](docs/output_structure.md) | Fichiers générés dans simnibs_output/ et results_dir/ |

<!-- ## Citation

If you use this pipeline in your research, please cite it via the `CITATION.cff` file included in this repository.

The pipeline also depends on several open-source tools. Their references are printed automatically at the start of each run. For a machine-readable BibTeX summary, install [duecredit](https://github.com/duecredit/duecredit) (optional) and run:

```bash
pip install "simnibs-pipeline[citations]"  # installs duecredit
DUECREDIT_ENABLE=1 simnibs-analyze --config config.yaml
python -m duecredit summary --format bibtex
```

Key tools to cite: SimNIBS, ANTsPy, nilearn, nibabel, NumPy, pandas, matplotlib, PyVista. -->
