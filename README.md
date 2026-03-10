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
