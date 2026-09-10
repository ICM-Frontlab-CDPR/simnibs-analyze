# simnibs-analyze

[![PyPI](https://img.shields.io/pypi/v/simnibs-analyze)](https://pypi.org/project/simnibs-analyze/)
[![Python](https://img.shields.io/pypi/pyversions/simnibs-analyze)](https://pypi.org/project/simnibs-analyze/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Cohort-level analysis and visualisation of SimNIBS e-field simulations, driven
by a YAML config. Built for non-invasive brain stimulation studies (TMS/tDCS).

`simnibs-analyze` sits on top of
[`simnibs-reader`](https://github.com/ICM-Frontlab-CDPR/simnibs-reader): the
reader handles NIfTI plumbing and ROI extraction, this package handles the
pipeline across subjects and conditions.

## Installation

```bash
pip install simnibs-analyze
```

Requires Python 3.10+. Pulls in `simnibs-reader >= 0.2.0` automatically.

## Prerequisites

SimNIBS must have been run already. You need, per subject:

- a head model — `m2m_<subID>/`, produced by `charm`
- a simulation and/or optimization output folder

## Two commands

```bash
simnibs-analyze --config config-analyze.yaml    # features + statistics
simnibs-viz     --config config-viz.yaml        # figures + cohort montages
```

### `simnibs-analyze`

Per (subject × condition × mode): locates the simulation folder, extracts the
target ROI (sphere / atlas / mask) in the chosen space, smooths and removes
outliers, then computes intra-ROI stats, extra-ROI stats and the intra/extra
ratio — one row per combination in `all_features_space-<space>.csv`. It then runs
the inter/intra-subject analysis and clustering.

```bash
simnibs-analyze --config config-analyze.yaml --skip-features   # reuse the CSV
simnibs-analyze --config config-analyze.yaml --skip-analysis   # features only
```

### `simnibs-viz`

Renders one figure per subject for each figure block, then composes cohort
montages for blocks flagged `cohort: true`, on a single shared e-field scale.
2D orthogonal and parallel-slice views, 3D surface renders, optional EEG
electrode caps.

Ready-made configs live in [`mkdocs/config/`](mkdocs/config/).

## Quick start

```bash
pip install simnibs-analyze
cp mkdocs/config/config-analyze_htacs.yaml my-config.yaml
# edit paths, subjects, stim_conditions and ROIs
simnibs-analyze --config my-config.yaml
```

Check a config without running the pipeline:

```bash
python -m simnibs_analyze._config_schema_analyze --config my-config.yaml
```

## What it covers

- **Target definition** — ROI masks from MNI coordinates or atlas parcels
- **E-field preparation** — smoothing, masking, intra/extra-ROI decomposition
- **Feature extraction** — mean, median, std, min, max, focality ratio per subject and condition
- **Group analysis** — inter-subject summaries, intra-subject contrasts, effect sizes
- **Clustering** — `specific`/`diffuse` × `high`/`low` labelling from the focality ratio
- **Visualisation** — 2D overlays, 3D renders, cohort montages

## Documentation

**[icm-frontlab-cdpr.github.io/simnibs-analyze](https://icm-frontlab-cdpr.github.io/simnibs-analyze/)**

| Page | Contents |
|---|---|
| [Getting Started](docs/getting-started.md) | Installation and both commands |
| [Configuration](docs/configuration.md) | Every key of both YAML schemas |

## Development

```bash
git clone https://github.com/ICM-Frontlab-CDPR/simnibs-analyze.git
cd simnibs-analyze
pip install -e ".[dev]"
pytest
```

Serve the docs locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Citation

If you use this pipeline in your research, please cite it via the
[`CITATION.cff`](CITATION.cff) file in this repository.

The pipeline depends on several open-source tools whose references are printed
at the start of each run. For a machine-readable BibTeX summary:

```bash
pip install "simnibs-analyze[citations]"
DUECREDIT_ENABLE=1 simnibs-analyze --config config.yaml
python -m duecredit summary --format bibtex
```

Key tools to cite: SimNIBS, nilearn, nibabel, NumPy, pandas, matplotlib.

## License

MIT — see [LICENSE](LICENSE).
