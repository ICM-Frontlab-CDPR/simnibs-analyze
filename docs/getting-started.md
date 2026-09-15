# Getting Started

## Installation

```bash
pip install simnibs-analyze
```

Pulls in [`simnibs-reader`](https://pypi.org/project/simnibs-reader/), which
handles all NIfTI reading and ROI extraction.

You need SimNIBS outputs already computed: a head model per subject
(`m2m_<subID>/`, from `charm`) and a simulation or optimization folder.

## The two commands

Both are driven entirely by a YAML file passed with `--config`. The schemas are
different and not interchangeable.

### `simnibs-analyze` — features and statistics

```bash
simnibs-analyze --config config-analyze.yaml
```

Per (subject × condition × mode) it extracts the target ROI, post-processes it,
computes intra-ROI stats, extra-ROI stats and their ratio, then appends a row to
`all_features_space-<space>.csv`. It finishes with the inter/intra-subject
analysis and clustering, writing to `paths.out_root`.

Feature extraction is the slow part. Once the CSV exists, `--skip-features`
re-runs only the statistics; `--skip-analysis` stops after extraction.

→ [Configuration reference](configuration-analyze.md)

### `simnibs-viz` — figures and cohort montages

```bash
simnibs-viz --config config-viz.yaml
```

Renders one figure per subject for each figure block, then composes cohort
montages for the blocks flagged `cohort: true`, on a single shared e-field
scale. Output goes to `paths.out_root`, one directory per
`{subject}_{simulation}`, with montages in `_cohort/`.

→ [Configuration reference](configuration-viz.md)

## Checking a config before running

```bash
python -m simnibs_analyze._config_schema_analyze --config my-config.yaml
```

Prints a one-line summary, or the validation error and exit code `1`.

## Development

```bash
git clone https://github.com/ICM-Frontlab-CDPR/simnibs-analyze.git
cd simnibs-analyze
pip install -e ".[dev]"
pytest
```
