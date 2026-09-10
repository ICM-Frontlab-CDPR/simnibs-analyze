# Getting Started

## Installation

```bash
pip install simnibs-analyze
```

This pulls in [`simnibs-reader`](https://pypi.org/project/simnibs-reader/) `>= 0.2.0`,
which handles all NIfTI reading and ROI extraction.

## Prerequisites

You need SimNIBS outputs already computed:

- a head-model folder per subject — `m2m_<subID>/`, produced by `charm`
- a simulation and/or optimization folder per subject

## The two commands

Installing the package puts two commands on your `PATH`. Both are driven
entirely by a YAML config file passed with `--config`.

### `simnibs-analyze` — features and statistics

```bash
simnibs-analyze --config config-analyze.yaml
```

Per (subject × condition × mode) it extracts the target ROI, post-processes it,
computes intra-ROI stats, extra-ROI stats and their ratio, then appends a row to
`all_features_space-<space>.csv`. It finishes with the inter/intra-subject
analysis and clustering, writing summary CSVs to `paths.results_dir`.

| Argument | Required | Description |
|---|---|---|
| `--config PATH` | yes | Analysis YAML config, validated against `PipelineConfig` |
| `--skip-features` | no | Reuse the existing `all_features_space-<space>.csv` |
| `--skip-analysis` | no | Stop after feature extraction |

!!! tip "Iterating on statistics"
    Feature extraction is the slow part. Once the CSV exists, use
    `--skip-features` to re-run only the statistics while tuning thresholds:

    ```bash
    simnibs-analyze --config config-analyze.yaml --skip-features
    ```

### `simnibs-viz` — figures and cohort montages

```bash
simnibs-viz --config config-viz.yaml
```

Renders one figure per subject for each figure block in the config, then
composes cohort montages for the blocks flagged `cohort: true`, using a single
shared e-field colour scale.

| Argument | Required | Description |
|---|---|---|
| `--config PATH` | yes | Visualisation YAML config, validated against `VizConfig` |

Output goes to `paths.out_root`, one directory per `{subject}_{simulation}`,
with montages in `_cohort/`.

## Writing the config

The two commands take **different, non-interchangeable** config schemas. See
[Configuration](configuration.md) for every key, and the ready-made examples in
`docs/examples/config/` of the repository.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Completed |
| `1` | Runtime failure (e.g. `--skip-features` but the CSV is missing) |
| `2` | Bad invocation — missing `--config`, or the file does not exist |

## Running from a source checkout

Both modules are importable, so you can run them without installing:

```bash
python -m simnibs_analyze.run_analyze --config config-analyze.yaml
python -m simnibs_analyze.run_viz     --config config-viz.yaml
```
