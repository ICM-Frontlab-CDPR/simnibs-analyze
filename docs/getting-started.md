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
entirely by a YAML config file passed with `--config`. The two schemas are
**different and not interchangeable** — see [Configuration](configuration.md)
for every key.

---

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

#### Example config

Fully commented, every key shown with its default:

[:material-download: config_analyze_example.yaml](examples/config/config_analyze_example.yaml){ .md-button .md-button--primary download="config_analyze_example.yaml" }

---

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

!!! warning "3D figures need Playwright"
    `type: 3D` figures are rendered by driving a headless browser over a NiiVue
    WebGL scene. That needs two steps, and the second is easy to forget — it
    downloads the browser itself:

    ```bash
    pip install "simnibs-analyze[viz3d]"
    playwright install chromium
    ```

    2D figures have no such requirement.

#### Example config

Covers 2D ortho, 2D slice mosaics and 3D rendering, with the layer registry
explained inline:

[:material-download: config_viz_example.yaml](examples/config/config_viz_example.yaml){ .md-button .md-button--primary download="config_viz_example.yaml" }

---

## Checking a config before running

Validate the file without launching the pipeline:

```bash
python -m simnibs_analyze._config_schema_analyze --config my-config.yaml
```

It prints a one-line summary on success, or the validation error and exit
code `1` on failure.
