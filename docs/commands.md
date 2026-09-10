# Commands

`simnibs-analyze` installs two console scripts. Both take a single required
`--config` argument pointing at a YAML file.

---

## `simnibs-analyze`

Runs the feature-extraction and statistical-analysis pipeline.

```bash
simnibs-analyze --config mkdocs/config/config-analyze_htacs.yaml
```

| Argument | Required | Description |
|---|---|---|
| `--config PATH` | yes | Path to the analysis YAML config (validated against `PipelineConfig`) |
| `--skip-features` | no | Reuse the existing `all_features_space-<space>.csv` instead of recomputing it |
| `--skip-analysis` | no | Stop after feature extraction; skip inter/intra-subject analysis |

!!! tip "Iterating on the analysis"
    Feature extraction is the slow part. Once `all_features_*.csv` exists, use
    `--skip-features` to re-run only the statistics while you tune thresholds.

```bash
simnibs-analyze --config config-analyze.yaml --skip-features
```

---

## `simnibs-viz`

Renders per-subject figures and cohort montages.

```bash
simnibs-viz --config mkdocs/config/config-viz_stimSD.yaml
```

| Argument | Required | Description |
|---|---|---|
| `--config PATH` | yes | Path to the visualisation YAML config (validated against `VizConfig`) |

Figures are written under `paths.out_root`, one directory per
`{subject}_{simulation}` key, with cohort montages in `_cohort/`.

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Pipeline completed |
| `1` | Runtime failure (e.g. `--skip-features` but the CSV is missing) |
| `2` | Bad invocation — missing `--config`, or the config file does not exist |

---

## Running without installing

Both modules are importable and expose the same entry points, so they can be
invoked directly from a source checkout:

```bash
python -m simnibs_analyze.run_analyze --config config-analyze.yaml
python -m simnibs_analyze.run_viz     --config config-viz.yaml
```
