---
hide:
  - navigation
---
# simnibs-analyze

**Turn SimNIBS simulation outputs into cohort-level e-field statistics and figures — driven entirely by a YAML config.**

`simnibs-analyze` sits on top of
[`simnibs-reader`](https://pypi.org/project/simnibs-reader/): the reader handles
NIfTI plumbing and ROI extraction, while this package handles the pipeline —
feature extraction across subjects and conditions, inter/intra-subject analysis,
clustering, and publication-ready visualisation.

---

## Installation

```bash
pip install simnibs-analyze
```

This pulls in `simnibs-reader >= 0.2.0` automatically.

---

## Two commands

The package installs two console entry points, both config-driven:

```bash
simnibs-analyze --config config-analyze.yaml   # features + statistics
simnibs-viz     --config config-viz.yaml       # figures + cohort montages
```

See [Commands](commands.md) for the full argument reference.

---

## What the analysis pipeline does

Per (subject × condition × mode) it:

1. locates the simulation folder via the reader,
2. extracts the target ROI (sphere / atlas / mask) in the chosen space,
3. post-processes (smoothing + outlier removal),
4. computes intra-ROI stats, extra-ROI (complement) stats, and the intra/extra ratio,
5. appends one row to `all_features_space-<space>.csv`.

It then runs the inter/intra-subject analysis and clustering, writing the
summary CSVs to `results_dir`.

---

## Relationship to simnibs-reader

| Package | Responsibility |
|---|---|
| [`simnibs-reader`](https://github.com/ICM-Frontlab-CDPR/simnibs-reader) | Read SimNIBS output trees, expose e-field volumes, extract ROIs, compute stats |
| `simnibs-analyze` | Orchestrate across a cohort, validate configs, analyse, cluster, visualise |

No direct `nibabel` / `nilearn` plumbing lives in the pipeline code — it all goes
through the reader.
