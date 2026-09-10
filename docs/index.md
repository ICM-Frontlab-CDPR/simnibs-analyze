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

See [Getting Started](getting-started.md) for the full argument reference.

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

## What it covers

- **Target definition** — ROI masks from MNI coordinates or atlas parcels (sphere, atlas-based)
- **E-field preparation** — smoothing, masking, intra/extra-ROI decomposition
- **Feature extraction** — mean, median, std, min, max and focality ratio, per subject and condition
- **Group analysis** — inter-subject summaries, intra-subject contrasts
- **Clustering** — `specific`/`diffuse` × `high`/`low` labelling from the focality ratio
- **Visualisation** — 2D overlays, 3D surface renders, lesion overlays, EEG electrode caps, cohort montages

---

## Relationship to simnibs-reader

| Package | Responsibility |
|---|---|
| [`simnibs-reader`](https://github.com/ICM-Frontlab-CDPR/simnibs-reader) | Read SimNIBS output trees, expose e-field volumes, extract ROIs, compute stats |
| `simnibs-analyze` | Orchestrate across a cohort, validate configs, analyse, cluster, visualise |

No direct `nibabel` / `nilearn` plumbing lives in the pipeline code — it all goes
through the reader.

---

## Citation

If you use this pipeline in your research, please cite it via the
[`CITATION.cff`](https://github.com/ICM-Frontlab-CDPR/simnibs-analyze/blob/main/CITATION.cff)
file in the repository.

The pipeline depends on several open-source tools whose references are printed
at the start of each run. For a machine-readable BibTeX summary:

```bash
pip install "simnibs-analyze[citations]"
DUECREDIT_ENABLE=1 simnibs-analyze --config config.yaml
python -m duecredit summary --format bibtex
```

Key tools to cite: SimNIBS, nilearn, nibabel, NumPy, pandas, matplotlib.
