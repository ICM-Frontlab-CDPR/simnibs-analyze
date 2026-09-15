# simnibs-analyze

> Cohort-level analysis and visualisation of SimNIBS e-field simulations, driven by YAML config.
>
> `simnibs-analyze` turns SimNIBS outputs into ROI statistics and publication figures
> across subjects and conditions, for non-invasive brain stimulation studies (TMS/tDCS).

[![PyPI version](https://badge.fury.io/py/simnibs-analyze.svg)](https://pypi.org/project/simnibs-analyze/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://ICM-Frontlab-CDPR.github.io/simnibs-analyze/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

📖 **Full documentation → [ICM-Frontlab-CDPR.github.io/simnibs-analyze](https://ICM-Frontlab-CDPR.github.io/simnibs-analyze/)**

---

## Install & run

```bash
pip install simnibs-analyze
```

Two config-driven commands:

```bash
simnibs-analyze --config config-analyze.yaml   # features + statistics
simnibs-viz     --config config-viz.yaml       # figures + cohort montages
```

Commented example configs in [`docs/examples/config/`](docs/examples/config/).
Full key reference: [analyze](https://ICM-Frontlab-CDPR.github.io/simnibs-analyze/configuration-analyze/) · [viz](https://ICM-Frontlab-CDPR.github.io/simnibs-analyze/configuration-viz/).

---

## Ecosystem

`simnibs-analyze` is built on:

* [simnibs-reader](https://github.com/ICM-Frontlab-CDPR/simnibs-reader) (structured access to SimNIBS outputs)

It consumes results produced by:

* simnibs
* [simnibs-modular](https://github.com/ICM-Frontlab-CDPR/simnibs-modular) (damaged-brain simnibs preparation)

---
