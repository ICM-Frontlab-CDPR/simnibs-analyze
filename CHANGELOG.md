# Changelog

All notable changes to **simnibs-analyze** are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-09-09

First release built on top of the published
[`simnibs-reader`](https://pypi.org/project/simnibs-reader/) package, closing the
`reader ↔ analyze` refactor. The pipeline no longer does its own NIfTI plumbing:
all volume reading, ROI extraction and statistics go through the reader.

Version jumps from `0.0.2` to `0.1.0` — the previous `0.0.x` line predates the
refactor and is not API-compatible.

### Added
- **Two console entry points**, both config-driven:
  - `simnibs-analyze --config <yaml>` — feature extraction and statistics,
    with `--skip-features` and `--skip-analysis`
  - `simnibs-viz --config <yaml>` — per-subject figures and cohort montages
- **Test suite (53 tests)** covering the config schemas, pipeline I/O, the CLI
  contract, the `Analysis` step, and packaging invariants. The packaging tests
  guard specifically against the regressions that broke earlier releases:
  machine-specific paths, missing dependency declarations, and version strings
  drifting apart across metadata files.
- **EEG electrode caps and lesion overlays** in 3D visualisation
  (`electrodes_cap`, `source: lesion_native` / `lesion_mni`).
- **Contour rendering** for ROI and anatomical layers (`render: contour`), with
  per-figure overrides via `contour_vols`.
- **Cohort montages** with a single locked e-field scale across subjects.
- Documentation site built with MkDocs Material, including a complete key
  reference for both YAML schemas.

### Fixed
- **`steps/analysis.py` was unimportable.** It imported `_pipeline_io`, a module
  deleted during the refactor, so the package raised `ImportError` on any
  analysis run. The three functions actually used (`load_csvs`, `check_output`,
  `save_dataframe`) were restored as a minimal module, without the old
  dependency on the equally removed `_config` module.
- **Broken console script.** The declared entry point pointed at
  `simnibs_analyze.run:cli`, which never existed; the real files were
  `run-analyze.py` and `run-viz.py`, unimportable because of the hyphen.
- **`simnibs-reader` was not declared as a dependency**, so an installed copy of
  the package could not run at all.
- **Hardcoded developer paths.** `sys.path.insert(0, "/Users/…/simnibs-reader")`
  in `run_viz.py` and in the config-generation script made the package
  unusable outside one machine.
- **`run-viz.py` argument parsing.** It read `sys.argv[1]` directly, so
  `--config=path` was taken literally as a filename and raised
  `FileNotFoundError`. Both commands now use `argparse`.
- **`_config-schema-analyze.py`** was named with hyphens while the code imported
  `_config_schema_analyze`; a `try/except ImportError` fallback silently hid the
  mismatch and always took the fallback path.
- **`.zenodo.json` was not valid JSON** (leading `#` comment lines), so Zenodo
  archiving metadata was ignored.
- **`CITATION.cff`** declared version `3.0.0`, unrelated to any released version.
- **`__all__`** listed five modules that do not exist.

### Changed
- Package version, `CITATION.cff` and `.zenodo.json` are now kept in sync, and a
  test enforces it.
- Example configs and tutorials moved from `mkdocs/` to `docs/examples/`. The
  old folder held no documentation and became misleading once a real
  `mkdocs.yml` existed at the repo root.
- Documentation migrated from pdoc to MkDocs Material, matching
  `simnibs-reader`. The generated HTML it replaced documented modules that had
  been removed.
- Dependencies corrected: `scipy` added (used but undeclared); `scikit-learn`
  and `pyvista` dropped (declared but unused).

### Removed
- `requirements-sa-dev-p312.txt` and `requirements-sr-dev-p312.txt` — frozen
  environment dumps, one of them belonging to a different repository and the
  other pinning a stale git commit under the package's former name. Use
  `pip install -e ".[dev]"` instead.
- Generated pdoc HTML under `docs/api/` and its generation scripts.

## [0.0.2] — 2026-05-15

Last release of the pre-refactor pipeline, with its own NIfTI handling.

## [0.0.1]

Initial release.
