"""
simnibs-analyze
===============

Modular pipeline for SimNIBS e-field analysis and visualisation, built on top
of `simnibs-reader <https://pypi.org/project/simnibs-reader/>`_.

Two command-line entry points are installed with the package::

    simnibs-analyze --config path/to/config-analyze.yaml
    simnibs-viz     --config path/to/config-viz.yaml
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "run_analyze",
    "run_viz",
    "steps",
    "_config_schema_analyze",
    "_config_schema_viz",
    "_pipeline_io",
    "_logging",
    "_citations",
    "__version__",
]
