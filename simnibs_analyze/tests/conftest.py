"""Shared fixtures for the simnibs-analyze test suite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def minimal_analyze_cfg(tmp_path: Path) -> dict:
    """Smallest dict that satisfies PipelineConfig."""
    return {
        "subjects": ["0001"],
        "stim_conditions": ["fef"],
        "mode": ["simulation"],
        "space": "mni",
        "target_generation": {
            "radius_mm": 10.0,
            "rois": {"fef": {"method": "sphere", "coords": [30.0, -2.0, 50.0]}},
        },
        "paths": {
            "sim_base": str(tmp_path / "simu"),
            "seg_base": str(tmp_path / "preps"),
            "out_root": str(tmp_path / "results"),
        },
    }


@pytest.fixture
def features_df() -> pd.DataFrame:
    """A small features table shaped like all_features_space-*.csv."""
    return pd.DataFrame(
        {
            "subject": ["0001", "0001", "0002", "0002"],
            "condition": ["fef", "ips", "fef", "ips"],
            "mean": [1.0, 2.0, 3.0, 4.0],
            "ratio_mean": [1.2, 0.8, 2.4, 0.5],
        }
    )
