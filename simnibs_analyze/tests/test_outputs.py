"""The analysis writes one extraction table, not two copies of it.

`clusters_<tag>.csv` used to be written alongside `all_features_<tag>.csv`,
but `assign_clusters` returns the features table with one extra column — so
the whole table was duplicated on disk to carry a single label.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from simnibs_analyze import run_analyze
from simnibs_analyze._config_schema_analyze import PipelineConfig
from simnibs_analyze.steps.analysis import Analysis


class TestSingleExtractionFile:
    def test_no_separate_clusters_file(self) -> None:
        src = inspect.getsource(run_analyze.run_analysis)
        assert "clusters_" not in src

    def test_clusters_written_back_into_features(self) -> None:
        src = inspect.getsource(run_analyze.run_analysis)
        assert "clustered.to_csv(features_csv" in src

    def test_cluster_is_only_an_added_column(self) -> None:
        """Justifies folding the two files together."""
        df = pd.DataFrame(
            {
                "subject": ["0001", "0002"],
                "mean": [1.0, 2.0],
                "efield_ratio_mean": [2.0, 0.5],
            }
        )
        out = Analysis(df).assign_clusters()
        assert list(out.columns) == [*df.columns, "cluster"]
        assert len(out) == len(df)


class TestFeaturesPath:
    @pytest.fixture
    def cfg(self, minimal_analyze_cfg: dict) -> PipelineConfig:
        return PipelineConfig.model_validate(minimal_analyze_cfg)

    def test_name_encodes_the_space(self, cfg: PipelineConfig) -> None:
        assert run_analyze.features_csv_path(cfg).name == "all_features_space-mni.csv"

    def test_lives_under_out_root(self, cfg: PipelineConfig) -> None:
        assert run_analyze.features_csv_path(cfg).parent == cfg.paths.out_root

    def test_native_space_gets_its_own_file(self, minimal_analyze_cfg: dict) -> None:
        minimal_analyze_cfg["space"] = "native"
        cfg = PipelineConfig.model_validate(minimal_analyze_cfg)
        assert run_analyze.features_csv_path(cfg).name == "all_features_space-native.csv"


class TestRunReportsProducedFiles:
    def test_main_lists_what_it_wrote(self) -> None:
        src = inspect.getsource(run_analyze.main)
        assert 'glob("*.csv")' in src
