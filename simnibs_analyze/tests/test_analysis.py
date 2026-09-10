"""Tests for the inter/intra-subject Analysis step."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simnibs_analyze.steps.analysis import Analysis


class TestInterSubjectSummary:
    def test_one_row_per_condition(self, features_df: pd.DataFrame) -> None:
        out = Analysis(features_df).inter_subject_summary()
        assert set(out["condition"]) == {"fef", "ips"}
        assert len(out) == 2

    def test_mean_is_correct(self, features_df: pd.DataFrame) -> None:
        out = Analysis(features_df).inter_subject_summary()
        fef = out.loc[out["condition"] == "fef", "mean"].item()
        assert fef == pytest.approx(2.0)  # (1.0 + 3.0) / 2

    def test_sem_matches_std_over_sqrt_n(self, features_df: pd.DataFrame) -> None:
        out = Analysis(features_df).inter_subject_summary()
        row = out.loc[out["condition"] == "fef"].iloc[0]
        assert row["sem"] == pytest.approx(row["std"] / np.sqrt(row["count"]))

    def test_missing_metric_raises(self, features_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="metric"):
            Analysis(features_df).inter_subject_summary(metric="absent")

    def test_missing_condition_col_raises(self, features_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError):
            Analysis(features_df).inter_subject_summary(condition_col="absent")


class TestIntraSubjectDiff:
    def test_difference_is_b_minus_a(self, features_df: pd.DataFrame) -> None:
        out = Analysis(features_df).intra_subject_diff(cond_a="fef", cond_b="ips")
        # subject 0001: ips(2.0) - fef(1.0) = 1.0 ; subject 0002: 4.0 - 3.0 = 1.0
        assert out.shape[0] == 2

    def test_unknown_condition_raises(self, features_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="Missing conditions"):
            Analysis(features_df).intra_subject_diff(cond_a="fef", cond_b="nope")


class TestAssignClusters:
    @pytest.fixture
    def ratio_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "subject": ["0001", "0002", "0003", "0004"],
                "mean": [1.0, 2.0, 3.0, 4.0],
                "efield_ratio_mean": [2.0, 0.5, 3.0, 1.0],
            }
        )

    def test_cluster_column_added(self, ratio_df: pd.DataFrame) -> None:
        out = Analysis(ratio_df).assign_clusters()
        assert "cluster" in out.columns

    def test_labels_are_from_expected_set(self, ratio_df: pd.DataFrame) -> None:
        out = Analysis(ratio_df).assign_clusters()
        allowed = {"specific_high", "specific_low", "diffuse_high", "diffuse_low"}
        assert set(out["cluster"]) <= allowed

    def test_specificity_threshold_applied(self, ratio_df: pd.DataFrame) -> None:
        out = Analysis(ratio_df).assign_clusters(specificity_threshold=1.5)
        # ratios 2.0 and 3.0 are above 1.5 -> specific ; 0.5 and 1.0 -> diffuse
        assert out.loc[out["efield_ratio_mean"] == 2.0, "cluster"].item().startswith("specific")
        assert out.loc[out["efield_ratio_mean"] == 0.5, "cluster"].item().startswith("diffuse")

    def test_original_df_not_mutated(self, ratio_df: pd.DataFrame) -> None:
        Analysis(ratio_df).assign_clusters()
        assert "cluster" not in ratio_df.columns

    def test_missing_ratio_column_raises(self, features_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="efield_ratio_mean"):
            Analysis(features_df).assign_clusters()
