"""Tests for the tabular I/O helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from simnibs_analyze._pipeline_io import check_output, load_csvs, save_dataframe


class TestLoadCsvs:
    def test_concatenates_multiple_files(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.csv", tmp_path / "b.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(a, index=False)
        pd.DataFrame({"x": [3]}).to_csv(b, index=False)
        df = load_csvs([a, b])
        assert list(df["x"]) == [1, 2, 3]

    def test_index_is_reset(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.csv", tmp_path / "b.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(a, index=False)
        pd.DataFrame({"x": [3, 4]}).to_csv(b, index=False)
        df = load_csvs([a, b])
        assert list(df.index) == [0, 1, 2, 3]

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError):
            load_csvs([])


class TestCheckOutput:
    def test_true_when_absent(self, tmp_path: Path) -> None:
        assert check_output(tmp_path / "new.csv") is True

    def test_overwrite_returns_true(self, tmp_path: Path) -> None:
        p = tmp_path / "x.csv"
        p.write_text("a")
        assert check_output(p, "overwrite") is True

    def test_skip_returns_false(self, tmp_path: Path) -> None:
        p = tmp_path / "x.csv"
        p.write_text("a")
        assert check_output(p, "skip") is False

    def test_error_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "x.csv"
        p.write_text("a")
        with pytest.raises(FileExistsError):
            check_output(p, "error")


class TestSaveDataframe:
    def test_creates_parent_dirs(self, tmp_path: Path, features_df: pd.DataFrame) -> None:
        out = tmp_path / "deep" / "nested" / "out.csv"
        save_dataframe(features_df, out, index=False)
        assert out.exists()

    def test_roundtrip(self, tmp_path: Path, features_df: pd.DataFrame) -> None:
        out = tmp_path / "out.csv"
        save_dataframe(features_df, out, index=False)
        assert pd.read_csv(out).shape == features_df.shape

    def test_skip_does_not_overwrite(self, tmp_path: Path, features_df: pd.DataFrame) -> None:
        out = tmp_path / "out.csv"
        out.write_text("untouched")
        save_dataframe(features_df, out, if_exists="skip", index=False)
        assert out.read_text() == "untouched"
