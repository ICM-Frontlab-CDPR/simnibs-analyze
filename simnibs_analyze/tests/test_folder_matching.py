"""Folder lookup tolerates both '-' and '_' in condition names.

ROI keys may not contain underscores (they clash with the output filename
convention), but SimNIBS folders often do: 'ips-left' as a key, 'ips_left' on
disk. That mismatch used to be papered over by a `folder_pattern` field the
user had to declare by hand, restating information already in the ROI name.
"""

from __future__ import annotations

from pathlib import Path

from simnibs_analyze.run_analyze import find_simulation_dir


def _make(root: Path, subject: str, folder: str) -> Path:
    d = root / subject / "simulations" / folder
    d.mkdir(parents=True)
    return d


class TestSeparatorTolerance:
    def test_underscore_folder_found_from_hyphen_key(self, tmp_path: Path) -> None:
        _make(tmp_path, "0001", "simulation_simulation_ips_left_study_abc")
        got = find_simulation_dir(tmp_path, "0001", "ips-left", "simulation")
        assert got is not None

    def test_hyphen_folder_found_from_hyphen_key(self, tmp_path: Path) -> None:
        _make(tmp_path, "0001", "simulation_simulation_ips-left_study_abc")
        got = find_simulation_dir(tmp_path, "0001", "ips-left", "simulation")
        assert got is not None

    def test_plain_condition_still_works(self, tmp_path: Path) -> None:
        _make(tmp_path, "0001", "simulation_simulation_fef_study_abc")
        got = find_simulation_dir(tmp_path, "0001", "fef", "simulation")
        assert got is not None


class TestMisses:
    def test_unknown_subject(self, tmp_path: Path) -> None:
        assert find_simulation_dir(tmp_path, "9999", "fef", "simulation") is None

    def test_unknown_condition(self, tmp_path: Path) -> None:
        _make(tmp_path, "0001", "simulation_simulation_fef_study_abc")
        assert find_simulation_dir(tmp_path, "0001", "nope", "simulation") is None

    def test_mode_is_respected(self, tmp_path: Path) -> None:
        _make(tmp_path, "0001", "simulation_simulation_fef_study_abc")
        assert find_simulation_dir(tmp_path, "0001", "fef", "optimization") is None


class TestFolderPatternGone:
    def test_not_in_schema(self) -> None:
        from simnibs_analyze._config_schema_analyze import AtlasROI, SphereROI

        assert "folder_pattern" not in SphereROI.model_fields
        assert "folder_pattern" not in AtlasROI.model_fields
