"""The analysis path must declare that config coordinates are MNI.

Config ROI coordinates are always given in MNI. In native space they were
passed straight to the reader, which treated them as subject coordinates: the
sphere landed outside the head, the mask came out empty, and the only symptom
was nilearn's "The mask is invalid as it is empty: it masks all data".
"""

from __future__ import annotations

import inspect

from simnibs_analyze import run_analyze


class TestExtractRoiDeclaresSpace:
    def test_sphere_passes_coords_space_mni(self) -> None:
        src = inspect.getsource(run_analyze._extract_roi)
        assert 'coords_space="mni"' in src

    def test_reader_supports_the_parameter(self) -> None:
        from simnibs_reader.nifti.efield import EField

        assert "coords_space" in inspect.signature(EField.get_roi).parameters


class TestSegmentationAlwaysAttached:
    """It is needed in both spaces: for the brain mask and for the warp."""

    def test_not_gated_on_native(self) -> None:
        src = inspect.getsource(run_analyze.process_subject_condition)
        assert 'if cfg.space == "native"' not in src

    def test_warns_when_m2m_missing(self) -> None:
        src = inspect.getsource(run_analyze.process_subject_condition)
        assert "no m2m folder" in src
