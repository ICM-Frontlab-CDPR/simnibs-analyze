"""
Convert preprocessed e-fields to scalar values stored in CSV files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Dict, Any

import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img

from .._pipeline_io import load_img, validate_binary, save_rows


class FeatureExtractor:
    """
    Extracts scalar statistics from preprocessed e-field images.

    Parameters
    ----------
    ratio_methods :
        Methods used to compute the intra/extra-ROI e-field ratio when a
        full-brain image is provided.  Each method produces an
        ``efield_ratio_<method>`` column in the output row.
    """

    def __init__(self, ratio_methods: tuple[str, ...] = ("mean",)) -> None:
        self.ratio_methods = ratio_methods
        self.row: Dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Private helpers (static so they are usable without an instance)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_values(efield_img, roi_mask=None) -> np.ndarray:
        img = load_img(efield_img)
        data = img.get_fdata(dtype=np.float32)
        if roi_mask is not None:
            mask_img = load_img(roi_mask)
            mask = mask_img.get_fdata().astype(bool)
            values = data[mask]
        else:
            # For preprocessed files, take only non-zero values
            # (values outside the ROI are set to 0 after unmasking)
            values = data.ravel()
            # Filter out zeros AND NaNs
            values = values[(values != 0) & np.isfinite(values)]
        # For raw files, filter out NaNs only
        if roi_mask is not None:
            values = values[np.isfinite(values)]
        return values

    @staticmethod
    def compute_stats(values: np.ndarray) -> Dict[str, Any]:
        """Return basic descriptive statistics for an array of values."""
        if values.size == 0:
            return {
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "n_voxels": 0,
            }
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "n_voxels": int(values.size),
        }

    @staticmethod
    def compute_efield_ratio(
        full_efield_img: nib.nifti1.Nifti1Image,
        roi_img: nib.nifti1.Nifti1Image,
        method: str = "mean",
    ) -> float:
        """
        Compute the intra-ROI / extra-ROI e-field ratio.

        Parameters
        ----------
        full_efield_img :
            Full-brain e-field magnitude image (nibabel).
        roi_img :
            Binary ROI mask image (nibabel).
        method : {"mean"}
            Statistic used to summarise intra and extra-ROI distributions.

        Returns
        -------
        float
            Ratio intra/extra.  Returns ``np.nan`` if method is unknown.
        """
        efield_data = np.squeeze(full_efield_img.get_fdata(dtype=np.float32))
        roi_data = np.squeeze(roi_img.get_fdata())

        # Resample ROI mask if grids differ
        if efield_data.shape != roi_data.shape:
            roi_img = resample_to_img(roi_img, full_efield_img, interpolation="nearest")
            roi_data = np.squeeze(roi_img.get_fdata())

        validate_binary(roi_data, name="ROI mask")
        roi_mask = roi_data.astype(bool)

        intra = efield_data[roi_mask]
        extra = efield_data[~roi_mask]

        if method == "mean":
            metric_intra = float(np.mean(intra)) if intra.size > 0 else 0.0
            metric_extra = float(np.mean(extra)) if extra.size > 0 else 0.0
        else:
            return np.nan

        if metric_extra < 1e-10:
            metric_extra = 1e-10
        return metric_intra / metric_extra

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        efield_path: Path,
        roi_path: Path | None,
        subject: str | None,
        condition: str | None,
        full_efield_img: nib.nifti1.Nifti1Image | None = None,
        ratio_roi_path: Path | None = None,
    ) -> "FeatureExtractor":
        """
        Build a feature row for a single e-field file.

        The result is stored in ``self.row`` and ``self`` is returned for
        method chaining.

        Parameters
        ----------
        efield_path :
            Path to preprocessed (masked) e-field NIfTI.
        roi_path :
            ROI mask path (``None`` if the e-field is already masked).
        subject :
            Subject identifier.
        condition :
            Condition label (e.g. ``"fef_simulation"``).
        full_efield_img :
            Full-brain e-field image (nibabel). When provided together with
            ``ratio_roi_path`` (or ``roi_path`` as fallback),
            ``efield_ratio_<method>`` columns are added for each method in
            ``self.ratio_methods``.
        ratio_roi_path :
            ROI mask used exclusively for ratio computation.  Useful when
            ``roi_path`` is ``None`` because the e-field is already masked
            (e.g. a cleaned/preprocessed file) but the ratio still needs the
            original binary ROI.
        """
        values = self._extract_values(efield_path, roi_path)
        stats = self.compute_stats(values)
        row: Dict[str, Any] = {
            "efield_path": str(efield_path),
            "roi_path": str(roi_path) if roi_path else "",
        }
        if subject is not None:
            row["subject"] = subject
        if condition is not None:
            row["condition"] = condition
        row.update(stats)

        _ratio_roi = ratio_roi_path or roi_path
        if full_efield_img is not None and _ratio_roi is not None:
            roi_img = load_img(_ratio_roi)
            for m in self.ratio_methods:
                row[f"efield_ratio_{m}"] = self.compute_efield_ratio(
                    full_efield_img, roi_img, method=m
                )

        self.row = row
        return self


def _parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description="Extract ROI e-field features to CSV")
    parser.add_argument(
        "--efield",
        nargs="+",
        required=True,
        help="Path(s) to preprocessed ROI e-field NIfTI",
    )
    parser.add_argument(
        "--roi", default=None, help="ROI mask path (optional if already masked)"
    )
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--subject", default=None, help="Subject ID")
    parser.add_argument(
        "--condition", default=None, help="Condition label (e.g., simu/opti)"
    )
    parser.add_argument(
        "--if-exists", choices=["overwrite", "skip", "error"], default="overwrite"
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    roi_path = Path(args.roi) if args.roi else None
    extractor = FeatureExtractor()
    rows = []
    for efield in args.efield:
        extractor.run(Path(efield), roi_path, args.subject, args.condition)
        if extractor.row is not None:
            rows.append(extractor.row)
    save_rows(rows, Path(args.out), if_exists=args.if_exists)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
