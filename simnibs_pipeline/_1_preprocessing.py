
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import nibabel as nib
from nilearn import image, masking

from _pipeline_io import load_img

#TODO add condition sur mni-space ou subject space (c'est dans un get_fpath que ça doit etre défini)
class Preprocessor:
    """
    Preprocess an e-field image within a ROI.

    Parameters are set once at construction and applied to each call of :meth:`run`.

    Parameters
    ----------
    smooth_fwhm :
        FWHM (mm) for Gaussian smoothing. Set to None or 0 to skip.
    outlier_method : {"iqr", "z"}
        Outlier removal strategy.
    portion :
        Central portion of values to keep (e.g. 0.95). None to skip.
    """

    def __init__(
        self,
        smooth_fwhm: float | None = 2.0,
        outlier_method: str = "iqr",
        portion: float | None = None,
    ) -> None:
        self.smooth_fwhm = smooth_fwhm
        self.outlier_method = outlier_method
        self.portion = portion

        # Outputs — populated after run()
        self.masked_img: nib.Nifti1Image | None = None
        self.cleaned_img: nib.Nifti1Image | None = None
        self.filtered_values: np.ndarray | None = None

    @staticmethod
    def build_extra_mask(roi_mask_path: Path) -> nib.Nifti1Image:
        """
        Construit le masque extra-ROI en inversant le masque ROI binaire.

        Parameters
        ----------
        roi_mask_path : Path
            Chemin vers le masque ROI binaire.

        Returns
        -------
        nib.Nifti1Image
            Image binaire : 1 partout sauf dans la ROI.
        """
        return image.math_img("1 - img", img=load_img(roi_mask_path))

    @staticmethod
    def _remove_outliers(
        values: np.ndarray,
        method: str = "iqr",
        z_thresh: float = 3.5,
        iqr_factor: float = 1.5,
        portion: float | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filtre les outliers sur un vecteur 1D.
        - method="iqr": conserve les valeurs dans [Q1 - iqr_factor*IQR, Q3 + iqr_factor*IQR]
        - method="z": conserve les valeurs avec |z| <= z_thresh (z-score robuste)
        - portion: si défini, garde la portion centrale (ex: 0.95 garde 2.5%-97.5%)

        Returns
        -------
        filtered_values : np.ndarray
            Valeurs filtrées (NaN sur les outliers).
        keep_mask : np.ndarray
            Masque booléen des valeurs conservées.
        """
        vals = np.asarray(values, dtype=float)
        keep = np.ones(vals.shape, dtype=bool)

        if portion is not None:
            if not (0.0 < portion <= 1.0):
                raise ValueError("portion must be in (0, 1]")
            lower_q = (1.0 - portion) / 2.0
            upper_q = 1.0 - lower_q
            low, high = np.quantile(vals, [lower_q, upper_q])
            keep &= (vals >= low) & (vals <= high)

        if method.lower() == "iqr":
            q1, q3 = np.quantile(vals, [0.25, 0.75])
            iqr = q3 - q1
            low = q1 - iqr_factor * iqr
            high = q3 + iqr_factor * iqr
            keep &= (vals >= low) & (vals <= high)
        elif method.lower() == "z":
            med = np.median(vals)
            mad = np.median(np.abs(vals - med))
            if mad == 0:
                keep &= np.isfinite(vals)
            else:
                robust_z = 0.6745 * (vals - med) / mad
                keep &= np.abs(robust_z) <= z_thresh
        else:
            raise ValueError("method must be 'iqr' or 'z'")

        filtered = vals.copy()
        filtered[~keep] = np.nan
        return filtered, keep

    def target_grab(
        self,
        efield_img: nib.nifti1.Nifti1Image | Path | str,
        roi_mask: nib.nifti1.Nifti1Image | Path | str,
    ) -> np.ndarray:
        """Extract e-field values inside the ROI mask."""
        return masking.apply_mask(load_img(efield_img), load_img(roi_mask))

    def run(
        self,
        efield_img: nib.nifti1.Nifti1Image | Path | str,
        roi_mask: nib.nifti1.Nifti1Image | Path | str,
    ) -> "Preprocessor":
        """
        Run the full preprocessing pipeline.

        Results are stored as attributes:
        - ``masked_img``     : e-field masked by ROI (before outlier removal)
        - ``cleaned_img``    : e-field masked and cleaned
        - ``filtered_values``: 1-D array of cleaned values

        Returns self for optional chaining.
        """
        efield = load_img(efield_img)
        roi = load_img(roi_mask)

        if not np.allclose(roi.affine, efield.affine):
            roi = image.resample_to_img(roi, efield, interpolation="nearest")

        if self.smooth_fwhm and self.smooth_fwhm > 0:
            efield = image.smooth_img(efield, fwhm=self.smooth_fwhm)

        roi_values = masking.apply_mask(efield, roi)
        self.masked_img = masking.unmask(roi_values, roi)

        self.filtered_values, _ = self._remove_outliers(
            roi_values, method=self.outlier_method, portion=self.portion
        )
        self.cleaned_img = masking.unmask(self.filtered_values, roi)
        return self


def _parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description="Preprocess e-field in ROI")
    parser.add_argument("--efield", required=True, help="Path to e-field NIfTI")
    parser.add_argument("--roi", required=True, help="Path to ROI mask NIfTI")
    parser.add_argument("--out", required=True, help="Output path for cleaned ROI NIfTI")
    parser.add_argument("--smooth-fwhm", type=float, default=2.0, help="FWHM for smoothing")
    parser.add_argument("--outlier-method", choices=["iqr", "z"], default="iqr")
    parser.add_argument("--portion", type=float, default=None,
                        help="Central portion to keep (e.g., 0.95)")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    from _pipeline_io import save_nifti
    args = _parse_args(argv)
    preproc = Preprocessor(
        smooth_fwhm=args.smooth_fwhm,
        outlier_method=args.outlier_method,
        portion=args.portion,
    ).run(args.efield, args.roi)
    save_nifti(preproc.cleaned_img, Path(args.out))
