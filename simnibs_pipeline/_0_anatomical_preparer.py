"""
Target generation step — creates spherical ROI masks in MNI space.

Usage (CLI):
    python _0_anatomical_preparer.py --config config.yaml --output /path/to/mni_target

Usage (API):
    gen = AnatomicalPreparer(radius_mm=10.0)
    gen.setup(rois={"fef": [28, -8, 54]}, output_dir=Path("mni_target"))
    gen.mask_imgs  # dict[str, nib.Nifti1Image]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
from nilearn import datasets, image
from nilearn.image import new_img_like

from _io import load_config, save_nifti, get_t1_conform, get_brainmask
from _logging import get_logger

logger = get_logger(__name__)


class AnatomicalPreparer:
    """
    Generates spherical ROI masks in MNI space from a dict of MNI coordinates.

    Parameters
    ----------
    reference_img_path : Path or None
        Path to a custom MNI template. If None, uses nilearn's MNI152 1 mm.
    radius_mm : float
        Sphere radius in millimetres (default 10.0).
    """

    def __init__(
        self,
        reference_img_path: Optional[Path] = None,
        radius_mm: float = 10.0,
    ) -> None:
        self.radius_mm = radius_mm
        self.mask_imgs: Dict[str, nib.Nifti1Image] = {}

        if reference_img_path is not None:
            logger.info(f"Loading reference image: {reference_img_path}")
            self._template = nib.load(str(reference_img_path))
        else:
            logger.info("Loading standard MNI152 1 mm template")
            self._template = datasets.load_mni152_template(resolution=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(
        self,
        rois: Dict[str, List[float]],
        output_dir: Path,
    ) -> "AnatomicalPreparer":
        """
        Create and save one spherical mask per ROI in MNI space.

        Subject-independent. Call this once before the subject loop.

        Parameters
        ----------
        rois : dict
            ``{roi_name: [x_mni, y_mni, z_mni]}`` coordinates in mm.
        output_dir : Path
            Directory where ``{roi_name}_mask.nii.gz`` files will be written.

        Returns
        -------
        self
            ``self.mask_imgs`` is populated with the generated NIfTI images.
            ``self.mni_output_dir`` is set for use in subsequent ``run()`` calls.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.mni_output_dir = output_dir
        logger.info(f"Generating {len(rois)} ROI mask(s) → {output_dir}")

        for roi_name, mni_coords in rois.items():
            mask_img = self._create_sphere_mask(self._template, mni_coords, self.radius_mm)
            out_path = output_dir / f"{roi_name}_mask.nii.gz"
            save_nifti(mask_img, out_path)
            self.mask_imgs[roi_name] = mask_img
            logger.info(f"  ✓ {roi_name}: {out_path.name}")

        logger.info(f"{len(self.mask_imgs)} mask(s) saved to {output_dir}")
        return self

    def run(
        self,
        m2m_dir: Path,
        output_dir: Path,
    ) -> "AnatomicalPreparer":
        """
        Subject-level processing: skull-strip the T1.

        Call once per subject inside the subject loop, consistent with
        ``Preprocessor.run()`` and ``FeatureExtractor.run()``.
        Uses :func:`_io.get_t1_conform` and :func:`_io.get_brainmask` to
        locate the correct files inside ``m2m_dir``.

        Parameters
        ----------
        m2m_dir : Path
            Path to the SimNIBS ``m2m_<subject>`` directory.
        output_dir : Path
            Directory where subject-space outputs will be written.

        Returns
        -------
        self
            ``self.stripped_t1_path`` is set if skull-stripping was performed.
        """
        m2m_dir = Path(m2m_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.stripped_t1_path: Optional[Path] = None

        try:
            t1_path = get_t1_conform(m2m_dir)
            mask_path = get_brainmask(m2m_dir)
            self.stripped_t1_path = self._skull_strip(t1_path, mask_path)
        except FileNotFoundError as e:
            logger.warning(f"Skull-stripping skipped — {e}")

        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_sphere_mask(
        template_img: nib.Nifti1Image,
        mni_coords: List[float],
        radius_mm: float,
    ) -> nib.Nifti1Image:
        """Return a binary NIfTI sphere mask centred on *mni_coords*."""
        affine = template_img.affine
        data = np.zeros(template_img.shape, dtype=np.uint8)

        # MNI → voxel coordinates
        mni_h = np.append(mni_coords, 1)
        vox = (np.linalg.inv(affine) @ mni_h)[:3].astype(int)

        # Voxel-space radius (isotropic approximation)
        voxel_size = np.abs(np.diag(affine)[:3]).mean()
        radius_vox = radius_mm / voxel_size

        shape = template_img.shape
        x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist_sq = (x - vox[0]) ** 2 + (y - vox[1]) ** 2 + (z - vox[2]) ** 2
        data[dist_sq <= radius_vox ** 2] = 1

        return new_img_like(template_img, data, affine=affine)
    

    def create_subject_roi_from_mni(self):
        """TODO: warp MNI masks to subject space (called from run())."""
        pass

    @staticmethod
    def _skull_strip(t1_path: Path, mask_path: Path) -> Path:
        """
        Apply a brain mask to a T1 image and save the result alongside the T1.

        Parameters
        ----------
        t1_path : Path
            Path to the T1 NIfTI image.
        mask_path : Path
            Path to the binary brain mask NIfTI image.

        Returns
        -------
        Path
            Path to the saved skull-stripped image (``<T1stem>_brain.nii.gz``).
        """
        
        #TODO extraire les io reponsabilités.
        t1_path = Path(t1_path)
        mask_path = Path(mask_path)

        t1_img = nib.load(str(t1_path))
        mask_img = nib.load(str(mask_path))

        # Resample mask to T1 space if needed
        if not np.allclose(mask_img.affine, t1_img.affine) or mask_img.shape != t1_img.shape:
            mask_img = image.resample_to_img(mask_img, t1_img, interpolation="nearest")

        mask_data = (np.asarray(mask_img.dataobj) > 0).astype(t1_img.get_data_dtype())
        stripped_data = np.asarray(t1_img.dataobj) * mask_data
        stripped_img = nib.Nifti1Image(stripped_data, t1_img.affine, t1_img.header)

        # Build output path: strip .nii or .nii.gz suffix, append _brain.nii.gz
        stem = t1_path.name.replace(".nii.gz", "").replace(".nii", "")
        out_path = t1_path.parent / f"{stem}_brain.nii.gz"
        nib.save(stripped_img, str(out_path))
        logger.info(f"Skull-stripped T1 saved → {out_path}")
        return out_path
    
    
    

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate spherical ROI masks in MNI space from config.yaml"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to the pipeline YAML config (must contain a 'rois' section)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (defaults to paths.simnibs_output/mni_target from config)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    config = load_config(args.config)
    rois: Dict[str, List[float]] = config.get("rois", {})
    if not rois:
        logger.error("No 'rois' section found in config — nothing to generate.")
        return 1

    ref_path = config.get("paths", {}).get("mni_template")
    output_dir = args.output or (
        Path(config["paths"]["simnibs_output"]) / "mni_target"
    )
    radius_mm = config.get("target_generation", {}).get("radius_mm", 10.0)

    AnatomicalPreparer(
        reference_img_path=Path(ref_path) if ref_path else None,
        radius_mm=radius_mm,
    ).run(rois, output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())