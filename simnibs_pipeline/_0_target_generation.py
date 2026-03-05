"""
Target generation step — creates spherical ROI masks in MNI space.

Usage (CLI):
    python _0_target_generation.py --config config.yaml --output /path/to/mni_target

Usage (API):
    gen = TargetGenerator(radius_mm=10.0)
    gen.run(rois={"fef": [28, -8, 54]}, output_dir=Path("mni_target"))
    gen.mask_imgs  # dict[str, nib.Nifti1Image]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
from nilearn import datasets
from nilearn.image import new_img_like

from _io import load_config, save_nifti
from _logging import get_logger

logger = get_logger(__name__)


class TargetGenerator:
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

    def run(
        self,
        rois: Dict[str, List[float]],
        output_dir: Path,
    ) -> "TargetGenerator":
        """
        Create and save one spherical mask per ROI.

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
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating {len(rois)} ROI mask(s) → {output_dir}")

        for roi_name, mni_coords in rois.items():
            mask_img = self._create_sphere_mask(self._template, mni_coords, self.radius_mm)
            out_path = output_dir / f"{roi_name}_mask.nii.gz"
            save_nifti(mask_img, out_path)
            self.mask_imgs[roi_name] = mask_img
            logger.info(f"  ✓ {roi_name}: {out_path.name}")

        logger.info(f"{len(self.mask_imgs)} mask(s) saved to {output_dir}")
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
    

    def create_subject_roi_from_mni():
        pass
    

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

    TargetGenerator(
        reference_img_path=Path(ref_path) if ref_path else None,
        radius_mm=radius_mm,
    ).run(rois, output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())