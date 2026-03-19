"""
Target generation step — creates spherical ROI masks in MNI space.

Usage (CLI):
    python _0_anatomical_preparer.py --config config.yaml --output /path/to/mni_target

Usage (API):
    gen = AnatomicalPreparer(radius_mm=10.0)
    gen.setup_mni_rois(
        rois={"fef": {"method": "sphere", "coords": [28, -8, 54]}},
        output_dir=Path("mni_target"),
    )
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
from scipy.ndimage import binary_fill_holes

from .._pipeline_io import (
    save_nifti,
    save_ants_image,
    check_output,
    get_t1_conform,
    get_brainmask,
    get_mni_tissues,
)
from .._logging import get_logger

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
        mni_brain_mask_path: Optional[Path] = None,
    ) -> None:
        self.radius_mm = radius_mm
        self.mask_imgs: Dict[str, nib.Nifti1Image] = {}
        self._mni_brain_mask_path = (
            Path(mni_brain_mask_path) if mni_brain_mask_path else None
        )

        if reference_img_path is not None:
            logger.info(f"Loading reference image: {reference_img_path}")
            self._template = nib.load(str(reference_img_path))
            self._template_path = Path(reference_img_path)
        else:
            logger.info("Loading standard MNI152 1 mm template")
            self._template = datasets.load_mni152_template(resolution=1)
            self._template_path = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup_mni_rois(
        self,
        rois: Dict[str, dict],
        output_dir: Path,
        if_exists: str = "overwrite",
    ) -> "AnatomicalPreparer":
        """
        Create and save one binary mask per ROI in MNI space.

        Subject-independent. Call this once before the subject loop.
        Dispatches to :meth:`_create_sphere_mask` or :meth:`_create_parcel_mask`
        based on the ``method`` key of each ROI definition.

        Parameters
        ----------
        rois : dict
            ``{roi_name: roi_def}`` where ``roi_def`` is::

                # sphere (MNI coordinates)
                {"method": "sphere", "coords": [x, y, z]}

                # atlas parcel
                {"method": "atlas", "atlas": "harvard-oxford",
                 "regions": "Frontal Eye Fields"}  # str or list[str]

        output_dir : Path
            Directory where ``{roi_name}_mask_space-mni.nii.gz`` files will be written.

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

        for roi_name, roi_def in rois.items():
            method = roi_def.method
            if method == "sphere":
                mask_img = self._create_sphere_mask(
                    self._template, roi_def.coords, self.radius_mm
                )
            elif method == "atlas":
                mask_img = self._create_parcel_mask(
                    self._template, roi_def.atlas, roi_def.regions
                )
            else:
                raise ValueError(
                    f"ROI '{roi_name}': unknown method '{method}'. "
                    "Expected 'sphere' or 'atlas'."
                )
            out_path = output_dir / f"{roi_name}_mask_space-mni.nii.gz"
            save_nifti(mask_img, out_path, if_exists=if_exists)
            self.mask_imgs[roi_name] = mask_img
            logger.info(f"  ✓ {roi_name} [{method}]: {out_path.name}")

        logger.info(f"{len(self.mask_imgs)} mask(s) saved to {output_dir}")
        return self

    def run(
        self,
        m2m_dir: Path,
        output_dir: Path,
        if_exists: str = "overwrite",
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
        self.mni_stripped_t1_path: Optional[Path] = None

        # ── Subject-space skull-strip ──────────────────────────────────
        # T1 source : m2m_<subject>/segmentation/T1_bias_corrected.nii.gz
        # Mask      : m2m_<subject>/label_prep/tissue_labeling_upsampled.nii.gz
        # Output    : subject_target/T1_subject_brain.nii.gz
        try:
            t1_path = get_t1_conform(m2m_dir)
            mask_path = get_brainmask(m2m_dir)
            self.stripped_t1_path = self._skull_strip(
                t1_path,
                mask_path,
                out_path=output_dir / "T1_subject_brain.nii.gz",
                if_exists=if_exists,
            )
        except FileNotFoundError as e:
            logger.warning(f"Skull-stripping (subject) skipped — {e}")

        # ── MNI-space skull-strip ───────────────────────────────────────
        # T1 source : templates/MNI152_T1_1mm.nii.gz  (self._template_path)
        # Masque    : m2m_<subject>/toMNI/final_tissues_MNI.nii.gz
        # Output    : subject_target/T1_MNI_brain.nii.gz
        # → same space as the e-fields (*_scalar_MNI_magnE.nii.gz), no resampling.
        try:
            mni_tissues_path = get_mni_tissues(m2m_dir)
            self.mni_stripped_t1_path = self._skull_strip(
                self._template_path,
                mni_tissues_path,
                out_path=output_dir / "T1_MNI_brain.nii.gz",
                if_exists=if_exists,
            )
        except FileNotFoundError as e:
            logger.warning(f"Skull-stripping (MNI) skipped — {e}")

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
        x, y, z = np.ogrid[: shape[0], : shape[1], : shape[2]]
        dist_sq = (x - vox[0]) ** 2 + (y - vox[1]) ** 2 + (z - vox[2]) ** 2
        data[dist_sq <= radius_vox**2] = 1

        return new_img_like(template_img, data, affine=affine)

    @staticmethod
    def _create_parcel_mask(
        template_img: nib.Nifti1Image,
        atlas_name: str,
        region_names: "str | List[str]",
    ) -> nib.Nifti1Image:
        """Return a binary NIfTI mask from one or more atlas parcels.

        Resampled to ``template_img`` space if needed — mirrors
        :meth:`_create_sphere_mask` in signature and return type.

        Parameters
        ----------
        template_img : nib.Nifti1Image
            Reference image that defines the output voxel grid.
        atlas_name : str
            One of ``'harvard-oxford'``, ``'aal'``, ``'destrieux'``.
        region_names : str or list of str
            One or more atlas region labels whose union forms the mask.

        Returns
        -------
        nib.Nifti1Image
            Binary mask in ``template_img`` space.
        """
        import urllib3
        from requests.adapters import HTTPAdapter
        from nilearn import datasets as nl_datasets

        if isinstance(region_names, str):
            region_names = [region_names]

        _FETCHERS = {
            "harvard-oxford": lambda: nl_datasets.fetch_atlas_harvard_oxford(
                "cort-maxprob-thr25-1mm"
            ),
            "aal": lambda: nl_datasets.fetch_atlas_aal(),
            "destrieux": lambda: nl_datasets.fetch_atlas_destrieux_2009(),
        }
        if atlas_name not in _FETCHERS:
            raise ValueError(
                f"Atlas '{atlas_name}' not supported. Available: {list(_FETCHERS)}"
            )

        # Disable SSL verification for atlas downloads (macOS cert issue).
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        _orig_send = HTTPAdapter.send
        HTTPAdapter.send = lambda self, *a, **kw: _orig_send(
            self, *a, **{**kw, "verify": False}
        )
        try:
            atlas_data = _FETCHERS[atlas_name]()
        finally:
            HTTPAdapter.send = _orig_send
        maps = atlas_data.maps
        atlas_img = maps if isinstance(maps, nib.Nifti1Image) else nib.load(maps)
        raw_labels = list(atlas_data.labels)

        # AAL provides a separate `indices` list; other atlases use positional indices.
        if hasattr(atlas_data, "indices"):
            label_map: Dict[str, int] = {
                str(name): int(idx) for name, idx in zip(raw_labels, atlas_data.indices)
            }
        else:
            label_map = {str(name): i for i, name in enumerate(raw_labels)}

        atlas_array = atlas_img.get_fdata()
        mask_data = np.zeros(atlas_array.shape, dtype=np.uint8)

        for region_name in region_names:
            if region_name not in label_map:
                matches = [k for k in label_map if k.lower() == region_name.lower()]
                if not matches:
                    raise ValueError(
                        f"Region '{region_name}' not found in atlas '{atlas_name}'. "
                        f"First available labels: {list(label_map)[:10]}"
                    )
                region_name = matches[0]
            mask_data[np.round(atlas_array).astype(int) == label_map[region_name]] = 1

        mask_img = nib.Nifti1Image(mask_data, atlas_img.affine)

        # Resample to pipeline template if voxel grids differ
        if atlas_img.shape[:3] != template_img.shape[:3] or not np.allclose(
            atlas_img.affine, template_img.affine
        ):
            mask_img = image.resample_to_img(
                mask_img, template_img, interpolation="nearest"
            )

        return mask_img

    def create_subject_roi_from_mni(
        self,
        m2m_dir: Path,
        output_dir: Path,
        if_exists: str = "overwrite",
    ) -> Dict[str, Path]:
        """
        Warp MNI ROI masks to subject space using ANTsPy.

        Parameters
        ----------
        m2m_dir : Path
            Path to the SimNIBS m2m_<subject> directory.
            Must contain toMNI/MNI2Conform_nonl.nii.gz.
        output_dir : Path
            Directory where subject-space ROI masks will be written
            (e.g., subject_target/).

        Returns
        -------
        Dict[str, Path]
            {roi_name: path_to_subject_space_mask}

        Raises
        ------
        FileNotFoundError
            If the warp field or MNI masks are missing.
        """
        import ants

        m2m_dir = Path(m2m_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Warp field (MNI → subject space) ──────────────────────────
        warp_mni2conform = m2m_dir / "toMNI" / "MNI2Conform_nonl.nii.gz"
        if not warp_mni2conform.exists():
            raise FileNotFoundError(
                f"MNI→Subject warp field not found: {warp_mni2conform}"
            )

        # ── Reference image (subject-space T1) ────────────────────────
        t1_subject_path = get_t1_conform(m2m_dir)
        fixed = ants.image_read(str(t1_subject_path))

        # Load MNI masks from disk in all cases to keep behavior deterministic.
        simnibs_output_dir = m2m_dir.parent.parent
        mni_output_dir = getattr(
            self, "mni_output_dir", simnibs_output_dir / "mni_target"
        )
        self.mni_output_dir = mni_output_dir
        mni_mask_paths = sorted(mni_output_dir.glob("*_mask_space-mni.nii.gz"))
        if not mni_mask_paths:
            raise FileNotFoundError(f"No MNI ROI masks found in: {mni_output_dir}")
        self.mask_imgs = {}
        for p in mni_mask_paths:
            roi_name = p.name.replace("_mask_space-mni.nii.gz", "")
            self.mask_imgs[roi_name] = nib.load(str(p))

        subject_roi_paths: Dict[str, Path] = {}

        # ── Warp each MNI ROI mask to subject space ────────────────────
        for roi_name, mni_mask_img in self.mask_imgs.items():
            logger.info(f"Warping {roi_name} from MNI to subject space...")

            # Ensure MNI mask is saved on disk (ANTsPy reads from file)
            mni_mask_path = self.mni_output_dir / f"{roi_name}_mask_space-mni.nii.gz"
            if not mni_mask_path.exists():
                save_nifti(mni_mask_img, mni_mask_path)

            moving = ants.image_read(str(mni_mask_path))

            warped = ants.apply_transforms(
                fixed=fixed,
                moving=moving,
                transformlist=[str(warp_mni2conform)],
                interpolator="nearestNeighbor",  # binary mask
            )

            subject_mask_path = output_dir / f"{roi_name}_mask_space-native.nii.gz"
            if check_output(subject_mask_path, if_exists):
                save_ants_image(warped, subject_mask_path)
                logger.info(f"  ✓ {roi_name} warped → {subject_mask_path.name}")
            else:
                logger.info(f"  skip {roi_name} (exists): {subject_mask_path.name}")
            subject_roi_paths[roi_name] = subject_mask_path

        return subject_roi_paths

    @staticmethod
    def _skull_strip(
        t1_path: Path,
        mask_path: Path,
        out_path: Optional[Path] = None,
        if_exists: str = "overwrite",
    ) -> Path:
        """
        Apply a brain mask to a T1 image and save the result.

        Works for both subject-space and MNI-space skull-stripping — the
        only difference is which T1 and mask files are passed.

        Parameters
        ----------
        t1_path : Path
            Path to the T1 NIfTI image (or a loaded template path).
        mask_path : Path
            Path to the tissue-labeling NIfTI (labels 1=WM, 2=GM …).
        out_path : Path or None
            Explicit output path.  If None, the result is saved alongside
            the T1 as ``<T1stem>_brain.nii.gz``.

        Returns
        -------
        Path
            Path to the saved skull-stripped image.
        """
        t1_path = Path(t1_path)
        mask_path = Path(mask_path)

        t1_img = nib.load(str(t1_path))
        mask_img = nib.load(str(mask_path))

        # Resample mask to T1 space if needed
        if (
            not np.allclose(mask_img.affine, t1_img.affine)
            or mask_img.shape != t1_img.shape
        ):
            mask_img = image.resample_to_img(mask_img, t1_img, interpolation="nearest")

        mask_raw = np.asarray(mask_img.dataobj)
        mask_labels = np.rint(np.squeeze(mask_raw)).astype(np.int16)

        # SimNIBS tissue labeling convention: 1=WM, 2=GM, 3=CSF.
        # Use only WM+GM (1, 2) — excluding CSF removes the subarachnoid
        # halo around the cortex. binary_fill_holes restores the ventricles.
        if np.all(np.isin(np.unique(mask_labels), [0, 1])):
            brain_vox = mask_labels > 0
        elif np.any(np.isin(mask_labels, [1, 2])):
            brain_vox = np.isin(mask_labels, [1, 2])
        else:
            logger.warning(
                "Brain mask: no WM/GM labels (1/2) found; falling back to >0."
            )
            brain_vox = mask_labels > 0

        mask_data = binary_fill_holes(brain_vox).astype(t1_img.get_data_dtype())

        stripped_data = np.squeeze(np.asarray(t1_img.dataobj)) * mask_data
        stripped_img = nib.Nifti1Image(stripped_data, t1_img.affine, t1_img.header)

        if out_path is None:
            stem = t1_path.name.replace(".nii.gz", "").replace(".nii", "")
            out_path = t1_path.parent / f"{stem}_brain.nii.gz"
        out_path = Path(out_path)
        save_nifti(stripped_img, out_path, if_exists=if_exists)
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

    from .._config import load_and_validate

    cfg = load_and_validate(args.config)
    rois = cfg.target_generation.rois
    if not rois:
        logger.error(
            "No 'target_generation.rois' section found in config — nothing to generate."
        )
        return 1

    output_dir = args.output or (cfg.paths.simnibs_output / "mni_target")

    AnatomicalPreparer(
        reference_img_path=cfg.paths.mni_template,
        radius_mm=cfg.target_generation.radius_mm,
    ).setup_mni_rois(rois, output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
