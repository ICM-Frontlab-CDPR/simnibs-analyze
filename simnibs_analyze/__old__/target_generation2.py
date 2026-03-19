from pathlib import Path
from typing import List
from nilearn import datasets
from nilearn.image import new_img_like
import nibabel as nib
import numpy as np


def create_sphere_mask(
    template_img: nib.Nifti1Image, mni_coords: List[float], radius_mm: float = 5.0
) -> nib.Nifti1Image:
    """Create a spherical mask around MNI coordinates."""
    affine = template_img.affine
    data = np.zeros(template_img.shape, dtype=np.uint8)

    # Convert MNI coordinates to voxel coordinates
    mni_coords_homogeneous = np.append(mni_coords, 1)
    voxel_coords = np.linalg.inv(affine) @ mni_coords_homogeneous
    voxel_coords = voxel_coords[:3].astype(int)

    # Create sphere
    shape = template_img.shape
    x, y, z = np.ogrid[: shape[0], : shape[1], : shape[2]]
    dist_sq = (
        (x - voxel_coords[0]) ** 2
        + (y - voxel_coords[1]) ** 2
        + (z - voxel_coords[2]) ** 2
    )

    # Convert radius from mm to voxels (approximate using voxel size)
    voxel_size = np.abs(np.diag(affine)[:3]).mean()
    radius_voxels = radius_mm / voxel_size

    data[dist_sq <= radius_voxels**2] = 1

    return new_img_like(template_img, data, affine=affine)


def main() -> None:
    rois = {"fef": [28, -8, 54], "ips_left": [-25, -58, 52], "ips_right": [25, -58, 52]}

    # Load MNI template
    mni_template = datasets.load_mni152_template(resolution=1)

    # Output directory
    output_dir = Path(
        "/Users/hippolyte.dreyfus/Documents/FRONTLAB-SimNIBS-pipeline/templates"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create and save masks for all ROIs
    for roi_name, mni_coords in rois.items():
        mask = create_sphere_mask(mni_template, mni_coords)
        output_path = output_dir / f"{roi_name}_mask.nii.gz"
        nib.save(mask, output_path)

    print(f"3 masks saved to: {output_dir}")


if __name__ == "__main__":
    main()
