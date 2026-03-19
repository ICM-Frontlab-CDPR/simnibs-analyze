import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiSpheresMasker


def create_roi(coords_mni, radius, fpath_reference):
    """
    Crée une ROI sphérique en MNI.

    Parameters
    ----------
    coords_mni : list
        Coordonnées MNI [x, y, z] du centre
    radius : float
        Rayon de la sphère en mm
    fpath_reference : str ou Nifti1Image
        Image de référence pour la géométrie (header, affine)

    Returns
    -------
    Nifti1Image
        Image de la ROI (masque binaire)
    """
    masker = NiftiSpheresMasker(seeds=[coords_mni], radius=radius, standardize=False)

    mask = masker.fit(fpath_reference)
    return mask.mask_img_


def extract_roi_efield(fpath_efield, coords_mni, radius=5.0):
    """
    Extrait les valeurs du champ électrique d'une ROI sphérique.

    Parameters
    ----------
    fpath_efield : str ou Nifti1Image
        Image du champ électrique en espace MNI
    coords_mni : list
        Coordonnées MNI [x, y, z] du centre de la ROI
    radius : float
        Rayon de la sphère en mm (défaut: 5.0)

    Returns
    -------
    dict
        Statistiques de la ROI (moyenne, max, etc.)
    """
    masker = NiftiSpheresMasker(
        seeds=[coords_mni],
        radius=radius,
        standardize=False,
        detrend=False,
    )

    roi_data = masker.fit_transform(fpath_efield)
    roi_vals = roi_data.ravel()

    return {
        "mean": float(np.mean(roi_vals)),
        "max": float(np.max(roi_vals)),
        "std": float(np.std(roi_vals)),
        "n_voxels": len(roi_vals),
    }


def save_roi_as_nifti(coords_mni, radius, fpath_reference, output_path):
    """
    Sauvegarde une ROI sphérique comme image NIfTI.

    Parameters
    ----------
    coords_mni : list
        Coordonnées MNI [x, y, z] du centre
    radius : float
        Rayon en mm
    fpath_reference : str ou Nifti1Image
        Image de référence pour la géométrie (header, affine)
    output_path : str
        Chemin de sortie
    """
    # Créer la ROI
    mask_img = create_roi(coords_mni, radius, fpath_reference)

    # Sauvegarder
    nib.save(mask_img, output_path)
    print(f"ROI sauvegardée: {output_path}")


if __name__ == "__main__":
    # Coordonnées MNI des ROIs
    IPS_LEFT_MNI = [-25, -60, 52]
    IPS_RIGHT_MNI = [25, -60, 52]
    FEF_LEFT_MNI = [-28, -8, 54]
    FEF_RIGHT_MNI = [28, -8, 54]

    # Template MNI pour définir la géométrie de la ROI
    fpath_mni_template = "/Users/hippolyte.dreyfus/Documents/FRONTLAB-SimNIBS-pipeline/templates/MNI152_T1_1mm.nii.gz"

    # Rayon de la ROI en mm
    radius = 5.0

    # Exemple 1: Créer et sauvegarder les ROIs
    save_roi_as_nifti(FEF_RIGHT_MNI, radius, fpath_mni_template, "FEF_right_roi.nii.gz")
    save_roi_as_nifti(FEF_LEFT_MNI, radius, fpath_mni_template, "FEF_left_roi.nii.gz")
    save_roi_as_nifti(IPS_RIGHT_MNI, radius, fpath_mni_template, "IPS_right_roi.nii.gz")
    save_roi_as_nifti(IPS_LEFT_MNI, radius, fpath_mni_template, "IPS_left_roi.nii.gz")

    # Exemple 2: Extraire les valeurs d'un e-field dans une ROI
    # fpath_efield_mni = "/path/to/efield_MNI.nii.gz"  # À adapter
    # stats = extract_roi_efield(fpath_efield_mni, FEF_RIGHT_MNI, radius=5.0)
    # print(f"FEF droit - Moyenne: {stats['mean']:.4f}, Max: {stats['max']:.4f}")
