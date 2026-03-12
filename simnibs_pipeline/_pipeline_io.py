"""
Module I/O du pipeline SimNIBS.
Centralise toutes les opérations d'entrées/sorties : recherche de fichiers,
chargement d'images NIfTI, lecture/écriture de CSV et de configuration YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any, Iterable, Tuple, Union
import numpy as np
import nibabel as nib
import pandas as pd
import yaml

from _logging import get_logger

logger = get_logger(__name__)

SPACE_MNI = "mni"
SPACE_NATIVE = "native"


def normalize_space(space: str) -> str:
    """Normalize and validate the working space value."""
    normalized = str(space).lower().strip()
    if normalized not in {SPACE_MNI, SPACE_NATIVE}:
        raise ValueError(
            f"Paramètre 'space' invalide: {space}. Valeurs autorisées: {SPACE_MNI}, {SPACE_NATIVE}"
        )
    return normalized


def space_tag(space: str) -> str:
    """Return tagged space label used in output paths and file names."""
    return f"space-{normalize_space(space)}"


def get_subject_paths(simnibs_output_dir: Path, subject: str) -> Dict[str, Path]:
    """Return canonical subject-level paths used across the pipeline."""
    subject_dir = Path(simnibs_output_dir) / subject
    return {
        "subject_dir": subject_dir,
        "m2m_dir": subject_dir / f"m2m_{subject}",
        "subject_target_dir": subject_dir / "subject_target",
    }


def get_analysis_dir(results_dir: Path, space: str) -> Path:
    """Return the shared analysis output directory (space encoded in filenames)."""
    _ = normalize_space(space)
    return Path(results_dir) / "analysis"


def get_features_csv_path(results_dir: Path, space: str) -> Path:
    """Return the canonical features CSV path for a given space."""
    return get_analysis_dir(results_dir, space) / f"all_features_{space_tag(space)}.csv"


def get_inter_subject_summary_csv_path(results_dir: Path, space: str) -> Path:
    """Return the inter-subject summary CSV path for a given space."""
    return get_analysis_dir(results_dir, space) / f"inter_subject_summary_{space_tag(space)}.csv"


def get_intra_subject_diff_csv_path(results_dir: Path, space: str, condition: str) -> Path:
    """Return the intra-subject diff CSV path for a condition and space."""
    return get_analysis_dir(results_dir, space) / f"intra_subject_diff_{condition}_{space_tag(space)}.csv"


def get_clusters_csv_path(results_dir: Path, space: str) -> Path:
    """Return the clustering CSV path for a given space."""
    return get_analysis_dir(results_dir, space) / f"clusters_{space_tag(space)}.csv"


def load_config(config_path: Path) -> Dict:
    """Charge le fichier de configuration YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_raw_efield(
    simnibs_output_dir: Path,
    subject: str,
    roi: str,
    mode: str
) -> Optional[Path]:
    """
    Trouve le fichier e-field brut (non préprocessé) dans la sortie SimNIBS.

    Parameters
    ----------
    simnibs_output_dir : Path
        Répertoire de sortie SimNIBS
    subject : str
        ID du sujet
    roi : str
        Nom de la ROI
    mode : str
        Mode (simulation ou optimization)

    Returns
    -------
    Path or None
        Chemin vers le fichier ou None si non trouvé
    """
    subject_dir = simnibs_output_dir / subject

    if not subject_dir.exists():
        return None

    if mode == "simulation":
        base_dir = subject_dir / "simulations"
    else:
        base_dir = subject_dir / "optimizations"

    if not base_dir.exists():
        return None

    pattern = f"{mode}_{mode}_{roi}_*"
    matching_dirs = list(base_dir.glob(pattern))

    if not matching_dirs:
        return None

    mode_dir = matching_dirs[0]

    if mode == "optimization":
        mni_volumes_dir = mode_dir / "simulation_with_optimal_montage" / "mni_volumes"
    else:
        mni_volumes_dir = mode_dir / "mni_volumes"

    if not mni_volumes_dir.exists():
        return None

    efield_files = list(mni_volumes_dir.glob("*_scalar_MNI_magnE.nii.gz"))
    return efield_files[0] if efield_files else None


def find_simulation_dirs(subject_dir: Path, condition: str, mode: str) -> List[Path]:
    """
    Trouve tous les répertoires de simulation/optimization pour une condition donnée.
    Gère les hashes dans les noms de dossiers.

    Parameters
    ----------
    subject_dir : Path
        Répertoire du sujet (ex: 001-CC)
    condition : str
        Condition de stimulation (ex: fef, ips_left, ips_right)
    mode : str
        Mode (simulation ou optimization)

    Returns
    -------
    List[Path]
        Liste des répertoires trouvés
    """
    pattern = f"{mode}_{mode}_{condition}_*"

    if mode == "simulation":
        base_dir = subject_dir / "simulations"
    else:
        base_dir = subject_dir / "optimizations"

    if not base_dir.exists():
        logger.warning(f"Répertoire {mode} non trouvé: {base_dir}")
        return []

    found_dirs = list(base_dir.glob(pattern))

    if not found_dirs:
        logger.warning(f"Aucune {mode} trouvée pour pattern: {pattern} dans {base_dir}")

    return found_dirs


def find_efield_files(simulation_dir: Path, mode: str, space: str = SPACE_MNI) -> List[Path]:
    """
    Trouve les fichiers e-field dans le répertoire de simulation/optimization.

    Parameters
    ----------
    simulation_dir : Path
        Répertoire de la simulation ou optimization
    mode : str
        Mode (simulation ou optimization)
    space : str
        ``'mni'`` (défaut) : fichiers ``*_scalar_MNI_magnE.nii.gz`` dans ``mni_volumes/``.
        ``'native'`` : fichiers ``*_scalar_magnE.nii.gz`` dans ``subject_volumes/``.

    Returns
    -------
    List[Path]
        Liste des fichiers e-field trouvés
    """
    space = normalize_space(space)

    if space == SPACE_NATIVE:
        if mode == "optimization":
            volumes_dir = simulation_dir / "simulation_with_optimal_montage" / "subject_volumes"
        else:
            volumes_dir = simulation_dir / "subject_volumes"
        glob_pattern = "*_scalar_magnE.nii.gz"
    else:
        if mode == "optimization":
            volumes_dir = simulation_dir / "simulation_with_optimal_montage" / "mni_volumes"
        else:
            volumes_dir = simulation_dir / "mni_volumes"
        glob_pattern = "*_scalar_MNI_magnE.nii.gz"

    if not volumes_dir.exists():
        logger.warning(f"Répertoire {space}_volumes non trouvé: {volumes_dir}")
        return []

    efield_files = list(volumes_dir.glob(glob_pattern))

    if not efield_files:
        logger.warning(f"Aucun fichier e-field trouvé dans {volumes_dir}")

    return efield_files


def get_t1_conform(
    m2m_dir: Path,
    filename: str = "segmentation/T1_bias_corrected.nii.gz",
) -> Path:
    """
    Retourne le chemin du T1 dans ``m2m_dir``.

    Parameters
    ----------
    m2m_dir : Path
        Répertoire ``m2m_<subject>`` produit par SimNIBS.
    filename : str
        Chemin relatif du fichier T1 (default: ``segmentation/T1_bias_corrected.nii.gz``).

    Raises
    ------
    FileNotFoundError
    """
    path = Path(m2m_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"T1 non trouvé : {path}")
    return path


def get_brainmask(
    m2m_dir: Optional[Path] = None,
    filename: str = "label_prep/tissue_labeling_upsampled.nii.gz",
    space: str = SPACE_NATIVE,
    mni_mask_path: Optional[Path] = None,
) -> Path:
    """
    Retourne le chemin du masque cerveau.

    Parameters
    ----------
    m2m_dir : Path or None
        Répertoire ``m2m_<subject>`` produit par SimNIBS. Ignoré si ``space='mni'``.
    filename : str
        Chemin relatif du fichier masque dans ``m2m_dir`` (espace sujet uniquement).
    space : str
        ``'native'`` (défaut) : masque dans ``m2m_dir``.
        ``'mni'`` : masque MNI passé via ``mni_mask_path`` (lu depuis config).
    mni_mask_path : Path or None
        Chemin du masque MNI, requis si ``space='mni'``.
        Doit provenir de ``config['paths']['mni_brain_mask']``.

    Raises
    ------
    FileNotFoundError, ValueError
    """
    space = normalize_space(space)

    if space == SPACE_MNI:
        if mni_mask_path is None:
            raise ValueError("mni_mask_path est requis pour space='mni' (config['paths']['mni_brain_mask'])")
        path = Path(mni_mask_path)
    else:
        if m2m_dir is None:
            raise ValueError("m2m_dir est requis pour space='native'")
        path = Path(m2m_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Masque cerveau non trouvé : {path}")
    return path


def get_mni_tissues(m2m_dir: Path) -> Path:
    """
    Retourne le chemin de la segmentation tissulaire en espace MNI.

    Produit par SimNIBS dans ``toMNI/final_tissues_MNI.nii.gz``.
    Labels : 1=WM, 2=GM, 3=CSF, 4=Bone, 5=Scalp …
    """
    path = Path(m2m_dir) / "toMNI" / "final_tissues_MNI.nii.gz"
    if not path.exists():
        raise FileNotFoundError(f"Tissues MNI non trouvés : {path}")
    return path


def get_roi_mask_path(
    simnibs_output_dir: Path,
    condition: str,
    space: str = SPACE_MNI,
    subject: Optional[str] = None,
) -> Path:
    """
    Récupère le chemin du masque ROI pour une condition donnée.

    Parameters
    ----------
    simnibs_output_dir : Path
        Répertoire de sortie SimNIBS
    condition : str
        Condition de stimulation
    space : str
        ``'mni'`` (défaut) : masques dans ``mni_target/``.
        ``'native'`` : masques sujets dans ``<subject>/subject_target/``.
    subject : str or None
        ID sujet requis si ``space='native'``.

    Returns
    -------
    Path
        Chemin du masque ROI
    
    Raises
    ------
    FileNotFoundError
        Si le masque demandé n'existe pas.
    """
    space = normalize_space(space)

    if space == SPACE_NATIVE:
        if not subject:
            raise ValueError("subject est requis pour space='native'")
        mask_path = (
            simnibs_output_dir
            / subject
            / "subject_target"
            / f"{condition}_mask_{space_tag(space)}.nii.gz"
        )
    else:
        mask_path = simnibs_output_dir / "mni_target" / f"{condition}_mask_{space_tag(space)}.nii.gz"

    if not mask_path.exists():
        raise FileNotFoundError(f"Masque ROI non trouvé ({space} space): {mask_path}")

    return mask_path


def get_preproc_dir(sim_dir: Path, mode: str, space: str = SPACE_MNI) -> Path:
    """
    Retourne le dossier de preprocessing pour un répertoire de simulation donné.

    Parameters
    ----------
    sim_dir : Path
        Répertoire de simulation SimNIBS.
    mode : str
        ``'simulation'`` ou ``'optimization'``.
    space : str
        ``'mni'`` (défaut) ou ``'native'``.
    """
    space = normalize_space(space)
    volumes_dir = "subject_volumes" if space == SPACE_NATIVE else "mni_volumes"
    if mode == "optimization":
        return sim_dir / "simulation_with_optimal_montage" / volumes_dir
    return sim_dir / volumes_dir


def get_preproc_paths(preproc_dir: Path, base_name: str) -> dict:
    """
    Retourne les chemins des fichiers preprocessés intra et extra-ROI.

    Parameters
    ----------
    preproc_dir : Path
        Dossier de sortie du preprocessing (ex: mni_volumes/).
    base_name : str
        Nom de base du fichier e-field (sans extension).

    Returns
    -------
    dict avec les clés :
        ``intra_masked``, ``intra_cleaned``,
        ``extra_masked``,  ``extra_cleaned``
    """
    return {
        "intra_masked":  preproc_dir / f"{base_name}_roi_masked.nii.gz",
        "intra_cleaned": preproc_dir / f"{base_name}_roi_cleaned.nii.gz",
        "extra_masked":  preproc_dir / f"{base_name}_extra_roi_masked.nii.gz",
        "extra_cleaned": preproc_dir / f"{base_name}_extra_roi_cleaned.nii.gz",
    }


def load_nifti(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    Charge un fichier NIfTI.

    Parameters
    ----------
    path : Path
        Chemin vers le fichier NIfTI

    Returns
    -------
    data : np.ndarray
        Données du volume
    img : nib.Nifti1Image
        Image NIfTI complète
    """
    img = nib.load(str(path))
    data = img.get_fdata()
    return data, img


def load_img(
    path_or_img: Union[str, Path, nib.spatialimages.SpatialImage],
) -> nib.spatialimages.SpatialImage:
    """Load a NIfTI image from a path or return the image if already loaded."""
    if isinstance(path_or_img, (str, Path)):
        return nib.load(str(path_or_img))
    if isinstance(path_or_img, nib.spatialimages.SpatialImage):
        return path_or_img
    raise TypeError(f"Expected path or nibabel image, got {type(path_or_img)}")


def validate_binary(data: np.ndarray, name: str = "mask") -> None:
    """Raise ValueError if *data* contains values other than 0 and 1."""
    unique_values = np.unique(data)
    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(
            f"{name} must be binary (contain only 0 and 1), "
            f"but contains values: {unique_values}"
        )


def save_nifti(img: nib.spatialimages.SpatialImage, output_path: Path) -> None:
    """Save a NIfTI image to disk, creating parent directories as needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.to_filename(str(output_path))


def load_csvs(csv_paths: Iterable[Path]) -> pd.DataFrame:
    """Load and concatenate multiple CSV files into a single DataFrame.

    Parameters
    ----------
    csv_paths : Iterable[Path]
        Iterable of paths to CSV files

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame from all CSV files
    """
    frames = [pd.read_csv(p) for p in csv_paths]
    return pd.concat(frames, ignore_index=True)


def save_rows(rows: List[Dict], out_csv: Path) -> None:
    """Save a list of row dicts to a CSV file, creating parent directories as needed.

    Parameters
    ----------
    rows :
        List of dicts, each representing one row (e.g. feature extraction output).
    out_csv :
        Destination CSV path.
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
