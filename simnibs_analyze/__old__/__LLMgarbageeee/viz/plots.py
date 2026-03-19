"""
Fonctions atomiques de plotting - niveau le plus bas.
Chaque fonction trace sur un axes matplotlib fourni.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import nibabel as nib
from matplotlib.axes import Axes
import matplotlib.pyplot as plt


# ============================================================================
# UTILITAIRES - Chargement et extraction de données
# ============================================================================


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


def extract_slice(
    data: np.ndarray, axis: int, slice_idx: Optional[int] = None
) -> np.ndarray:
    """
    Extrait une coupe 2D d'un volume 3D.

    Parameters
    ----------
    data : np.ndarray
        Volume 3D
    axis : int
        Axe de coupe (0=sagittal, 1=coronal, 2=axial)
    slice_idx : int, optional
        Index de la coupe (si None, prend la coupe centrale)

    Returns
    -------
    slice_2d : np.ndarray
        Coupe 2D
    """
    # Squeeze pour enlever les dimensions singleton
    data = np.squeeze(data)

    # Vérifier que c'est bien 3D maintenant
    if data.ndim != 3:
        raise ValueError(f"Data must be 3D after squeezing, got shape {data.shape}")

    if slice_idx is None:
        slice_idx = data.shape[axis] // 2

    if axis == 0:
        return data[slice_idx, :, :]
    elif axis == 1:
        return data[:, slice_idx, :]
    else:
        return data[:, :, slice_idx]


# ============================================================================
# PLOTS ATOMIQUES - Prennent un axes matplotlib
# ============================================================================


def plot_segmentation_overlay(
    ax: Axes, t1_slice: np.ndarray, seg_slice: np.ndarray, title: Optional[str] = None
) -> Axes:
    """
    Plot une coupe de segmentation overlay sur T1.

    Parameters
    ----------
    ax : Axes
        Axes matplotlib
    t1_slice : np.ndarray
        Coupe 2D du T1
    seg_slice : np.ndarray
        Coupe 2D de la segmentation
    title : str, optional
        Titre du plot

    Returns
    -------
    ax : Axes
        Axes modifié
    """
    # Afficher T1
    ax.imshow(t1_slice.T, cmap="gray", origin="lower")

    # Overlay de la segmentation
    seg_masked = np.ma.masked_where(seg_slice == 0, seg_slice)
    ax.imshow(seg_masked.T, cmap="jet", alpha=0.3, origin="lower")

    if title:
        ax.set_title(title)
    ax.axis("off")

    return ax


def plot_efield_slice(
    ax: Axes,
    efield_slice: np.ndarray,
    title: Optional[str] = None,
    vmin: float = 0,
    vmax: Optional[float] = None,
    cmap: str = "hot",
    colorbar: bool = True,
) -> Axes:
    """
    Plot une coupe d'e-field.

    Parameters
    ----------
    ax : Axes
        Axes matplotlib
    efield_slice : np.ndarray
        Coupe 2D de l'e-field
    title : str, optional
        Titre du plot
    vmin : float
        Valeur minimale de l'échelle
    vmax : float, optional
        Valeur maximale de l'échelle
    cmap : str
        Colormap
    colorbar : bool
        Afficher la colorbar

    Returns
    -------
    ax : Axes
        Axes modifié
    """
    if vmax is None:
        vmax = np.nanmax(efield_slice)

    im = ax.imshow(efield_slice.T, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")

    if title:
        ax.set_title(title)
    ax.axis("off")

    if colorbar:
        plt.colorbar(im, ax=ax, label="E-field (V/m)")

    return ax


def plot_efield_difference(
    ax: Axes, diff_slice: np.ndarray, title: Optional[str] = None, colorbar: bool = True
) -> Axes:
    """
    Plot une coupe de différence d'e-field.

    Parameters
    ----------
    ax : Axes
        Axes matplotlib
    diff_slice : np.ndarray
        Coupe 2D de la différence
    title : str, optional
        Titre du plot
    colorbar : bool
        Afficher la colorbar

    Returns
    -------
    ax : Axes
        Axes modifié
    """
    vmax_diff = max(abs(np.nanmin(diff_slice)), abs(np.nanmax(diff_slice)))
    im = ax.imshow(
        diff_slice.T, cmap="RdBu_r", vmin=-vmax_diff, vmax=vmax_diff, origin="lower"
    )

    if title:
        ax.set_title(title)
    ax.axis("off")

    if colorbar:
        plt.colorbar(im, ax=ax, label="ΔE-field (V/m)")

    return ax


def plot_roi_overlay(
    ax: Axes,
    roi_slice: np.ndarray,
    background_slice: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    roi_cmap: str = "Reds",
    roi_alpha: float = 0.6,
) -> Axes:
    """
    Plot une coupe de ROI avec optionnellement un background.

    Parameters
    ----------
    ax : Axes
        Axes matplotlib
    roi_slice : np.ndarray
        Coupe 2D de la ROI
    background_slice : np.ndarray, optional
        Coupe 2D du template/background
    title : str, optional
        Titre du plot
    roi_cmap : str
        Colormap pour la ROI
    roi_alpha : float
        Transparence de la ROI

    Returns
    -------
    ax : Axes
        Axes modifié
    """
    # Background si fourni
    if background_slice is not None:
        ax.imshow(background_slice.T, cmap="gray", origin="lower", alpha=0.7)

    # ROI en overlay
    roi_masked = np.ma.masked_where(roi_slice == 0, roi_slice)
    ax.imshow(roi_masked.T, cmap=roi_cmap, origin="lower", alpha=roi_alpha)

    if title:
        ax.set_title(title)
    ax.axis("off")

    return ax


def plot_histogram(
    ax: Axes,
    values: np.ndarray,
    title: Optional[str] = None,
    bins: int = 50,
    color: str = "blue",
    show_stats: bool = True,
    xlabel: str = "E-field (V/m)",
    ylabel: str = "Frequency",
) -> Axes:
    """
    Plot un histogramme avec statistiques.

    Parameters
    ----------
    ax : Axes
        Axes matplotlib
    values : np.ndarray
        Valeurs à afficher
    title : str, optional
        Titre du plot
    bins : int
        Nombre de bins
    color : str
        Couleur de l'histogramme
    show_stats : bool
        Afficher les lignes de moyenne/médiane
    xlabel : str
        Label de l'axe x
    ylabel : str
        Label de l'axe y

    Returns
    -------
    ax : Axes
        Axes modifié
    """
    # Histogramme
    ax.hist(values, bins=bins, alpha=0.7, color=color, edgecolor="black")

    # Statistiques
    if show_stats:
        mean_val = np.mean(values)
        median_val = np.median(values)
        ax.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.3f}",
        )
        ax.axvline(
            median_val,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_val:.3f}",
        )
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    ax.grid(alpha=0.3)

    return ax


def plot_boxplot_comparison(
    ax: Axes,
    data_dict: dict,
    title: Optional[str] = None,
    ylabel: str = "E-field (V/m)",
) -> Axes:
    """
    Plot un boxplot comparatif.

    Parameters
    ----------
    ax : Axes
        Axes matplotlib
    data_dict : dict
        Dictionnaire {label: values}
    title : str, optional
        Titre du plot
    ylabel : str
        Label de l'axe y

    Returns
    -------
    ax : Axes
        Axes modifié
    """
    labels = list(data_dict.keys())
    data = list(data_dict.values())

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
        medianprops=dict(color="red", linewidth=2),
    )

    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    ax.grid(alpha=0.3, axis="y")

    return ax


def plot_paired_data(
    ax: Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_label: str,
    y_label: str,
    title: Optional[str] = None,
    show_mean: bool = True,
) -> Axes:
    """
    Plot des données paired (lignes connectant deux conditions).

    Parameters
    ----------
    ax : Axes
        Axes matplotlib
    x_values : np.ndarray
        Valeurs condition 1
    y_values : np.ndarray
        Valeurs condition 2
    x_label : str
        Label condition 1
    y_label : str
        Label condition 2
    title : str, optional
        Titre du plot
    show_mean : bool
        Afficher la moyenne

    Returns
    -------
    ax : Axes
        Axes modifié
    """
    # Lignes individuelles
    for i in range(len(x_values)):
        ax.plot([0, 1], [x_values[i], y_values[i]], "o-", alpha=0.5, color="gray")

    # Moyennes
    if show_mean:
        ax.plot(
            [0, 1],
            [np.mean(x_values), np.mean(y_values)],
            "o-",
            linewidth=3,
            markersize=10,
            color="red",
            label="Mean",
        )
        ax.legend()

    ax.set_xticks([0, 1])
    ax.set_xticklabels([x_label, y_label])
    ax.set_ylabel("E-field (V/m)")

    if title:
        ax.set_title(title)

    ax.grid(alpha=0.3, axis="y")

    return ax
