from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pyvista as pv
from nilearn import plotting, image as nl_image

from _logging import get_logger

logger = get_logger(__name__)


class Visualizer:
    """
    Generates all pipeline visualisations.

    Rendering parameters (colormap, thresholds, etc.) are set once at
    construction time and reused across all methods.

    Parameters
    ----------
    output_dir :
        Base output directory.  Each method writes into a named sub-directory
        (``simu/``, ``preprocess/``, ``targets/``, ``analysis/``).
    cmap :
        Matplotlib / PyVista colormap for e-field figures.
    threshold_percentile :
        Non-zero voxel percentile below which values are zeroed in 3D renders.
    bins :
        Number of histogram bins for preprocessing histograms.
    camera_position :
        PyVista camera position string (``'xy'``, ``'xz'``, ``'yz'``).
    """

    def __init__(
        self,
        output_dir: Path,
        cmap: str = "hot",
        threshold_percentile: float = 50.0,
        bins: int = 50,
        camera_position: str = "xy",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.cmap = cmap
        self.threshold_percentile = threshold_percentile
        self.bins = bins
        self.camera_position = camera_position

    # ------------------------------------------------------------------
    # Private rendering helper
    # ------------------------------------------------------------------

    @staticmethod
    def _create_3d_view(
        efield_path: Path,
        camera_position: str = "xy",
        cmap: str = "hot",
        threshold_percentile: float = 0.0,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        skull_stripped_t1_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Render an e-field NIfTI volume with PyVista (offscreen) and return an RGBA array.

        When ``skull_stripped_t1_path`` is provided, the e-field is first resampled
        into the skull-stripped T1 space (colocalization) before PyVista rendering,
        giving an anatomically-aligned view.

        Parameters
        ----------
        efield_path :
            Path to the e-field NIfTI file.
        camera_position :
            PyVista camera position string (``'xy'``, ``'xz'``, ``'yz'``).
        cmap :
            Colormap name.
        threshold_percentile :
            Voxels below this percentile of non-zero values are zeroed.
        vmin, vmax :
            Explicit colour scale limits. If both ``None``, PyVista auto-scales.
        skull_stripped_t1_path :
            Optional path to the skull-stripped T1 (e.g. ``T1fs_nu_conform_brain.nii.gz``).
            When set, the e-field is resampled into T1 space before rendering.
        """
        efield_img = nib.load(str(efield_path))

        if skull_stripped_t1_path is not None:
            skull_stripped_t1_img = nib.load(str(skull_stripped_t1_path))
            efield_img = nl_image.resample_to_img(efield_img, skull_stripped_t1_img, interpolation="continuous")

        data = np.squeeze(efield_img.get_fdata())
        if threshold_percentile > 0:
            nonzero = data[data > 0]
            if nonzero.size > 0:
                thresh = np.percentile(nonzero, threshold_percentile)
                data[data < thresh] = 0

        efield_affine = efield_img.affine
        efield_spacing = efield_img.header.get_zooms()[:3]
        efield_origin = efield_affine[:3, 3]

        grid = pv.ImageData()
        grid.dimensions = np.array(data.shape) + 1
        grid.spacing = efield_spacing
        grid.origin = efield_origin
        grid.cell_data["values"] = data.flatten(order="F")

        plotter = pv.Plotter(off_screen=True)
        if vmin is not None and vmax is not None:
            plotter.add_volume(grid, cmap=cmap, clim=[vmin, vmax])
        else:
            plotter.add_volume(grid, cmap=cmap)
        plotter.camera_position = camera_position
        image = plotter.screenshot(return_img=True)
        plotter.close()
        return image

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------
    def efields_figures(
        self,
        file_info_by_roi_mode: Dict[Tuple[str, str], List[Tuple[str, Path]]],
        t1_brain_by_subject: Optional[Dict[str, Path]] = None,
    ) -> None:
        """
        Generate one 3D figure per (roi, mode) pair.

        Parameters
        ----------
        file_info_by_roi_mode :
            Mapping ``(roi, mode) → [(subject, efield_path), ...]``.
        t1_brain_by_subject :
            Optional mapping ``subject → skull_stripped_t1_path``. When provided,
            the e-field is resampled into T1 space before PyVista rendering,
            giving an anatomically-aligned view.  When ``None``, the e-field
            is rendered directly in its original MNI space.
        """
        output_dir = self.output_dir / "simu"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Global colour scale across all files
        all_nonzero: List[np.ndarray] = []
        for subject_files in file_info_by_roi_mode.values():
            for _, efield_path in subject_files:
                data = np.squeeze(nib.load(str(efield_path)).get_fdata())
                nonzero = data[data > 0]
                if nonzero.size > 0:
                    if self.threshold_percentile > 0:
                        thresh = np.percentile(nonzero, self.threshold_percentile)
                        nonzero = nonzero[nonzero >= thresh]
                    all_nonzero.append(nonzero)

        if not all_nonzero:
            logger.warning("Aucune donnée non-nulle trouvée, abandon de efields_figures.")
            return

        all_values = np.concatenate(all_nonzero)
        vmin, vmax = float(np.min(all_values)), float(np.max(all_values))
        logger.info(f"Échelle de couleur globale : {vmin:.3f} – {vmax:.3f} V/m")

        for (roi, mode), subject_files in file_info_by_roi_mode.items():
            logger.info(f"Génération de la figure : {roi} – {mode}")
            n_subjects = len(subject_files)
            n_cols = min(6, n_subjects)
            n_rows = (n_subjects + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            if n_subjects == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)

            for idx, (subject, efield_path) in enumerate(subject_files):
                row, col = divmod(idx, n_cols)
                ax = axes[row, col]
                t1_path = (t1_brain_by_subject or {}).get(subject)
                image = self._create_3d_view(
                    efield_path=efield_path,
                    camera_position=self.camera_position,
                    cmap=self.cmap,
                    threshold_percentile=self.threshold_percentile,
                    vmin=vmin,
                    vmax=vmax,
                    skull_stripped_t1_path=t1_path,
                )
                ax.imshow(image)
                ax.axis("off")
                ax.set_title(subject, fontsize=12)

            for idx in range(n_subjects, n_rows * n_cols):
                row, col = divmod(idx, n_cols)
                axes[row, col].axis("off")

            fig.suptitle(f"{roi} – {mode}", fontsize=16, fontweight="bold")
            plt.tight_layout()
            out_path = output_dir / f"efields_3d_{roi}_{mode}_{self.camera_position}.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"  Sauvegardé : {out_path}")

        logger.info(f"Toutes les figures 3D dans {output_dir}")

    def efields_histograms(
        self,
        data_by_subject: Dict[str, List[Tuple[str, str, Path, Path]]],
    ) -> None:
        """
        Generate one histogram figure per subject comparing masked vs cleaned e-fields.

        Parameters
        ----------
        data_by_subject :
            Mapping ``subject → [(roi, mode, masked_path, cleaned_path), ...]``.
            Resolve file paths in the caller using
            :func:`file_io.find_simulation_dirs`.
        """
        output_dir = self.output_dir / "preprocess"
        output_dir.mkdir(parents=True, exist_ok=True)

        for subject, subject_data in data_by_subject.items():
            logger.info(f"Histogrammes pour {subject}")
            n_plots = len(subject_data)
            n_cols = min(3, n_plots)
            n_rows = (n_plots + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_plots == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)

            for idx, (roi, mode, masked_path, cleaned_path) in enumerate(subject_data):
                row, col = divmod(idx, n_cols)

                masked_data = nib.load(str(masked_path)).get_fdata().ravel()
                masked_data = masked_data[masked_data > 0]
                cleaned_data = nib.load(str(cleaned_path)).get_fdata().ravel()
                cleaned_data = cleaned_data[cleaned_data > 0]

                ax = axes[row, col]
                ax.hist(masked_data, bins=self.bins, alpha=0.6, label="Avant", color="red", density=True)
                ax.hist(cleaned_data, bins=self.bins, alpha=0.6, label="Après", color="blue", density=True)
                ax.set_xlabel("E-field (V/m)", fontsize=10)
                ax.set_ylabel("Densité", fontsize=10)
                ax.set_title(f"{roi}\n{mode}", fontsize=11)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            for idx in range(n_plots, n_rows * n_cols):
                row, col = divmod(idx, n_cols)
                axes[row, col].axis("off")

            fig.suptitle(
                f"Histogrammes E-field preprocessing – {subject}",
                fontsize=16,
                fontweight="bold",
            )
            plt.tight_layout()
            out_path = output_dir / f"efields_histograms_{subject}.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"  Sauvegardé : {out_path}")

        logger.info(f"Tous les histogrammes dans {output_dir}")

    def visualize_roi_masks(
        self,
        mask_imgs: List[nib.nifti1.Nifti1Image],
        roi_names: List[str],
        mni_template: nib.nifti1.Nifti1Image,
    ) -> None:
        """
        Visualise ROI masks in MNI space using nilearn.

        Parameters
        ----------
        mask_imgs :
            Binary ROI mask images (already loaded).
        roi_names :
            Names corresponding to each mask — used for titles and filenames.
        mni_template :
            MNI background image (already loaded).
        """
        output_dir = self.output_dir / "targets"
        output_dir.mkdir(parents=True, exist_ok=True)

        for mask_img, roi_name in zip(mask_imgs, roi_names):
            logger.info(f"Visualisation du masque : {roi_name}")
            fig = plt.figure(figsize=(12, 10))
            plotting.plot_roi(
                mask_img,
                bg_img=mni_template,
                title=f"ROI Mask: {roi_name}",
                display_mode="ortho",
                cmap="autumn",
                alpha=0.7,
                figure=fig,
            )
            out_path = output_dir / f"{roi_name}_mask_visualization.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"  Sauvegardé : {out_path}")

        if len(mask_imgs) > 1:
            logger.info("Création d'une vue combinée de tous les masques")
            fig = plt.figure(figsize=(16, 12))
            fig.suptitle("All ROI Masks – MNI Space", fontsize=18, fontweight="bold")

            n_masks = len(mask_imgs)
            for idx, (mask_img, roi_name) in enumerate(zip(mask_imgs, roi_names)):
                ax = plt.subplot(2, n_masks, idx + 1)
                plotting.plot_roi(
                    mask_img,
                    bg_img=mni_template,
                    title=roi_name,
                    display_mode="ortho",
                    axes=ax,
                    cmap="autumn",
                    alpha=0.7,
                )

            ax = plt.subplot(2, 1, 2)
            combined_data = np.zeros(mask_imgs[0].shape)
            for idx, mask_img in enumerate(mask_imgs):
                combined_data[mask_img.get_fdata() > 0] = idx + 1
            combined_img = nib.Nifti1Image(combined_data, mask_imgs[0].affine)
            plotting.plot_roi(
                combined_img,
                bg_img=mni_template,
                title="All ROIs Combined",
                display_mode="ortho",
                axes=ax,
                cmap="tab10",
                alpha=0.7,
            )

            out_path = output_dir / "all_masks_combined.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"  Vue combinée sauvegardée : {out_path}")

        logger.info(f"Toutes les visualisations de masques dans {output_dir}")

    def plot_simulation_vs_optimization(
        self,
        df: pd.DataFrame,
        metric: str = "mean",
        subject_col: str = "subject",
        condition_col: str = "condition",
    ) -> None:
        """
        Scatter plot: per subject, simulation (x) vs optimisation (y) for each ROI.

        Parameters
        ----------
        df :
            Features DataFrame whose ``condition`` column encodes
            ``<roi>_simulation`` / ``<roi>_optimization`` labels.
        metric :
            Column used for both axes.
        subject_col :
            Column containing subject identifiers.
        condition_col :
            Column containing condition labels.
        """
        output_dir = self.output_dir / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        df = df.copy()
        df["type"] = df[condition_col].apply(
            lambda x: "optimization" if "optimization" in x else "simulation"
        )
        df["roi"] = df[condition_col].apply(
            lambda x: x.replace("_simulation", "").replace("_optimization", "")
        )

        rois = df["roi"].unique()
        n_rois = len(rois)

        fig, axes = plt.subplots(1, n_rois, figsize=(6 * n_rois, 5))
        if n_rois == 1:
            axes = [axes]

        for i, roi in enumerate(rois):
            df_roi = df[df["roi"] == roi]
            pivot = df_roi.pivot_table(
                index=subject_col, columns="type", values=metric, aggfunc="first"
            ).dropna()

            if pivot.empty or "simulation" not in pivot.columns or "optimization" not in pivot.columns:
                continue

            ax = axes[i]
            ax.scatter(pivot["simulation"], pivot["optimization"], alpha=0.7, s=100)

            for subject, row in pivot.iterrows():
                ax.annotate(
                    subject,
                    (row["simulation"], row["optimization"]),
                    fontsize=8,
                    alpha=0.6,
                    xytext=(3, 3),
                    textcoords="offset points",
                )

            min_val = min(pivot["simulation"].min(), pivot["optimization"].min())
            max_val = max(pivot["simulation"].max(), pivot["optimization"].max())
            ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.3, label="y=x")
            ax.set_xlabel(f"Simulation {metric} (V/m)")
            ax.set_ylabel(f"Optimization {metric} (V/m)")
            ax.set_title(f"ROI: {roi}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        out_path = output_dir / "simulation_vs_optimization.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Sauvegardé : {out_path}")
