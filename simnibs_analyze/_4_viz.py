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
from _pipeline_io import space_tag, check_output, save_figure

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
        if_exists: str = "overwrite",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.cmap = cmap
        self.threshold_percentile = threshold_percentile
        self.bins = bins
        self.camera_position = camera_position
        self.if_exists = if_exists

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
        brain_bg_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Render an e-field NIfTI volume with PyVista (offscreen) and return an RGBA array.

        When ``brain_bg_path`` is provided, the brain surface is rendered as a
        semi-transparent white mesh and the e-field is overlaid as a coloured
        volume.  **Both must be in the same coordinate space** — for MNI e-fields
        (``*_scalar_MNI_magnE.nii.gz``), pass the ``T1_MNI_brain.nii.gz`` produced
        by :meth:`AnatomicalPreparer._make_mni_brain_bg`.  No resampling is
        performed here.

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
        brain_bg_path :
            Optional path to a skull-stripped T1 **in the same space as the
            e-field**.  Rendered as a semi-transparent brain surface behind the
            e-field volume.
        """
        efield_img = nib.as_closest_canonical(nib.load(str(efield_path)))
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

        # ── Fond anatomique (surface du cerveau) ──────────────────────────
        if brain_bg_path is not None:
            t1_img = nib.as_closest_canonical(nib.load(str(brain_bg_path)))
            t1_data = np.squeeze(t1_img.get_fdata())
            t1_spacing = t1_img.header.get_zooms()[:3]
            t1_origin = t1_img.affine[:3, 3]
            t1_grid = pv.ImageData()
            t1_grid.dimensions = np.array(t1_data.shape) + 1
            t1_grid.spacing = t1_spacing
            t1_grid.origin = t1_origin
            t1_grid.cell_data["t1"] = t1_data.flatten(order="F")
            nonzero_t1 = t1_data[t1_data > 0]
            iso_val = float(np.percentile(nonzero_t1, 20)) if nonzero_t1.size > 0 else 0.1
            brain_surface = t1_grid.cell_data_to_point_data().contour([iso_val])
            plotter.add_mesh(brain_surface, color="white", opacity=0.15, smooth_shading=True)

        # ── Volume e-field ────────────────────────────────────────────────
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
        space: str = "mni",
    ) -> None:
        """
        Generate one 3D figure per (roi, mode) pair.

        Parameters
        ----------
        file_info_by_roi_mode :
            Mapping ``(roi, mode) → [(subject, efield_path), ...]``.
        t1_brain_by_subject :
            Optional mapping ``subject → T1 brain path``. Brain surface is
            rendered behind the e-field.  Must be in the same space as the
            e-fields (use ``T1_MNI_brain.nii.gz`` for MNI,
            ``T1_subject_brain.nii.gz`` for subject space).
        space : str
            ``'mni'`` or ``'native'`` — included in the output filename so
            figures from both spaces are saved without overwriting each other.
        """
        output_dir = self.output_dir / "simu"
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = space_tag(space)

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
                brain_bg = (t1_brain_by_subject or {}).get(subject)
                image = self._create_3d_view(
                    efield_path=efield_path,
                    camera_position=self.camera_position,
                    cmap=self.cmap,
                    threshold_percentile=self.threshold_percentile,
                    vmin=vmin,
                    vmax=vmax,
                    brain_bg_path=brain_bg,
                )
                ax.imshow(image)
                ax.axis("off")
                ax.set_title(subject, fontsize=12)

            for idx in range(n_subjects, n_rows * n_cols):
                row, col = divmod(idx, n_cols)
                axes[row, col].axis("off")

            fig.suptitle(f"{roi} – {mode} ({space.upper()})", fontsize=16, fontweight="bold")
            plt.tight_layout()
            out_path = output_dir / f"efields_3d_{roi}_{mode}_{tag}_{self.camera_position}.png"
            save_figure(out_path, if_exists=self.if_exists, dpi=300, bbox_inches="tight")
            logger.info(f"  Sauvegardé : {out_path}")

        logger.info(f"Toutes les figures 3D dans {output_dir}")

    def efields_histograms(
        self,
        data_by_subject: Dict[str, List[Tuple[str, str, Path, Path]]],
        region: str = "intra",
        space: str = "mni",
    ) -> None:
        """
        Generate one histogram figure per subject comparing masked vs cleaned e-fields.

        Parameters
        ----------
        data_by_subject :
            Mapping ``subject → [(roi, mode, masked_path, cleaned_path), ...]``.
        region :
            Label inclus dans le titre et le nom de fichier (``"intra"`` ou ``"extra"``).
        space :
            ``'mni'`` ou ``'native'`` — suffixe de nom de fichier.
        """
        output_dir = self.output_dir / "preprocess"
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = space_tag(space)

        for subject, subject_data in data_by_subject.items():
            logger.info(f"Histogrammes {region}-ROI pour {subject}")
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
                row_idx, col = divmod(idx, n_cols)

                masked_data = nib.load(str(masked_path)).get_fdata().ravel()
                masked_data = masked_data[masked_data > 0]
                cleaned_data = nib.load(str(cleaned_path)).get_fdata().ravel()
                cleaned_data = cleaned_data[np.isfinite(cleaned_data) & (cleaned_data > 0)]

                ax = axes[row_idx, col]
                if masked_data.size > 0 or cleaned_data.size > 0:
                    all_vals = np.concatenate([a for a in [masked_data, cleaned_data] if a.size > 0])
                    xrange = (float(all_vals.min()), float(all_vals.max()))
                else:
                    xrange = None
                if masked_data.size > 0:
                    ax.hist(masked_data, bins=self.bins, range=xrange, alpha=0.6, label="Avant", color="red", density=True)
                if cleaned_data.size > 0:
                    ax.hist(cleaned_data, bins=self.bins, range=xrange, alpha=0.6, label="Après", color="blue", density=True)
                if masked_data.size == 0 and cleaned_data.size == 0:
                    ax.text(0.5, 0.5, "Aucune donnée", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlabel("E-field (V/m)", fontsize=10)
                ax.set_ylabel("Densité", fontsize=10)
                ax.set_title(f"{roi} | {mode}", fontsize=11)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            for idx in range(n_plots, n_rows * n_cols):
                row_idx, col = divmod(idx, n_cols)
                axes[row_idx, col].axis("off")

            fig.suptitle(
                f"Preprocessing {region}-ROI – {subject}",
                fontsize=16,
                fontweight="bold",
            )
            plt.tight_layout()
            out_path = output_dir / f"efields_histograms_{subject}_{region}_{tag}.png"
            save_figure(out_path, if_exists=self.if_exists, dpi=300, bbox_inches="tight")
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
            save_figure(out_path, if_exists=self.if_exists, dpi=150, bbox_inches="tight")
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
            save_figure(out_path, if_exists=self.if_exists, dpi=150, bbox_inches="tight")
            logger.info(f"  Vue combinée sauvegardée : {out_path}")

        logger.info(f"Toutes les visualisations de masques dans {output_dir}")

    def plot_simulation_vs_optimization(
        self,
        df: pd.DataFrame,
        metric: str = "mean",
        subject_col: str = "subject",
        condition_col: str = "condition",
        output_tag: str = "",
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
        output_tag :
            Optional suffix added to output filename (e.g., ``"mni"`` or ``"native"``).
        """
        output_dir = self.output_dir / "figures"
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
                missing = [c for c in ("simulation", "optimization") if c not in pivot.columns]
                logger.warning(
                    f"plot_simulation_vs_optimization: ROI '{roi}' ignorée — "
                    f"données manquantes : {missing}. "
                    "Assurez-vous que mode: [simulation, optimization] est configuré."
                )
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
        tagged = space_tag(output_tag) if output_tag else ""
        suffix = f"_{tagged}" if tagged else ""
        out_path = output_dir / f"simulation_vs_optimization{suffix}.png"
        save_figure(out_path, if_exists=self.if_exists, dpi=150, bbox_inches="tight")
        logger.info(f"Sauvegardé : {out_path}")
