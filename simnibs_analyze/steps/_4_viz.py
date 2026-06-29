from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting

from .._logging import get_logger
from .._pipeline_io import space_tag, save_figure

logger = get_logger(__name__)


class SimnibsViz(): ## Niivue for ipynivue? # + plotting de nilearn ?
    """
    An helper class for simnibs visualisations : 2d and 3D methods niftii viz + statistical viz
    """

    def __init__(
        self,
        output_dir: Path,
        config: Path, #config_viz.yaml
        t1;
        
    ) -> None:
        self.output_dir = Path(output_dir)
        self.cmap = cmap
        self.threshold_percentile = threshold_percentile
        self.bins = bins
        self.camera_position = camera_position
        self.if_exists = if_exists

    # ------------------------------------------------------------------
    # Methodes de visualisations 3D 
    # ------------------------------------------------------------------

    @staticmethod
    def _build_plotter(
        efield_path: Path,
        camera_position: str = "xy",
        cmap: str = "hot",
        threshold_percentile: float = 0.0,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        mask_path: Optional[Path] = None,
        mask_color: str = "cyan",
        mask_opacity: float = 0.3,
        off_screen: bool = True,
        component: Union[str, int] = "magnitude",
    ):  # -> pv.Plotter (lazy import, not annotated to avoid import at module level)
        """
        Build and return a configured PyVista Plotter for e-field visualisation.

        Shared by :meth:`_create_3d_view` (offscreen static render) and
        :meth:`view_efield_interactive` (interactive window).

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
        mask_path :
            Optional binary mask NIfTI to colocalize with the e-field
            (e.g. ``cereb_mask.nii.gz`` from SimNIBS surfaces/).  Rendered as
            a semi-transparent coloured surface in the same space.
        mask_color :
            Colour of the mask surface mesh (default ``'cyan'``).
        mask_opacity :
            Opacity of the mask surface mesh (default 0.3).
        off_screen :
            If ``True``, creates an offscreen plotter (static PNG).
            If ``False``, creates an interactive window plotter.
        component :
            How to reduce a 4D vector field (Ex/Ey/Ez) to a scalar volume.
            ``'magnitude'`` (default) → computes ‖E‖ = √(Ex²+Ey²+Ez²).
            ``0``, ``1``, ``2`` → extracts a single cartesian component
            (Ex, Ey, Ez respectively); a diverging colormap (``coolwarm``) is
            used automatically so that negative values are visible.
            Ignored for 3D (scalar) inputs.
        """
        _COMPONENT_LABELS = {0: "Ex", 1: "Ey", 2: "Ez"}

        efield_img = nib.as_closest_canonical(nib.load(str(efield_path)))
        raw = np.squeeze(efield_img.get_fdata())

        # ── Cas 1 : fichier scalaire 3D (scalar_magnE, …) ─────────────────
        if raw.ndim == 3:
            logger.info(f"3D scalar e-field {raw.shape} — loading as-is")
            data = raw
            # cmap par défaut : hot (valeurs ≥ 0)

        # ── Cas 2 : champ vectoriel 4D (scalar_E → Ex/Ey/Ez) ─────────────
        elif raw.ndim == 4:
            if component == "magnitude":
                logger.info(
                    f"4D vector e-field {raw.shape} — computing ‖E‖ "
                    f"(tip: use scalar_magnE directly for faster loading)"
                )
                data = np.linalg.norm(raw, axis=-1)
                # cmap par défaut : hot (valeurs ≥ 0)
            elif isinstance(component, int) and 0 <= component <= 2:
                label = _COMPONENT_LABELS[component]
                logger.info(
                    f"4D vector e-field {raw.shape} — extracting component "
                    f"{label} (vol[{component}]), signed"
                )
                data = raw[..., component]
                if cmap == "hot":
                    cmap = "coolwarm"  # colormap divergente pour valeurs signées
            else:
                raise ValueError(
                    f"component must be 'magnitude', 0, 1, or 2 — got {component!r}"
                )
        else:
            raise ValueError(f"Unexpected e-field shape: {raw.shape}")

        try:
            import pyvista as pv
        except ImportError as exc:
            raise ImportError(
                "pyvista is required for 3D rendering. "
                "Install it with: pip install pyvista"
            ) from exc

        # Les voxels à 0 (hors-cerveau, masque fond) → NaN = transparents en volume rendering
        data = data.astype(float)
        data[data == 0] = np.nan

        efield_spacing = efield_img.header.get_zooms()[:3]
        efield_origin = efield_img.affine[:3, 3]

        grid = pv.ImageData()
        grid.dimensions = np.array(data.shape) + 1
        grid.spacing = efield_spacing
        grid.origin = efield_origin
        grid.cell_data["values"] = data.flatten(order="F")

        plotter = pv.Plotter(off_screen=off_screen)

        # ── Mask surface overlay ──────────────────────────────────────────
        if mask_path is not None:
            mask_img = nib.as_closest_canonical(nib.load(str(mask_path)))
            mask_data = np.squeeze(mask_img.get_fdata()).astype(np.float32)
            mask_spacing = mask_img.header.get_zooms()[:3]
            mask_origin = mask_img.affine[:3, 3]
            mask_grid = pv.ImageData()
            mask_grid.dimensions = np.array(mask_data.shape) + 1
            mask_grid.spacing = mask_spacing
            mask_grid.origin = mask_origin
            mask_grid.cell_data["mask"] = mask_data.flatten(order="F")
            mask_surface = mask_grid.threshold(0.5).extract_surface()
            if mask_surface.n_points > 0:
                plotter.add_mesh(
                    mask_surface,
                    color=mask_color,
                    opacity=mask_opacity,
                    smooth_shading=True,
                )

        # ── Volume e-field ────────────────────────────────────────────────
        if vmin is not None and vmax is not None:
            plotter.add_volume(grid, cmap=cmap, clim=[vmin, vmax])
        else:
            plotter.add_volume(grid, cmap=cmap)

        plotter.camera_position = camera_position
        return plotter

    @staticmethod
    def _create_3d_view(
        efield_path: Path,
        camera_position: str = "xy",
        cmap: str = "hot",
        threshold_percentile: float = 0.0,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        mask_path: Optional[Path] = None,
        mask_color: str = "cyan",
        mask_opacity: float = 0.3,
        component: Union[str, int] = "magnitude",
    ) -> np.ndarray:
        """Offscreen render — returns an RGBA array (used by :meth:`efields_figures`)."""
        plotter = Visualizer._build_plotter(
            efield_path=efield_path,
            camera_position=camera_position,
            cmap=cmap,
            threshold_percentile=threshold_percentile,
            vmin=vmin,
            vmax=vmax,
            mask_path=mask_path,
            mask_color=mask_color,
            mask_opacity=mask_opacity,
            off_screen=True,
            component=component,
        )
        image = plotter.screenshot(return_img=True)
        plotter.close()
        return image

    # ------------------------------------------------------------------
    # Toutes les methodes 2D sont à remplacer par un simple appel au methodes de visualisation de plotting de nilearn (plot anat par exemple)
    # ------------------------------------------------------------------

   

    # ------------------------------------------------------------------
    # Methodes pour garder les figures de cohortes
    # ------------------------------------------------------------------
    def set_scale()
    
    # ------------------------------------------------------------------
    # Methods for statistical visualisation
    # ------------------------------------------------------------------ 
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
            Label included in the figure title and filename (``"intra"`` or ``"extra"``).
        space :
            ``'mni'`` or ``'native'`` — filename suffix.
        """
        output_dir = self.output_dir / "2-preprocess"
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = space_tag(space)

        for subject, subject_data in data_by_subject.items():
            logger.info(f"{region}-ROI histograms for {subject}")
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
                cleaned_data = cleaned_data[
                    np.isfinite(cleaned_data) & (cleaned_data > 0)
                ]

                ax = axes[row_idx, col]
                if masked_data.size > 0 or cleaned_data.size > 0:
                    all_vals = np.concatenate(
                        [a for a in [masked_data, cleaned_data] if a.size > 0]
                    )
                    xrange = (float(all_vals.min()), float(all_vals.max()))
                else:
                    xrange = None
                if masked_data.size > 0:
                    ax.hist(
                        masked_data,
                        bins=self.bins,
                        range=xrange,
                        alpha=0.6,
                        label="Before",
                        color="red",
                        density=True,
                    )
                if cleaned_data.size > 0:
                    ax.hist(
                        cleaned_data,
                        bins=self.bins,
                        range=xrange,
                        alpha=0.6,
                        label="After",
                        color="blue",
                        density=True,
                    )
                if masked_data.size == 0 and cleaned_data.size == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "No data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                ax.set_xlabel("E-field (V/m)", fontsize=10)
                ax.set_ylabel("Density", fontsize=10)
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
            save_figure(
                out_path, if_exists=self.if_exists, dpi=300, bbox_inches="tight"
            )
            logger.info(f"  Saved: {out_path}")

        logger.info(f"All histograms saved in {output_dir}")

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
        output_dir = self.output_dir / "3-analysis"
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

            if (
                pivot.empty
                or "simulation" not in pivot.columns
                or "optimization" not in pivot.columns
            ):
                missing = [
                    c for c in ("simulation", "optimization") if c not in pivot.columns
                ]
                logger.warning(
                    f"plot_simulation_vs_optimization: ROI '{roi}' skipped — "
                    f"missing data: {missing}. "
                    "Make sure mode: [simulation, optimization] is configured."
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
            ax.plot(
                [min_val, max_val], [min_val, max_val], "k--", alpha=0.3, label="y=x"
            )
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
        logger.info(f"Saved: {out_path}")

    def plot_simulation_summary(
        self,
        df: pd.DataFrame,
        metric: str = "mean",
        subject_col: str = "subject",
        condition_col: str = "condition",
        output_tag: str = "",
    ) -> None:
        """
        Per-condition summary figure (boxplot + stripplot) showing e-field values
        across subjects for each mode (simulation and/or optimization).

        Works with simulation-only datasets, unlike
        :meth:`plot_simulation_vs_optimization` which requires both modes.

        Parameters
        ----------
        df :
            Features DataFrame with at least ``subject``, ``condition``, and
            ``<metric>`` columns.
        metric :
            Numeric column to plot on the y-axis (default ``'mean'``).
        subject_col :
            Column containing subject identifiers.
        condition_col :
            Column containing condition labels (e.g. ``'stimSD_simulation'``).
        output_tag :
            Optional suffix added to the output filename.
        """
        output_dir = self.output_dir / "3-analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        if metric not in df.columns:
            logger.warning(
                f"plot_simulation_summary: metric '{metric}' not in DataFrame columns "
                f"({list(df.columns)}). Skipped."
            )
            return

        df = df.copy()
        df["mode"] = df[condition_col].apply(
            lambda x: "optimization" if "optimization" in x else "simulation"
        )
        df["roi"] = df[condition_col].apply(
            lambda x: x.replace("_simulation", "").replace("_optimization", "")
        )

        rois = sorted(df["roi"].unique())
        modes = sorted(df["mode"].unique())
        n_rois = len(rois)

        fig, axes = plt.subplots(
            1, n_rois, figsize=(max(5, 4 * n_rois), 5), squeeze=False
        )
        axes = axes[0]

        palette = {"simulation": "#4C8BE2", "optimization": "#E2824C"}

        for ax, roi in zip(axes, rois):
            df_roi = df[df["roi"] == roi].copy()

            # boxplot per mode
            positions = {m: i for i, m in enumerate(modes)}
            for mode, grp in df_roi.groupby("mode"):
                vals = grp[metric].dropna().values
                pos = positions[mode]
                ax.boxplot(
                    vals,
                    positions=[pos],
                    widths=0.4,
                    patch_artist=True,
                    boxprops=dict(facecolor=palette.get(mode, "#aaa"), alpha=0.4),
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(color="gray"),
                    capprops=dict(color="gray"),
                    flierprops=dict(marker="", linestyle="none"),
                    showfliers=False,
                )
                # individual points
                jitter = (np.random.default_rng(42).random(len(vals)) - 0.5) * 0.2
                ax.scatter(
                    np.full(len(vals), pos) + jitter,
                    vals,
                    color=palette.get(mode, "#aaa"),
                    alpha=0.8,
                    s=60,
                    zorder=3,
                )

            ax.set_xticks(list(positions.values()))
            ax.set_xticklabels(list(positions.keys()), rotation=15, ha="right")
            ax.set_ylabel(f"E-field {metric} (V/m)")
            ax.set_title(f"ROI: {roi}")
            ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        tagged = space_tag(output_tag) if output_tag else ""
        suffix = f"_{tagged}" if tagged else ""
        out_path = output_dir / f"simulation_summary{suffix}.png"
        save_figure(out_path, if_exists=self.if_exists, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {out_path}")



#  def view_efield_interactive(
#         self,
#         efield_path: Path,
#         mask_path: Optional[Path] = None,
#         mask_color: str = "cyan",
#         mask_opacity: float = 0.3,
#         threshold_percentile: Optional[float] = None,
#         vmin: Optional[float] = None,
#         vmax: Optional[float] = None,
#         cmap: Optional[str] = None,
#         camera_position: Optional[str] = None,
#         title: Optional[str] = None,
#         component: Union[str, int] = "magnitude",
#     ) -> None:
#         """
#         Open an interactive, rotatable 3D PyVista window for a single e-field.

#         The window is fully interactive (rotate, zoom, pan) and blocks until
#         closed.  No extra dependencies beyond PyVista are required.

#         Parameters
#         ----------
#         efield_path :
#             Path to the e-field NIfTI file.
#         mask_path :
#             Optional binary mask to colocalize (e.g.
#             ``m2m_<sub>/surfaces/cereb_mask.nii.gz``).  Rendered as a
#             semi-transparent coloured surface.
#         mask_color :
#             Colour of the mask surface (default ``'cyan'``).
#         mask_opacity :
#             Opacity of the mask surface in [0, 1] (default 0.3).
#         threshold_percentile :
#             Percentile cutoff for non-zero voxels.  Defaults to
#             ``self.threshold_percentile``.
#         vmin, vmax :
#             Explicit colour scale limits.  Auto-scales if both ``None``.
#         cmap :
#             Colormap override.  Defaults to ``self.cmap``.
#         camera_position :
#             Initial camera position override.  Defaults to
#             ``self.camera_position``.
#         title :
#             Window title.  Defaults to the e-field filename stem.

#         Examples
#         --------
#         >>> viz = Visualizer(output_dir="output/")
#         >>> viz.view_efield_interactive(
#         ...     efield_path="sub-0011_magnE.nii.gz",
#         ...     mask_path="m2m_0011/surfaces/cereb_mask.nii.gz",
#         ...     mask_color="cyan",
#         ...     mask_opacity=0.3,
#         ... )
#         """
#         plotter = self._build_plotter(
#             efield_path=Path(efield_path),
#             camera_position=camera_position or self.camera_position,
#             cmap=cmap or self.cmap,
#             threshold_percentile=(
#                 threshold_percentile
#                 if threshold_percentile is not None
#                 else self.threshold_percentile
#             ),
#             vmin=vmin,
#             vmax=vmax,
#             mask_path=Path(mask_path) if mask_path is not None else None,
#             mask_color=mask_color,
#             mask_opacity=mask_opacity,
#             off_screen=False,
#             component=component,
#         )
#         comp_label = {0: " [Ex]", 1: " [Ey]", 2: " [Ez]"}.get(component, "")
#         plotter.title = (title or Path(efield_path).stem) + comp_label
#         logger.info(f"Opening interactive 3D viewer: {efield_path}")
#         plotter.show()
