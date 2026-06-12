"""visualizer.py — ColocVolume: multi-layer colocalised NIfTI visualisation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import image, plotting


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------
ROLE_BACKGROUND = "background"  # T1 — bg_img in nilearn, isosurface in PyVista
ROLE_STAT_MAP = "stat_map"  # e-field — stat_map in nilearn, volume in PyVista
ROLE_OVERLAY = "overlay"  # binary mask — add_overlay in nilearn, surface in PyVista
_VALID_ROLES = {ROLE_BACKGROUND, ROLE_STAT_MAP, ROLE_OVERLAY}


# ---------------------------------------------------------------------------
# Layer dataclass
# ---------------------------------------------------------------------------
@dataclass
class Layer:
    """Rendering properties for a single 3-D volume."""

    img: nib.Nifti1Image
    role: str
    cmap: str = "gray"
    opacity: float = 1.0
    clim: Optional[Tuple[float, float]] = None  # (vmin, vmax) or None=auto
    color: Optional[str] = None  # solid colour for binary masks
    label: str = ""

    def __repr__(self) -> str:
        return f"Layer('{self.label}', role={self.role}, shape={self.img.shape[:3]})"


# ---------------------------------------------------------------------------
# Input resolver
# ---------------------------------------------------------------------------
def _resolve_img(source) -> nib.Nifti1Image:
    """Resolve Path / str / EFieldAccessor / ROIResult → nib.Nifti1Image."""
    if isinstance(source, nib.Nifti1Image):
        return source
    if isinstance(source, (str, Path)):
        return nib.load(str(source))
    if hasattr(source, "img"):  # EFieldAccessor
        return source.img
    if hasattr(source, "mask_img"):  # ROIResult
        return source.mask_img
    raise TypeError(
        f"Cannot resolve {type(source).__name__} to a NIfTI image. "
        "Expected: nib.Nifti1Image, Path, str, EFieldAccessor, or ROIResult."
    )


# ---------------------------------------------------------------------------
# ColocVolume
# ---------------------------------------------------------------------------
class ColocVolume:
    """
    Multi-layer colocalised volume for 2-D and 3-D visualisation.

    Build the scene with :meth:`add_layer`, then call any of the plot methods.

    Example
    -------
    >>> vol = ColocVolume()
    >>> vol.add_layer(seg.t1,    role="background", cmap="gray",  opacity=0.15, label="T1")
    >>> vol.add_layer(sim.magnE, role="stat_map",   cmap="hot",   opacity=1.0,  label="magnE")
    >>> vol.add_layer(lesion,    role="overlay",    color="magenta", opacity=0.4, label="lesion")
    >>> vol.plot_slices(cut_coords=[28, -8, 54])
    >>> vol.plot_3d(camera_position="xy")
    """

    def __init__(self) -> None:
        self._layers: list[Layer] = []
        self._reference: nib.Nifti1Image | None = None

    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------

    def add_layer(
        self,
        source,
        role: str = "stat_map",
        cmap: str = "gray",
        opacity: float = 1.0,
        clim: Optional[Tuple[float, float]] = None,
        color: Optional[str] = None,
        label: str = "",
        resample: bool = True,
    ) -> "ColocVolume":
        """Add a 3-D volume as a rendering layer. Returns self (chainable)."""
        if role not in _VALID_ROLES:
            raise ValueError(f"role must be one of {_VALID_ROLES}, got '{role}'")

        img = _resolve_img(source)

        # Squeeze (182,218,182,1) → (182,218,182)
        if img.ndim == 4 and img.shape[3] == 1:
            img = image.index_img(img, 0)

        # First layer sets the reference grid
        if self._reference is None:
            self._reference = img
        elif resample and not self._grids_match(img, self._reference):
            img = image.resample_to_img(img, self._reference, interpolation="nearest")

        self._layers.append(
            Layer(
                img=img,
                role=role,
                cmap=cmap,
                opacity=opacity,
                clim=clim,
                color=color,
                label=label,
            )
        )
        return self

    def set_clim(self, label: str, vmin: float, vmax: float) -> None:
        """Override colour limits on a named layer (e.g. for cohort normalisation)."""
        self._get_layer(label).clim = (vmin, vmax)

    @property
    def layers(self) -> list[Layer]:
        return list(self._layers)

    def _get_layer(self, label: str) -> Layer:
        for layer in self._layers:
            if layer.label == label:
                return layer
        raise KeyError(
            f"No layer '{label}'. Available: {[layer.label for layer in self._layers]}"
        )

    def _by_role(self, role: str) -> list[Layer]:
        return [layer for layer in self._layers if layer.role == role]

    @staticmethod
    def _grids_match(a: nib.Nifti1Image, b: nib.Nifti1Image) -> bool:
        return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=1e-4)

    def __repr__(self) -> str:
        lines = [f"ColocVolume ({len(self._layers)} layers)"]
        for layer in self._layers:
            lines.append(f"  {layer}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 2-D — nilearn
    # ------------------------------------------------------------------

    def plot_slices(
        self,
        cut_coords=None,
        display_mode: str = "ortho",
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Render 2-D orthogonal (or single-axis) slices via nilearn.

        Parameters
        ----------
        cut_coords :
            MNI coordinates ``(x, y, z)`` for ortho cross-hairs, or an int for
            the number of slices when *display_mode* is a single axis.
        display_mode :
            ``'ortho'`` (default), ``'x'``, ``'y'``, ``'z'``, ``'xz'``, …
        title :
            Figure title — defaults to the stat_map layer label.
        figsize :
            Matplotlib figure size in inches.
        """
        bg_layers = self._by_role(ROLE_BACKGROUND)
        stat_layers = self._by_role(ROLE_STAT_MAP)
        ov_layers = self._by_role(ROLE_OVERLAY)

        if not stat_layers and not ov_layers:
            raise ValueError("Need at least one 'stat_map' or 'overlay' layer.")

        bg_img = bg_layers[0].img if bg_layers else None
        fig = plt.figure(figsize=figsize or (14, 4))

        if stat_layers:
            stat = stat_layers[0]
            kw = dict(
                stat_map_img=stat.img,
                bg_img=bg_img,
                display_mode=display_mode,
                cut_coords=cut_coords,
                cmap=stat.cmap,
                colorbar=True,
                title=title or stat.label,
                figure=fig,
                dim=-1,
            )
            if stat.clim:
                kw["vmax"] = stat.clim[1]
                kw["threshold"] = stat.clim[0]
            display = plotting.plot_stat_map(**kw)
        else:
            display = plotting.plot_anat(
                bg_img,
                display_mode=display_mode,
                cut_coords=cut_coords,
                title=title or "",
                figure=fig,
            )

        for ov in ov_layers:
            display.add_overlay(ov.img, cmap=ov.cmap or "autumn", alpha=ov.opacity)

        plt.show()

    def plot_parallel_slices(
        self,
        axis: str = "z",
        n_slices: int = 12,
        title: Optional[str] = None,
    ) -> None:
        """
        Render *n_slices* evenly-spaced slices along one axis.

        Parameters
        ----------
        axis :
            ``'z'`` axial (default), ``'x'`` sagittal, ``'y'`` coronal.
        n_slices :
            Number of parallel slices.
        """
        bg_layers = self._by_role(ROLE_BACKGROUND)
        stat_layers = self._by_role(ROLE_STAT_MAP)

        if not stat_layers:
            raise ValueError("Need a 'stat_map' layer for parallel slicing.")

        stat = stat_layers[0]
        bg_img = bg_layers[0].img if bg_layers else None

        kw = dict(
            stat_map_img=stat.img,
            bg_img=bg_img,
            display_mode=axis,
            cut_coords=n_slices,
            cmap=stat.cmap,
            colorbar=True,
            title=title or f"{stat.label} — {axis} × {n_slices}",
            dim=-1,
        )
        if stat.clim:
            kw["vmax"] = stat.clim[1]
            kw["threshold"] = stat.clim[0]

        display = plotting.plot_stat_map(**kw)

        for ov in self._by_role(ROLE_OVERLAY):
            display.add_overlay(ov.img, cmap=ov.cmap or "autumn", alpha=ov.opacity)

        plt.show()

    # ------------------------------------------------------------------
    # 3-D helpers (shared by plot_3d and view_interactive)
    # ------------------------------------------------------------------

    def _build_pyvista_plotter(
        self,
        camera_position: str = "xy",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        off_screen: bool = True,
    ):
        """Build a PyVista Plotter from all registered layers."""
        try:
            import pyvista as pv
        except ImportError as exc:
            raise ImportError(
                "pyvista is required for 3-D rendering. "
                "Install with: pip install pyvista"
            ) from exc

        plotter = pv.Plotter(off_screen=off_screen)
        plotter.set_background("black")  # ✅ FIX 1 : fond noir

        # ── stat_map → volume rendering (RENDU EN PREMIER) ───────────────
        #    PyVista compose mal volume + mesh : le volume doit être
        #    ajouté AVANT les surfaces pour que le depth-peeling fonctionne.
        for layer in self._by_role(ROLE_STAT_MAP):
            canonical = nib.as_closest_canonical(layer.img)
            raw = np.squeeze(canonical.get_fdata()).astype(np.float32)
            if raw.ndim == 4:
                raw = np.linalg.norm(raw, axis=-1)

            grid = self._to_pv_image(canonical, raw, key="values")
            grid = grid.cell_data_to_point_data()  # ✅ FIX 2 : point_data

            _vmin = layer.clim[0] if layer.clim else (vmin if vmin is not None else 0.0)
            _vmax = (
                layer.clim[1]
                if layer.clim
                else (
                    vmax
                    if vmax is not None
                    else (
                        float(np.percentile(raw[raw > 0], 99))
                        if np.any(raw > 0)
                        else 1.0
                    )
                )
            )

            # ✅ FIX 3 : rampe d'opacité — zéros totalement transparents
            opacity = [0.0, 0.05, 0.2, 0.4, 0.7, 1.0]

            plotter.add_volume(
                grid,
                scalars="values",
                cmap=layer.cmap,
                clim=[_vmin, _vmax],
                opacity=opacity,
            )

        # ── background → T1 isosurface (semi-transparent, APRÈS volume) ──
        for layer in self._by_role(ROLE_BACKGROUND):
            canonical = nib.as_closest_canonical(layer.img)
            data = np.squeeze(canonical.get_fdata()).astype(np.float32)
            grid = self._to_pv_image(canonical, data, key="t1")
            pts = grid.cell_data_to_point_data()
            thresh = float(np.percentile(data[data > 0], 15)) if data.any() else 1.0
            surface = pts.contour([thresh], scalars="t1")
            if surface.n_points > 0:
                plotter.add_mesh(
                    surface,
                    color="white",
                    opacity=layer.opacity,
                    smooth_shading=True,
                )

        # ── overlays → binary surface ────────────────────────────────────
        for layer in self._by_role(ROLE_OVERLAY):
            canonical = nib.as_closest_canonical(layer.img)
            data = np.squeeze(canonical.get_fdata()).astype(np.float32)
            grid = self._to_pv_image(canonical, data, key="mask")
            surface = grid.threshold(0.5).extract_surface()
            if surface.n_points > 0:
                plotter.add_mesh(
                    surface,
                    color=layer.color or "cyan",
                    opacity=layer.opacity,
                    smooth_shading=True,
                )

        # Depth peeling pour transparence correcte (volume vu à travers mesh)
        plotter.enable_depth_peeling(number_of_peels=8)
        plotter.camera_position = camera_position
        return plotter

    @staticmethod
    def _to_pv_image(img: nib.Nifti1Image, data: np.ndarray, key: str):
        """Wrap a 3-D numpy array in a pv.ImageData cell-data grid."""
        import pyvista as pv

        data = np.squeeze(data)
        grid = pv.ImageData()
        grid.dimensions = np.array(data.shape) + 1
        grid.spacing = img.header.get_zooms()[:3]
        grid.origin = img.affine[:3, 3]
        grid.cell_data[key] = data.flatten(order="F")
        return grid

    # ------------------------------------------------------------------
    # 3-D — PyVista (offscreen PNG)
    # ------------------------------------------------------------------

    def plot_3d(
        self,
        camera_position: str = "xy",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        title: Optional[str] = None,
    ) -> np.ndarray:
        """
        Offscreen 3-D render — returns an RGB numpy array.

        Parameters
        ----------
        camera_position :
            PyVista camera preset: ``'xy'``, ``'xz'``, ``'yz'``.
        vmin, vmax :
            Global colour scale limits (override per-layer *clim*).
        title :
            Window title (shown in screenshot metadata only).

        Returns
        -------
        np.ndarray
            RGB image array ``(H, W, 3)``.
        """
        plotter = self._build_pyvista_plotter(
            camera_position=camera_position,
            vmin=vmin,
            vmax=vmax,
            off_screen=True,
        )
        if title:
            plotter.title = title
        img = plotter.screenshot(return_img=True)
        plotter.close()
        return img

    # ------------------------------------------------------------------
    # 3-D — PyVista (interactive window)
    # ------------------------------------------------------------------

    def view_interactive(
        self,
        camera_position: str = "xy",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        title: Optional[str] = None,
    ) -> None:
        """
        Open an interactive, rotatable PyVista window (blocks until closed).

        Parameters
        ----------
        camera_position :
            Initial camera preset: ``'xy'``, ``'xz'``, ``'yz'``.
        vmin, vmax :
            Global colour scale limits.
        title :
            Window title.
        """
        plotter = self._build_pyvista_plotter(
            camera_position=camera_position,
            vmin=vmin,
            vmax=vmax,
            off_screen=False,
        )
        plotter.title = title or " | ".join(
            layer.label for layer in self._layers if layer.label
        )
        plotter.show()
