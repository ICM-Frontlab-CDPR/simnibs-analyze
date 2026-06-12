"""
visualizer.py
-------------
Colocalised multi-layer NIfTI visualization.

Provides :class:`ColocVolume` — an object that stacks 3D volumes with
per-layer rendering properties, then exposes unified 2D (nilearn) and
3D (PyVista) plotting methods.

Usage
-----
>>> import simnibs_reader as snr
>>> from simnibs_analyze.steps.visualizer import ColocVolume
>>>
>>> sim = snr.simulation("sim_sub01/")
>>> seg = snr.segmentation("m2m_sub01/")
>>>
>>> vol = ColocVolume()
>>> vol.add_layer(seg.t1, role="background", cmap="gray", opacity=0.15, label="T1")
>>> vol.add_layer(sim.magnE, role="stat_map", cmap="hot", label="magnE")
>>> vol.add_layer(lesion_path, role="overlay", color="magenta", opacity=0.4, label="lesion")
>>>
>>> vol.plot_slices(cut_coords=[28, -8, 54])
>>> vol.plot_3d(camera_position="xy")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import image, plotting

logger = logging.getLogger(__name__)

# Types that can be resolved to a NIfTI image
NiftiLike = Union[nib.Nifti1Image, "Path", str]

# Roles
ROLE_BACKGROUND = "background"
ROLE_STAT_MAP = "stat_map"
ROLE_OVERLAY = "overlay"
ROLE_CONTOUR = "contour"
_VALID_ROLES = {ROLE_BACKGROUND, ROLE_STAT_MAP, ROLE_OVERLAY, ROLE_CONTOUR}


@dataclass
class Layer:
    """Rendering properties for a single 3D volume."""
    img: nib.Nifti1Image
    role: str
    cmap: str = "gray"
    opacity: float = 1.0
    clim: Optional[Tuple[float, float]] = None
    color: Optional[str] = None
    label: str = ""
    
    def __repr__(self) -> str:
        shape = self.img.shape[:3]
        return f"Layer('{self.label}', role={self.role}, shape={shape})"


def _resolve_img(source) -> nib.Nifti1Image:
    """Resolve various input types to a nibabel NIfTI image."""
    if isinstance(source, nib.Nifti1Image):
        return source
    if isinstance(source, (str, Path)):
        return nib.load(str(source))
    # EFieldAccessor / ROIResult — duck-type on .img
    if hasattr(source, "img"):
        img = source.img
        if isinstance(img, nib.Nifti1Image):
            return img
    # ROIResult — duck-type on .mask_img
    if hasattr(source, "mask_img"):
        return source.mask_img
    raise TypeError(
        f"Cannot resolve {type(source).__name__} to a NIfTI image. "
        f"Expected: nib.Nifti1Image, Path, str, EFieldAccessor, or ROIResult."
    )


class ColocVolume:
    """Multi-layer colocalised volume for visualization."""
    
    def __init__(self) -> None:
        self._layers: list[Layer] = []
        self._reference: nib.Nifti1Image | None = None
    
    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------
    
    def add_layer(
        self,
        source: NiftiLike,
        role: str = "stat_map",
        cmap: str = "gray",
        opacity: float = 1.0,
        clim: tuple[float, float] | None = None,
        color: str | None = None,
        label: str = "",
        resample: bool = True,
    ) -> "ColocVolume":
        """Add a 3D volume as a rendering layer.
        
        Parameters
        ----------
        source
            NIfTI image, file path, EFieldAccessor, or ROIResult.
        role
            One of ``"background"``, ``"stat_map"``, ``"overlay"``, ``"contour"``.
        cmap
            Matplotlib/PyVista colormap name.
        opacity
            Rendering opacity (0.0–1.0).
        clim
            Colour scale limits ``(vmin, vmax)``. ``None`` = auto.
        color
            Solid colour for binary masks (e.g. ``"magenta"``). 
            Used for ``"overlay"`` role.
        label
            Human-readable label for titles/legends.
        resample
            If ``True`` and grids differ, resample to the reference grid.
        
        Returns
        -------
        self
            For method chaining.
        """
        if role not in _VALID_ROLES:
            raise ValueError(f"role must be one of {_VALID_ROLES}, got '{role}'")
        
        img = _resolve_img(source)
        
        # Squeeze to 3D if needed (e.g. (182, 218, 182, 1))
        if img.ndim == 4 and img.shape[3] == 1:
            img = image.index_img(img, 0)
        
        # Set reference from first layer, or resample to match
        if self._reference is None:
            self._reference = img
            logger.debug(f"Reference grid set from '{label}': {img.shape}")
        elif resample and not self._grids_match(img, self._reference):
            logger.info(f"Resampling '{label}' to reference grid")
            img = image.resample_to_img(img, self._reference, interpolation="nearest")
        
        layer = Layer(
            img=img, role=role, cmap=cmap, opacity=opacity,
            clim=clim, color=color, label=label,
        )
        self._layers.append(layer)
        logger.debug(f"Added {layer}")
        return self
    
    @property
    def layers(self) -> list[Layer]:
        """All registered layers."""
        return list(self._layers)
    
    @property
    def n_layers(self) -> int:
        return len(self._layers)
    
    def get_layer(self, label: str) -> Layer:
        """Retrieve a layer by label."""
        for layer in self._layers:
            if layer.label == label:
                return layer
        raise KeyError(f"No layer with label '{label}'. Available: {[l.label for l in self._layers]}")
    
    def set_clim(self, label: str, vmin: float, vmax: float) -> None:
        """Set colour limits on a layer (useful for cohort-level normalisation)."""
        self.get_layer(label).clim = (vmin, vmax)
    
    def _by_role(self, role: str) -> list[Layer]:
        return [l for l in self._layers if l.role == role]
    
    @staticmethod
    def _grids_match(img_a: nib.Nifti1Image, img_b: nib.Nifti1Image) -> bool:
        return (
            img_a.shape[:3] == img_b.shape[:3]
            and np.allclose(img_a.affine, img_b.affine, atol=1e-4)
        )
    
    # ------------------------------------------------------------------
    # 2D — nilearn
    # ------------------------------------------------------------------
    
    def plot_slices(
        self,
        cut_coords: list[float] | int | None = None,
        display_mode: str = "ortho",
        title: str | None = None,
        figsize: tuple[int, int] | None = None,
        output_path: Path | str | None = None,
        dpi: int = 150,
    ) -> None:
        """Render 2D slices using nilearn.
        
        Parameters
        ----------
        cut_coords
            MNI coordinates for the cross-hairs, or number of slices (int).
        display_mode
            ``"ortho"``, ``"x"``, ``"y"``, ``"z"``, ``"xz"``, etc.
        title
            Figure title.
        figsize
            Figure size in inches.
        output_path
            If provided, save figure to this path.
        dpi
            Resolution for saved figure.
        """
        if not self._layers:
            raise ValueError("No layers added. Call add_layer() first.")
        
        # Find background and stat_map
        bg_layers = self._by_role(ROLE_BACKGROUND)
        stat_layers = self._by_role(ROLE_STAT_MAP)
        overlay_layers = self._by_role(ROLE_OVERLAY) + self._by_role(ROLE_CONTOUR)
        
        bg_img = bg_layers[0].img if bg_layers else None
        
        if not stat_layers and not overlay_layers:
            raise ValueError("Need at least one 'stat_map' or 'overlay' layer to plot.")
        
        fig = plt.figure(figsize=figsize or (12, 4))
        
        # Plot stat_map (main e-field)
        if stat_layers:
            stat = stat_layers[0]
            kwargs = dict(
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
            if stat.clim is not None:
                kwargs["vmax"] = stat.clim[1]
                # nilearn uses threshold, not vmin for stat_map
                kwargs["threshold"] = stat.clim[0]
            
            display = plotting.plot_stat_map(**kwargs)
        else:
            # No stat_map — just plot background with ROI overlays
            display = plotting.plot_anat(
                bg_img,
                display_mode=display_mode,
                cut_coords=cut_coords,
                title=title or "",
                figure=fig,
            )
        
        # Add overlay layers
        for overlay in overlay_layers:
            cmap = overlay.cmap
            if overlay.color and overlay.color in plt.colormaps():
                cmap = overlay.color
            display.add_overlay(
                overlay.img,
                cmap=cmap,
                alpha=overlay.opacity,
            )
        
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved: {output_path}")
        
        plt.show()
    
    def plot_parallel_slices(
        self,
        axis: str = "z",
        n_slices: int = 12,
        title: str | None = None,
        output_path: Path | str | None = None,
        dpi: int = 150,
    ) -> None:
        """Multiple parallel slices along one axis.
        
        Parameters
        ----------
        axis
            ``"x"`` (sagittal), ``"y"`` (coronal), ``"z"`` (axial).
        n_slices
            Number of evenly-spaced slices.
        """
        if not self._layers:
            raise ValueError("No layers added.")
        
        bg_layers = self._by_role(ROLE_BACKGROUND)
        stat_layers = self._by_role(ROLE_STAT_MAP)
        
        bg_img = bg_layers[0].img if bg_layers else None
        
        if not stat_layers:
            raise ValueError("Need a 'stat_map' layer for parallel slicing.")
        
        stat = stat_layers[0]
        kwargs = dict(
            stat_map_img=stat.img,
            bg_img=bg_img,
            display_mode=axis,
            cut_coords=n_slices,
            cmap=stat.cmap,
            colorbar=True,
            title=title or f"{stat.label} — {axis} slices",
        )
        if stat.clim:
            kwargs["vmax"] = stat.clim[1]
            kwargs["threshold"] = stat.clim[0]
        
        display = plotting.plot_stat_map(**kwargs)
        
        # Add overlays
        for overlay in self._by_role(ROLE_OVERLAY):
            display.add_overlay(overlay.img, alpha=overlay.opacity)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved: {output_path}")
        
        plt.show()
    
    # ------------------------------------------------------------------
    # 3D — PyVista
    # ------------------------------------------------------------------
    
    def _build_pyvista_plotter(
        self,
        camera_position: str = "xy",
        off_screen: bool = True,
    ):
        """Build a PyVista plotter from all layers."""
        try:
            import pyvista as pv
        except ImportError as exc:
            raise ImportError(
                "pyvista is required for 3D rendering. "
                "Install with: pip install pyvista"
            ) from exc
        
        plotter = pv.Plotter(off_screen=off_screen)
        
        for layer in self._layers:
            if layer.role == ROLE_BACKGROUND:
                self