"""
---------
SimnibsViz — 2D (nilearn) + 3D (NiiVue headless) visualisation class.

2D methods wrap ``nilearn.plotting`` for slice overlays.
3D method uses Playwright + NiiVue for headless WebGL rendering to PNG.
Scale can be locked across a cohort via ``set_scale_from_cohort()``.
"""

from __future__ import annotations

import http.server
import json
import shutil
import socketserver
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting

from .._logging import get_logger

logger = get_logger(__name__)


# =====================================================================
# NiiVue headless helpers (module-private)
# =====================================================================


@contextmanager
def _serve(directory: Path):
    """Spin up a throwaway HTTP server for a temp directory."""
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(
        *a, directory=str(directory), **k
    )
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            yield httpd.server_address[1]
        finally:
            httpd.shutdown()


_HTML = """\
<!doctype html><html><head><meta charset="utf-8"></head>
<body style="margin:0"><canvas id="gl" width="{w}" height="{h}"></canvas>
<script type="module">
  import {{ Niivue, SLICE_TYPE }} from "{niivue_url}";
  const nv = new Niivue({nv_opts});
  window.nv = nv;
  await nv.attachTo("gl");
  await nv.loadVolumes({volumes});
  nv.setSliceType(SLICE_TYPE.RENDER);
  nv.setRenderAzimuthElevation({azim}, {elev});
  nv.drawScene();
  requestAnimationFrame(() => requestAnimationFrame(() => {{
      window.__ready = true;
  }}));
</script></body></html>"""


# =====================================================================
# SimnibsViz
# =====================================================================


class SimnibsViz:
    """2D + 3D SimNIBS visualisation helper.

    Parameters
    ----------
    output_dir : Path
        Root output directory for all figures.
    cmap : str
        Default colormap for e-field overlays.
    threshold : float
        Default overlay threshold (voxels below this value are transparent).
    bins : int
        Number of bins for histograms.
    if_exists : str
        ``"overwrite"``, ``"skip"``, or ``"error"`` for existing files.
    """

    def __init__(
        self,
        output_dir: Path,
        cmap: str = "hot",
        threshold: float = 0.1,
        bins: int = 50,
        if_exists: str = "overwrite",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.cmap = cmap
        self.threshold = threshold
        self.bins = bins
        self.if_exists = if_exists
        self._scale: tuple[float, float] | None = None  # (vmin, vmax)

    # =================================================================
    # Scale management (cohort-level)
    # =================================================================

    def set_scale(self, vmin: float, vmax: float) -> None:
        """Lock colour scale manually."""
        self._scale = (vmin, vmax)
        logger.info(f"Scale locked: [{vmin:.4f}, {vmax:.4f}]")

    def set_scale_from_cohort(
        self,
        efields: list,
        lower_pct: float = 5.0,
        upper_pct: float = 95.0,
    ) -> tuple[float, float]:
        """Compute global [pN, pM] from a list of EField / Nifti1Image objects.

        Parameters
        ----------
        efields : list of nib.Nifti1Image (or EField subclass)
            One per subject.
        lower_pct, upper_pct : float
            Percentiles used for the colour-scale bounds.

        Returns
        -------
        (vmin, vmax)
        """
        all_vals = []
        for ef in efields:
            d = ef.get_fdata().ravel()
            all_vals.append(d[d > 0])
        pooled = np.concatenate(all_vals)
        vmin = float(np.percentile(pooled, lower_pct))
        vmax = float(np.percentile(pooled, upper_pct))
        self.set_scale(vmin, vmax)
        return vmin, vmax

    # =================================================================
    # 2D — nilearn wrappers
    # =================================================================

    def plot_anat(
        self,
        t1,
        cut_coords=None,
        display_mode: str = "ortho",
        output: str | Path | None = None,
        title: str | None = None,
    ):
        """Plot T1 anatomy (no overlay)."""
        disp = plotting.plot_anat(
            t1,
            cut_coords=cut_coords,
            display_mode=display_mode,
            title=title,
        )
        return self._finish_2d(disp, output)

    def plot_efield(
        self,
        t1,
        efield,
        cut_coords=None,
        display_mode: str = "ortho",
        cmap: str | None = None,
        threshold: float | None = None,
        output: str | Path | None = None,
        title: str | None = None,
    ):
        """Plot T1 + continuous e-field overlay."""
        cmap = cmap or self.cmap
        threshold = threshold if threshold is not None else self.threshold
        vmin, vmax = self._scale or (None, None)

        disp = plotting.plot_anat(
            t1,
            cut_coords=cut_coords,
            display_mode=display_mode,
            title=title,
        )
        disp.add_overlay(
            efield,
            cmap=cmap,
            threshold=threshold,
            vmin=vmin,
            vmax=vmax,
        )
        return self._finish_2d(disp, output)

    def plot_efield_roi(
        self,
        t1,
        efield,
        roi_mask,
        cut_coords=None,
        display_mode: str = "ortho",
        cmap: str | None = None,
        threshold: float | None = None,
        contour_color: str = "cyan",
        output: str | Path | None = None,
        title: str | None = None,
    ):
        """Plot T1 + e-field overlay + ROI contour."""
        disp = self.plot_efield(
            t1,
            efield,
            cut_coords=cut_coords,
            display_mode=display_mode,
            cmap=cmap,
            threshold=threshold,
            title=title,
        )
        disp.add_contours(roi_mask, levels=[0.5], colors=contour_color)
        return self._finish_2d(disp, output)

    def plot_mosaic(
        self,
        t1,
        efield=None,
        roi_mask=None,
        n_cuts: int = 7,
        display_mode: str = "z",
        cmap: str | None = None,
        threshold: float | None = None,
        contour_color: str = "cyan",
        output: str | Path | None = None,
        title: str | None = None,
    ):
        """Multi-slice mosaic — T1 with optional e-field and ROI contours."""
        cmap = cmap or self.cmap
        threshold = threshold if threshold is not None else self.threshold
        vmin, vmax = self._scale or (None, None)

        disp = plotting.plot_anat(
            t1,
            display_mode=display_mode,
            cut_coords=n_cuts,
            title=title,
        )
        if efield is not None:
            disp.add_overlay(
                efield,
                cmap=cmap,
                threshold=threshold,
                vmin=vmin,
                vmax=vmax,
            )
        if roi_mask is not None:
            disp.add_contours(roi_mask, levels=[0.5], colors=contour_color)
        return self._finish_2d(disp, output)

    def _finish_2d(self, disp, output):
        """Save and/or return a nilearn OrthoSlicer display."""
        if output is not None:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            disp.savefig(str(output), dpi=200)
            logger.info(f"Saved 2D: {output}")
            disp.close()
        return disp

    # =================================================================
    # 3D — NiiVue headless (Playwright + WebGL)
    # =================================================================

    def render_3d(
        self,
        volumes: list[dict],
        output: str | Path,
        azimuth: float = 120,
        elevation: float = 15,
        nv_opts: dict | None = None,
        width: int = 800,
        height: int = 600,
        niivue_url: str = "https://unpkg.com/@niivue/niivue/dist/index.js",
        timeout_ms: int = 60_000,
    ) -> Path:
        """Render a list of NiiVue volumes to a PNG file (headless).

        Parameters
        ----------
        volumes : list of dict
            Each dict must have ``"path"`` (str or Path to a NIfTI file)
            plus any NiiVue keys: ``"colormap"``, ``"opacity"``,
            ``"cal_min"``, ``"cal_max"``, etc.
        output : Path
            Destination PNG path.
        azimuth, elevation : float
            Camera angles.
        nv_opts : dict or None
            NiiVue scene options.  Defaults to dark background + colorbar.

        Notes
        -----
        If ``self._scale`` is set, ``cal_min`` / ``cal_max`` are
        auto-injected on non-gray volumes (unless already specified).
        """
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        nv_opts = nv_opts or {
            "backColor": [0.1, 0.1, 0.1, 1],
            "isColorbar": True,
        }

        # Auto-inject cohort scale on overlay (non-gray) volumes
        if self._scale:
            vmin, vmax = self._scale
            volumes = [
                (
                    {
                        **v,
                        "cal_min": v.get("cal_min", vmin),
                        "cal_max": v.get("cal_max", vmax),
                    }
                    if v.get("colormap", "gray") != "gray"
                    else v
                )
                for v in volumes
            ]

        from playwright.sync_api import sync_playwright

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            served = []
            for i, vol in enumerate(volumes):
                v = dict(vol)
                src = v.pop("path", None)
                if src is None:
                    raise ValueError(f"volume {i}: key 'path' is required.")
                fname = f"vol{i}_{Path(src).name}"
                shutil.copy(src, tmp / fname)
                v["url"] = fname
                served.append(v)

            (tmp / "index.html").write_text(
                _HTML.format(
                    w=width,
                    h=height,
                    niivue_url=niivue_url,
                    nv_opts=json.dumps(nv_opts),
                    volumes=json.dumps(served),
                    azim=azimuth,
                    elev=elevation,
                )
            )

            with _serve(tmp) as port:
                with sync_playwright() as p:
                    browser = p.chromium.launch(
                        headless=True,
                        args=["--use-gl=angle", "--use-angle=swiftshader"],
                    )
                    page = browser.new_page(
                        viewport={"width": width, "height": height},
                    )
                    page.goto(f"http://127.0.0.1:{port}/index.html")
                    page.wait_for_function(
                        "window.__ready === true",
                        timeout=timeout_ms,
                    )
                    with page.expect_download() as dl:
                        page.evaluate("nv.saveScene('scene.png')")
                    dl.value.save_as(str(output))
                    browser.close()

        logger.info(f"Saved 3D: {output}")
        return output

    # =================================================================
    # Statistical visualisations
    # =================================================================

    def efields_histograms(
        self,
        data_by_subject: Dict[str, List[Tuple[str, str, Path, Path]]],
        region: str = "intra",
        space: str = "mni",
    ) -> None:
        """One histogram figure per subject: masked vs cleaned e-fields.

        Parameters
        ----------
        data_by_subject :
            ``subject → [(roi, mode, masked_path, cleaned_path), ...]``
        region :
            Label for title/filename (``"intra"`` or ``"extra"``).
        space :
            ``"mni"`` or ``"native"`` — filename suffix.
        """
        output_dir = self.output_dir / "2-preprocess"
        output_dir.mkdir(parents=True, exist_ok=True)
        # tag = space_tag(space)

        for subject, subject_data in data_by_subject.items():
            logger.info(f"{region}-ROI histograms for {subject}")
            n_plots = len(subject_data)
            n_cols = min(3, n_plots)
            n_rows = (n_plots + n_cols - 1) // n_cols

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(5 * n_cols, 4 * n_rows),
            )
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
            # save_figure(
            #     out_path, if_exists=self.if_exists, dpi=300, bbox_inches="tight",
            # )
            # logger.info(f"  Saved: {out_path}")

        logger.info(f"All histograms saved in {output_dir}")

    def plot_simulation_vs_optimization(
        self,
        df: pd.DataFrame,
        metric: str = "mean",
        subject_col: str = "subject",
        condition_col: str = "condition",
        output_tag: str = "",
    ) -> None:
        """Scatter: simulation (x) vs optimisation (y) per subject/ROI."""
        output_dir = self.output_dir / "3-analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        df = df.copy()
        df["type"] = df[condition_col].apply(
            lambda x: "optimization" if "optimization" in x else "simulation"
        )
        df["roi"] = df[condition_col].apply(
            lambda x: x.replace("_simulation", "").replace("_optimization", "")
        )

        r
