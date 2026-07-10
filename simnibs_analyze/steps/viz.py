"""
viz.py
------
SimnibsViz — 2D (nilearn) + 3D (NiiVue headless) visualisation class.

2D methods wrap ``nilearn.plotting`` for slice overlays.
3D method uses Playwright + NiiVue for headless WebGL rendering to PNG.
Scale can be locked across a cohort via ``set_scale_from_cohort()``.
Cohort montages are composed with ``plot_cohort_montage()`` (single shared scale).
"""

from __future__ import annotations

import http.server
import json
import math
import shutil
import socketserver
import tempfile
import threading
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from nilearn import plotting

from .._logging import get_logger

logger = get_logger(__name__)


# =====================================================================
# NiiVue headless helpers (module-private)
# =====================================================================


@contextmanager
def _serve(directory: Path):
    """Spin up a throwaway HTTP server for a temp directory."""
    # partial instead of a lambda (E731) — same effect, no bound-name warning
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
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
  const electrodes = {electrodes};
  if (electrodes.length) {{
    try {{
      nv.loadConnectome({{
        name: "electrodes",
        nodeColormap: "warm",
        nodeColormapNegative: "",
        nodeMinColor: 0, nodeMaxColor: 1,
        nodeScale: 3,
        nodes: electrodes.map(e => ({{
          name: e.label, x: e.x, y: e.y, z: e.z,
          colorValue: 1.0, sizeValue: e.size !== undefined ? e.size : 4.0
        }})),
        edges: []
      }});
    }} catch(err) {{
      console.warn("electrodes (loadConnectome) failed:", err);
    }}
  }}
  nv.drawScene();
  setTimeout(() => {{ window.__ready = true; }}, 500);
</script></body></html>"""


def _is_dark(rgba) -> bool:
    """Perceived-luminance test for choosing black vs white text."""
    r, g, b = rgba[:3]
    return (0.299 * r + 0.587 * g + 0.114 * b) < 0.5


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
    bg_color : str or tuple
        Figure background, as a cohort-wide aesthetic. Any matplotlib colour
        (``"black"``, ``"white"``, ``"#101010"``, ``(0.1, 0.1, 0.1)``).
        It drives both the 2D figures (``black_bg`` + facecolor) and the 3D
        NiiVue scene (``backColor``), so a single setting keeps them coherent.
    """

    def __init__(
        self,
        output_dir: Path,
        cmap: str = "hot",
        threshold: float = 0.1,
        bins: int = 50,
        if_exists: str = "overwrite",
        bg_color: str | tuple = (0.1, 0.1, 0.1),
    ) -> None:
        self.output_dir = Path(output_dir)
        self.cmap = cmap
        self.threshold = threshold
        self.bins = bins
        self.if_exists = if_exists
        self._scale: tuple[float, float] | None = None  # (vmin, vmax)

        # Background: one source of truth → derive 2D + 3D forms
        self.bg_color = bg_color
        self._bg_rgba = mcolors.to_rgba(bg_color)  # 3D NiiVue backColor
        self._black_bg = _is_dark(self._bg_rgba)  # 2D nilearn black_bg

    # =================================================================
    # Background helpers
    # =================================================================

    def set_background(self, bg_color: str | tuple) -> None:
        """Change the cohort-wide background colour (2D + 3D at once)."""
        self.bg_color = bg_color
        self._bg_rgba = mcolors.to_rgba(bg_color)
        self._black_bg = _is_dark(self._bg_rgba)

    @property
    def _fg_color(self) -> str:
        """Contrasting text/title colour for the current background."""
        return "white" if self._black_bg else "black"

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
        alpha: float = 0.0,
    ) -> tuple[float, float]:
        """Compute global [vmin, vmax] from a list of EField / Nifti1Image objects.

        Parameters
        ----------
        efields : list of nib.Nifti1Image (or EField)
            One per subject.
        lower_pct, upper_pct : float
            Percentiles used for the raw colour-scale bounds.
        alpha : float
            Symmetric widening factor applied to the [lo, hi] window *after*
            the percentile computation. ``margin = alpha * (hi - lo)``:
            0.0 → unchanged, 0.5 → width ×2, 1.0 → width ×3, <0 → tightened.
            ``vmin`` is clamped to 0 (magnE ≥ 0).

        Returns
        -------
        (vmin, vmax)
        """
        all_vals = []
        for ef in efields:
            d = np.asarray(ef.get_fdata()).ravel()
            all_vals.append(d[d > 0])
        pooled = np.concatenate(all_vals)

        lo = float(np.percentile(pooled, lower_pct))
        hi = float(np.percentile(pooled, upper_pct))

        margin = alpha * (hi - lo)
        vmin = max(0.0, lo - margin)  # magnE ≥ 0 → clamp
        vmax = hi + margin

        if vmin >= vmax:
            raise ValueError(
                f"Échelle dégénérée (vmin={vmin:.4f} ≥ vmax={vmax:.4f}). "
                f"alpha={alpha} resserre trop la fenêtre [{lo:.4f}, {hi:.4f}] "
                "(alpha ≤ -0.5 la fait s'effondrer). Utilise alpha ∈ ]-0.5, 0], "
                "ou resserre plutôt via lower_pct / upper_pct."
            )

        self.set_scale(vmin, vmax)
        return vmin, vmax

    # =================================================================
    # 2D — nilearn wrappers
    # =================================================================

    _CMAP_ALIASES = {"blue": "Blues", "red": "Reds", "green": "Greens"}
    _CMAP_TO_CONTOUR_COLOR = {"Blues": "blue", "Reds": "red", "Greens": "green"}

    @classmethod
    def _cmap(cls, name: str) -> str:
        return cls._CMAP_ALIASES.get(name, name)

    @classmethod
    def _contour_color(cls, name: str) -> str:
        """Map a colormap name to a single contour color string."""
        resolved = cls._cmap(name)
        return cls._CMAP_TO_CONTOUR_COLOR.get(resolved, name)

    @staticmethod
    def _as_niimg(src):
        """str/Path → left as-is (nilearn loads) ; EField → .img ; niimg → as-is."""
        if hasattr(src, "img"):
            return src.img
        return src

    @staticmethod
    def _overlay_transparency(disp, img, **kw):
        """add_overlay with the new `transparency=` kwarg, falling back to `alpha=`."""
        opacity = kw.pop("opacity", 1.0)
        try:
            disp.add_overlay(img, transparency=opacity, **kw)
        except TypeError:  # nilearn < 0.12
            disp.add_overlay(img, alpha=opacity, **kw)

    def plot_anat(
        self,
        t1_or_vols,  # niimg/Path/EField  OR  list[dict]
        cut_coords=None,
        display_mode: str = "ortho",
        output: str | Path | None = None,
        title: str | None = None,
    ):
        """Plot T1 anatomy. Accepts a single image OR a NiiVue-style vols list.

        vols format (first = background, rest = overlays)::

            [{"path": ..., "colormap": ..., "opacity": ...}, ...]
        """
        if isinstance(t1_or_vols, list):
            bg, *overlays = t1_or_vols
            disp = plotting.plot_anat(
                self._as_niimg(bg["path"]),
                cut_coords=cut_coords,
                display_mode=display_mode,
                title=title,
                black_bg=self._black_bg,
            )
            for ov in overlays:
                img = self._as_niimg(ov["path"])
                if ov.get("render") == "contour":
                    disp.add_contours(
                        img,
                        levels=[0.5],
                        colors=[self._contour_color(ov.get("colormap", "blue"))],
                        linewidths=2.0,
                    )
                else:
                    self._overlay_transparency(
                        disp,
                        img,
                        cmap=self._cmap(ov.get("colormap", "hot")),
                        opacity=ov.get("opacity", 1.0),
                        threshold=ov.get("threshold", 0.0),
                        vmin=ov.get("cal_min"),
                        vmax=ov.get("cal_max"),
                    )
        else:
            disp = plotting.plot_anat(
                self._as_niimg(t1_or_vols),
                cut_coords=cut_coords,
                display_mode=display_mode,
                title=title,
                black_bg=self._black_bg,
            )
        return self._finish_2d(disp, output)

    def _finish_2d(self, disp, output):
        """Save and/or return a nilearn OrthoSlicer display."""
        if output is not None:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            # best-effort: paint the whole figure canvas with bg_color
            try:
                disp.frame_axes.figure.set_facecolor(self.bg_color)
            except Exception:  # noqa: BLE001 — display internals vary by version
                pass
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
        colorbar: bool = True,
        width: int = 800,
        height: int = 600,
        niivue_url: str = "https://unpkg.com/@niivue/niivue/dist/index.js",
        timeout_ms: int = 30_000,
        electrodes: list[dict] | None = None,  # ← nouveau
        electrode_radius: float = 4.0,
    ) -> Path:
        """Render a list of NiiVue volumes to a PNG file (headless).

        Parameters
        ----------
        volumes : list of dict
            Each dict must have ``"path"`` (str/Path to a NIfTI file) plus any
            NiiVue keys: ``"colormap"``, ``"opacity"``, ``"cal_min"``,
            ``"cal_max"``, etc.
        output : Path
            Destination PNG path.
        azimuth, elevation : float
            Camera angles.
        nv_opts : dict or None
            NiiVue scene options. Defaults to the class ``bg_color`` background
            + a colorbar toggled by ``colorbar``.
        colorbar : bool
            Whether NiiVue draws its own colorbar. Set ``False`` for per-subject
            renders that will be tiled by :meth:`plot_cohort_montage` (which
            adds a single shared colorbar instead).

        Notes
        -----
        If ``self._scale`` is set, ``cal_min`` / ``cal_max`` are auto-injected
        on non-gray volumes (unless already specified).
        """
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        nv_opts = nv_opts or {
            "backColor": list(self._bg_rgba),
            "isColorbar": colorbar,
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
                    electrodes=json.dumps(
                        [
                            {
                                "label": e.get("label", ""),
                                "x": e["x"],
                                "y": e["y"],
                                "z": e["z"],
                                "size": e.get("size", electrode_radius),
                            }
                            for e in (electrodes or [])
                        ]
                    ),
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
                    page.on("console", lambda msg: print("JS:", msg.text))
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
    # Cohort composition (one figure, one shared scale)
    # =================================================================

    def plot_cohort_montage(
        self,
        panels: Sequence[dict],
        output: str | Path,
        ncols: int | None = None,
        title: str | None = None,
        add_colorbar: bool = True,
        cbar_label: str = "E-field (V/m)",
        panel_h: float = 4.0,
        panel_title_size: int = 11,
        dpi: int = 200,
    ) -> Path:
        """Tile per-subject figures into ONE cohort montage with a single scale.

        Feed it the PNGs produced by the per-subject calls (``render_3d`` /
        ``plot_*``). The colour scale is drawn **once** as a shared colorbar
        derived from the locked cohort scale (``self._scale``) — so render the
        per-subject panels with ``colorbar=False`` to avoid a scale per subject.

        Parameters
        ----------
        panels : sequence of dict
            One dict per subject: ``{"label": "0001", "image": <png path>}``.
            ``label`` is the subject number shown above each panel.
        output : Path
            Destination PNG.
        ncols : int or None
            Columns in the grid. Defaults to ``ceil(sqrt(n))``.
        title : str or None
            Figure-level suptitle.
        add_colorbar : bool
            Draw the single shared colorbar (requires ``self._scale`` set).
        cbar_label : str
            Colorbar axis label.

        Returns
        -------
        Path
        """
        panels = list(panels)
        if not panels:
            raise ValueError("plot_cohort_montage: `panels` is empty.")

        n = len(panels)
        ncols = ncols or math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)

        # Auto-size panels from the first image's pixel aspect ratio so that
        # wide parallel-slice strips aren't squeezed into square cells.
        first_img = plt.imread(str(panels[0]["image"]))
        ph_px, pw_px = first_img.shape[:2]
        panel_h_in = panel_h
        panel_w_in = panel_h_in * (pw_px / ph_px)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(panel_w_in * ncols, panel_h_in * nrows),
            squeeze=False,
        )
        fig.set_facecolor(self.bg_color)

        for idx in range(nrows * ncols):
            ax = axes[idx // ncols][idx % ncols]
            ax.set_facecolor(self.bg_color)
            ax.axis("off")
            if idx >= n:
                continue
            panel = panels[idx]
            img = plt.imread(str(panel["image"])) if idx > 0 else first_img
            ax.imshow(img)
            label = str(panel.get("label", idx))
            ax.set_title(label, fontsize=panel_title_size, color=self._fg_color)

        # ── single shared colorbar ──────────────────────────────────
        if add_colorbar:
            if self._scale is None:
                logger.warning(
                    "plot_cohort_montage: no locked scale — call "
                    "set_scale_from_cohort() first, or pass add_colorbar=False."
                )
            else:
                vmin, vmax = self._scale
                sm = ScalarMappable(
                    norm=Normalize(vmin=vmin, vmax=vmax),
                    cmap=self._cmap(self.cmap),
                )
                sm.set_array([])
                # dedicated axis on the right → one bar for the whole grid
                cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
                cbar = fig.colorbar(sm, cax=cax)
                cbar.set_label(cbar_label, color=self._fg_color)
                cbar.ax.yaxis.set_tick_params(color=self._fg_color)
                plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=self._fg_color)

        if title:
            fig.suptitle(title, fontsize=16, fontweight="bold", color=self._fg_color)

        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        # rect leaves room for the colorbar on the right
        fig.tight_layout(rect=(0, 0, 0.9 if add_colorbar else 1.0, 0.97))
        fig.savefig(str(output), dpi=dpi, facecolor=self.bg_color, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved cohort montage: {output}")
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
        tag = space  # filename suffix (was an undefined `space_tag(space)`)

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
            fig.tight_layout()

            out_path = output_dir / f"efields_histograms_{subject}_{region}_{tag}.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=self.bg_color)
            plt.close(fig)
            logger.info(f"  Saved: {out_path}")

        logger.info(f"All histograms saved in {output_dir}")

    def plot_simulation_vs_optimization(
        self,
        df: pd.DataFrame,
        metric: str = "mean",
        subject_col: str = "subject",
        condition_col: str = "condition",
        output_tag: str = "",
    ) -> Path | None:
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

        wide = df.pivot_table(
            index=[subject_col, "roi"], columns="type", values=metric
        ).reset_index()

        if "simulation" not in wide or "optimization" not in wide:
            logger.warning(
                "plot_simulation_vs_optimization: need both 'simulation' and "
                "'optimization' rows; nothing to plot."
            )
            return None

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.set_facecolor(self.bg_color)
        for roi, sub in wide.groupby("roi"):
            ax.scatter(
                sub["simulation"], sub["optimization"], label=str(roi), alpha=0.7
            )

        hi = float(np.nanmax([wide["simulation"].max(), wide["optimization"].max()]))
        lims = [0.0, hi * 1.05]
        ax.plot(lims, lims, "--", color="gray", lw=1)  # y = x reference
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Simulation ({metric})")
        ax.set_ylabel(f"Optimization ({metric})")
        ax.set_title("Simulation vs Optimization")
        ax.legend(fontsize=8)

        suffix = f"_{output_tag}" if output_tag else ""
        out_path = output_dir / f"sim_vs_opt_{metric}{suffix}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=self.bg_color)
        plt.close(fig)
        logger.info(f"Saved: {out_path}")
        return out_path
