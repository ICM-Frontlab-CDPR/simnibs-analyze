"""
Meta-analysis: compare intra-ROI mean e-field across spaces and ROI methods.

Inputs : one or more  all_features_space-<X>.csv  files produced by run.py.
Outputs: saved in  <results_dir>/meta_analysis/

Usage (CLI):
    python meta.py --results-dir /path/to/results-V3 --metric mean

Two comparisons are implemented:
  1. space_comparison   — same condition, mni vs native
  2. roi_method_comparison — same target zone, different ROI definition
                             (requires explicit pairs in --roi-pairs)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from _pipeline_io import get_features_csv_path, save_dataframe, save_figure
from _logging import get_logger

logger = get_logger(__name__)

SPACES = ("mni", "native")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_all_features(results_dir: Path, spaces: tuple = SPACES) -> pd.DataFrame:
    """Load and concatenate all_features CSVs found in *results_dir*/analysis/."""
    frames = []
    for space in spaces:
        path = get_features_csv_path(results_dir, space)
        if path.exists():
            df = pd.read_csv(path)
            df["space"] = space  # ensure column present even if missing
            frames.append(df)
            logger.info(f"Loaded {len(df)} rows from {path.name}")
        else:
            logger.info(f"Not found (skipped): {path.name}")
    if not frames:
        raise FileNotFoundError(
            f"No all_features_*.csv found in {results_dir}/analysis/"
        )
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Comparison 1 — space (mni vs native)
# ---------------------------------------------------------------------------


def compare_spaces(
    df: pd.DataFrame,
    metric: str = "mean",
    condition_col: str = "condition",
) -> pd.DataFrame:
    """
    For each condition, compare *metric* between mni and native space.

    Returns a wide DataFrame:
        condition | mean_mni | std_mni | n_mni | mean_native | std_native | n_native | delta_mean
    """
    if "space" not in df.columns:
        raise KeyError("Column 'space' missing — load CSVs from multiple spaces.")

    rows = []
    for cond, grp in df.groupby(condition_col):
        row: dict = {"condition": cond}
        for space in ("mni", "native"):
            sub = grp[grp["space"] == space][metric].dropna()
            row[f"mean_{space}"] = sub.mean()
            row[f"std_{space}"] = sub.std()
            row[f"n_{space}"] = len(sub)
        row["delta_mean"] = row["mean_mni"] - row["mean_native"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_space_comparison(
    summary: pd.DataFrame,
    metric: str = "mean",
    out_path: Optional[Path] = None,
) -> None:
    """Bar chart: mni vs native mean e-field per condition."""
    conditions = summary["condition"].tolist()
    x = range(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(conditions) * 1.5), 5))
    ax.bar(
        [i - width / 2 for i in x],
        summary["mean_mni"],
        width,
        yerr=summary["std_mni"],
        label="MNI",
        capsize=4,
        alpha=0.8,
    )
    ax.bar(
        [i + width / 2 for i in x],
        summary["mean_native"],
        width,
        yerr=summary["std_native"],
        label="Native",
        capsize=4,
        alpha=0.8,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_ylabel(f"Intra-ROI {metric} e-field (V/m)")
    ax.set_title("Space comparison: MNI vs Native")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if out_path:
        save_figure(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Comparison 2 — ROI method
# ---------------------------------------------------------------------------


def compare_roi_methods(
    df: pd.DataFrame,
    roi_pairs: List[tuple],
    metric: str = "mean",
    condition_col: str = "condition",
) -> pd.DataFrame:
    """
    Compare *metric* between pairs of conditions targeting the same brain zone
    but using different ROI definitions.

    Parameters
    ----------
    roi_pairs : list of (cond_a, cond_b) tuples
        e.g. [("fef_simulation", "HA-fef_simulation"),
               ("fef_simulation", "AAL-fef_simulation")]

    Returns a DataFrame with one row per pair:
        cond_a | cond_b | mean_a | std_a | n_a | mean_b | std_b | n_b | delta_mean
    """
    rows = []
    for cond_a, cond_b in roi_pairs:
        sub_a = df[df[condition_col] == cond_a][metric].dropna()
        sub_b = df[df[condition_col] == cond_b][metric].dropna()
        rows.append(
            {
                "cond_a": cond_a,
                "cond_b": cond_b,
                "mean_a": sub_a.mean(),
                "std_a": sub_a.std(),
                "n_a": len(sub_a),
                "mean_b": sub_b.mean(),
                "std_b": sub_b.std(),
                "n_b": len(sub_b),
                "delta_mean": sub_a.mean() - sub_b.mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_roi_method_comparison(
    summary: pd.DataFrame,
    metric: str = "mean",
    out_path: Optional[Path] = None,
) -> None:
    """Bar chart: condition A vs B for each pair."""
    n = len(summary)
    fig, ax = plt.subplots(figsize=(max(6, n * 2), 5))
    x = range(n)
    width = 0.35

    ax.bar(
        [i - width / 2 for i in x],
        summary["mean_a"],
        width,
        yerr=summary["std_a"],
        label="ROI A",
        capsize=4,
        alpha=0.8,
    )
    ax.bar(
        [i + width / 2 for i in x],
        summary["mean_b"],
        width,
        yerr=summary["std_b"],
        label="ROI B",
        capsize=4,
        alpha=0.8,
    )
    labels = [f"{r.cond_a}\nvs\n{r.cond_b}" for _, r in summary.iterrows()]
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(f"Intra-ROI {metric} e-field (V/m)")
    ax.set_title("ROI method comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if out_path:
        save_figure(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    results_dir: Path,
    metric: str = "mean",
    roi_pairs: Optional[List[tuple]] = None,
    spaces: tuple = SPACES,
) -> None:
    """Run all meta-analyses and save outputs to <results_dir>/meta_analysis/."""
    out_dir = results_dir / "meta_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_features(results_dir, spaces=spaces)

    # ── Space comparison ────────────────────────────────────────────────────
    n_spaces = df["space"].nunique()
    if n_spaces >= 2:
        space_summary = compare_spaces(df, metric=metric)
        save_dataframe(
            space_summary, out_dir / f"space_comparison_{metric}.csv", index=False
        )
        plot_space_comparison(
            space_summary,
            metric=metric,
            out_path=out_dir / f"space_comparison_{metric}.png",
        )
        logger.info(f"Space comparison saved → {out_dir}")
    else:
        logger.info(f"Space comparison skipped — only {df['space'].unique()} found.")

    # ── ROI method comparison ───────────────────────────────────────────────
    if roi_pairs:
        roi_summary = compare_roi_methods(df, roi_pairs=roi_pairs, metric=metric)
        save_dataframe(
            roi_summary, out_dir / f"roi_method_comparison_{metric}.csv", index=False
        )
        plot_roi_method_comparison(
            roi_summary,
            metric=metric,
            out_path=out_dir / f"roi_method_comparison_{metric}.png",
        )
        logger.info(f"ROI method comparison saved → {out_dir}")
    else:
        logger.info("ROI method comparison skipped — no --roi-pairs provided.")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Meta-analysis across spaces and ROI methods."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Root results directory (contains analysis/)",
    )
    parser.add_argument(
        "--metric", default="mean", help="Feature column to compare (default: mean)"
    )
    parser.add_argument(
        "--spaces",
        nargs="+",
        default=list(SPACES),
        help="Spaces to load (default: mni native)",
    )
    parser.add_argument(
        "--roi-pairs",
        nargs="+",
        default=[],
        metavar="A:B",
        help="Pairs of conditions to compare as ROI methods, e.g. "
        "fef_simulation:HA-fef_simulation fef_simulation:AAL-fef_simulation",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    pairs = [tuple(p.split(":")) for p in args.roi_pairs]
    run(
        results_dir=args.results_dir,
        metric=args.metric,
        spaces=tuple(args.spaces),
        roi_pairs=pairs or None,
    )
