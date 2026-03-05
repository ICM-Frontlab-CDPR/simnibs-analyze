"""
Inter/Intra-subject analysis from per-subject CSVs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from _io import load_csvs


class Analysis:
    """Inter/intra-subject analysis from a features DataFrame."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    # -------------------------------------------------------------------------
    # Inter-subject
    # -------------------------------------------------------------------------

    def inter_subject_summary(
        self,
        metric: str = "mean",
        condition_col: str = "condition",
    ) -> pd.DataFrame:
        """Compute inter-subject summary statistics (mean, std, sem) by condition."""
        if condition_col not in self.df.columns:
            raise KeyError(f"Missing column: {condition_col}")
        if metric not in self.df.columns:
            raise KeyError(f"Missing metric column: {metric}")
        summary = (
            self.df.groupby(condition_col)[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        summary["sem"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
        return summary

    # -------------------------------------------------------------------------
    # Intra-subject
    # -------------------------------------------------------------------------

    def intra_subject_diff(
        self,
        metric: str = "mean",
        subject_col: str = "subject",
        condition_col: str = "condition",
        cond_a: str = "simu",
        cond_b: str = "opti",
    ) -> pd.DataFrame:
        """Compute intra-subject differences (cond_b - cond_a) for each subject."""
        for col in [subject_col, condition_col, metric]:
            if col not in self.df.columns:
                raise KeyError(f"Missing column: {col}")
        pivot = self.df.pivot_table(index=subject_col, columns=condition_col, values=metric)
        if cond_a not in pivot.columns or cond_b not in pivot.columns:
            raise KeyError(f"Missing conditions: {cond_a} or {cond_b}")
        diff = pivot[cond_b] - pivot[cond_a]
        return diff.reset_index().rename(columns={0: "diff", cond_b: cond_b, cond_a: cond_a})

    # -------------------------------------------------------------------------
    # E-field clustering
    # -------------------------------------------------------------------------

    def assign_clusters(
        self,
        method: str = "mean",
        specificity_threshold: float = 1.5,
        intensity_col: str = "mean",
    ) -> pd.DataFrame:
        """
        Assign a cluster label to each row based on pre-computed ratio columns.

        Requires ``efield_ratio_<method>`` column produced by
        ``compute_efield_ratio`` during feature extraction.

        Parameters
        ----------
        method :
            Method suffix of the ratio column to use (e.g. ``"mean"``
            → reads ``"efield_ratio_mean"``).
        specificity_threshold :
            Ratio above which a simulation is considered *specific*.
        intensity_col :
            Column used to determine high/low intensity level.
            The threshold is the group-level median of this column.

        Returns
        -------
        pd.DataFrame
            Copy of ``self.df`` with an added ``cluster`` column containing
            one of: ``specific_high``, ``specific_low``,
            ``diffuse_high``, ``diffuse_low``.
        """
        ratio_col = f"efield_ratio_{method}"
        for col in [ratio_col, intensity_col]:
            if col not in self.df.columns:
                raise KeyError(
                    f"Missing column '{col}'. "
                    f"Make sure compute_efield_ratio was called during feature extraction."
                )

        df = self.df.copy()
        intensity_threshold = df[intensity_col].median()

        specificity = np.where(df[ratio_col] > specificity_threshold, "specific", "diffuse")
        intensity = np.where(df[intensity_col] > intensity_threshold, "high", "low")
        df["cluster"] = [f"{s}_{i}" for s, i in zip(specificity, intensity)]
        return df


    def correlate_with():
        # compute statistics between stim success and others variables (patient response / anatomic variables)
        pass


def _parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description="Run inter/intra-subject analysis")
    parser.add_argument("--inputs", nargs="+", required=True, help="CSV files")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--metric", default="mean", help="Metric column to analyze")
    parser.add_argument("--subject-col", default="subject")
    parser.add_argument("--condition-col", default="condition")
    parser.add_argument("--cond-a", default="simu")
    parser.add_argument("--cond-b", default="opti")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    analysis = Analysis(load_csvs([Path(p) for p in args.inputs]))

    inter = analysis.inter_subject_summary(metric=args.metric, condition_col=args.condition_col)
    inter_csv = out_dir / "inter_subject_summary.csv"
    inter_csv.parent.mkdir(parents=True, exist_ok=True)
    inter.to_csv(inter_csv, index=False)

    diff_df = analysis.intra_subject_diff(
        metric=args.metric,
        subject_col=args.subject_col,
        condition_col=args.condition_col,
        cond_a=args.cond_a,
        cond_b=args.cond_b,
    )
    diff_csv = out_dir / "intra_subject_diff.csv"
    diff_df.to_csv(diff_csv, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

