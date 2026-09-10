"""
Pipeline I/O helpers.

Small, dependency-free helpers for reading and writing the tabular artefacts
produced by the analysis pipeline. Kept deliberately minimal: NIfTI I/O lives
in ``simnibs-reader``, not here.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ._logging import get_logger

logger = get_logger(__name__)


def load_csvs(csv_paths: Iterable[Path]) -> pd.DataFrame:
    """Load and concatenate multiple CSV files into a single DataFrame.

    Parameters
    ----------
    csv_paths : Iterable[Path]
        Iterable of paths to CSV files.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame from all CSV files.
    """
    frames = [pd.read_csv(p) for p in csv_paths]
    if not frames:
        raise ValueError("load_csvs: no CSV path provided")
    return pd.concat(frames, ignore_index=True)


def check_output(path: Path, if_exists: str = "overwrite") -> bool:
    """Return True if the file should be written, False if it should be skipped.

    Parameters
    ----------
    path : Path
        Target output path.
    if_exists : str
        ``'overwrite'`` — always write (default).
        ``'skip'``      — silently skip if the file exists.
        ``'error'``     — raise FileExistsError if the file exists.

    Returns
    -------
    bool
        Whether the caller should proceed with writing.
    """
    path = Path(path)
    if path.exists():
        if if_exists == "skip":
            logger.info(f"Skip (already exists): {path.name}")
            return False
        if if_exists == "error":
            msg = f"Output already exists (if_exists='error'): {path}"
            logger.error(msg)
            raise FileExistsError(msg)
    return True


def save_dataframe(
    df: pd.DataFrame,
    out_path: Path,
    if_exists: str = "overwrite",
    **to_csv_kwargs,
) -> None:
    """Save a DataFrame to CSV, creating parent directories as needed.

    Parameters
    ----------
    df : pd.DataFrame
        Table to write.
    out_path : Path
        Destination CSV path.
    if_exists : str
        Passed to :func:`check_output`.
    **to_csv_kwargs
        Forwarded to :meth:`pandas.DataFrame.to_csv`.
    """
    out_path = Path(out_path)
    if not check_output(out_path, if_exists):
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, **to_csv_kwargs)
