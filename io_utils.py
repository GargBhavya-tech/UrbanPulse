"""IO helpers for loading the raw CSV and reading/writing parquet artifacts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config


def load_raw(path: Path = config.RAW_CSV) -> pd.DataFrame:
    """Load the raw Pangyo sensor CSV.

    Args:
        path: Path to the raw CSV. Defaults to ``data/raw.csv``.

    Returns:
        The raw DataFrame with no transformations applied.

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {path}")
    return pd.read_csv(path)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet, creating parent dirs as needed.

    Args:
        df: DataFrame to persist.
        path: Destination ``.parquet`` path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet artifact.

    Args:
        path: Source ``.parquet`` path.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the parquet file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")
    return pd.read_parquet(path)
