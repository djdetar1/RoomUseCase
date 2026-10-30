"""Loading and splitting the occupancy CSV data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATETIME_COLUMN, TARGET


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read an occupancy CSV, dropping the unnamed row-id column and parsing dates."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, index_col=0).reset_index(drop=True)

    if DATETIME_COLUMN not in df.columns:
        raise ValueError(f"Expected '{DATETIME_COLUMN}' column in {path}")

    df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN])
    return df


def split_xy(df: pd.DataFrame):
    """Split into features X and the Occupancy target y."""
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' missing")
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    return X, y
