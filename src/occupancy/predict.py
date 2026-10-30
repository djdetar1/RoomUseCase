"""Load the saved model and score a single reading.

Used by both the API and the tests so the inference path is the same everywhere.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from .config import MODEL_PATH, NUMERIC_FEATURES
from .features import add_datetime_features


@lru_cache(maxsize=1)
def load_model(path: str | None = None):
    model_path = Path(path) if path else MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train first: python -m occupancy.train"
        )
    return joblib.load(model_path)


def _to_frame(record: dict) -> pd.DataFrame:
    missing = [c for c in ["date", *NUMERIC_FEATURES] if c not in record]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    return pd.DataFrame([record])


def predict_one(record: dict, model=None) -> dict:
    """Score one reading. Requires date, Temperature, Humidity, Light, CO2, HumidityRatio."""
    model = model or load_model()
    df = add_datetime_features(_to_frame(record))
    proba = float(model.predict_proba(df)[:, 1][0])
    return {"occupancy": int(proba >= 0.5), "probability": proba}
