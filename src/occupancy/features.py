"""Feature engineering: datetime categoricals + numeric sensor passthrough."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import DATETIME_CATEGORICAL, DATETIME_COLUMN, NUMERIC_FEATURES


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour (0-23) and weekday (0=Mon) columns derived from the timestamp."""
    out = df.copy()
    ts = pd.to_datetime(out[DATETIME_COLUMN])
    out["hour"] = ts.dt.hour
    out["weekday"] = ts.dt.day_of_week
    return out


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode the datetime categoricals, pass numeric features through."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), DATETIME_CATEGORICAL),
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def make_feature_pipeline(estimator) -> Pipeline:
    """Preprocessing + estimator as one Pipeline.

    add_datetime_features must run before fitting since it adds the hour/weekday
    columns the preprocessor expects.
    """
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("clf", estimator),
        ]
    )
