"""Paths, feature definitions, and training settings.

Paths can be overridden with environment variables so the same code runs
locally, in CI, and in a container.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODEL_DIR / "occupancy_model.joblib"))

TRAIN_FILE = Path(os.getenv("TRAIN_FILE", DATA_DIR / "datatraining.csv"))
TEST_FILES = [
    DATA_DIR / "datatest.csv",
    DATA_DIR / "datatest2.csv",
]

TARGET = "Occupancy"
NUMERIC_FEATURES = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
DATETIME_CATEGORICAL = ["hour", "weekday"]
DATETIME_COLUMN = "date"


@dataclass
class TrainingConfig:
    random_state: int = 42
    n_iter: int = 20
    cv_folds: int = 3
    scoring: str = "roc_auc"
    n_jobs: int = -1
    param_distributions: dict = field(
        default_factory=lambda: {
            "clf__n_estimators": [100, 200, 300, 400, 500],
            "clf__max_depth": [3, 4, 5, 6, 8],
            "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "clf__subsample": [0.6, 0.8, 1.0],
            "clf__colsample_bytree": [0.6, 0.8, 1.0],
            "clf__min_child_weight": [1, 3, 5],
        }
    )


TRAINING_CONFIG = TrainingConfig()
