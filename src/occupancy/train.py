"""Train the occupancy model: tune an XGBoost pipeline, evaluate, and save it.

    python -m occupancy.train
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from .config import MODEL_PATH, TEST_FILES, TRAIN_FILE, TRAINING_CONFIG
from .data import load_raw, split_xy
from .features import add_datetime_features, make_feature_pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("occupancy.train")


def build_search() -> RandomizedSearchCV:
    cfg = TRAINING_CONFIG
    estimator = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        tree_method="hist",
    )
    pipeline = make_feature_pipeline(estimator)
    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=cfg.param_distributions,
        n_iter=cfg.n_iter,
        scoring=cfg.scoring,
        cv=cfg.cv_folds,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
        refit=True,
        verbose=1,
    )


def evaluate(model, path) -> float:
    df = add_datetime_features(load_raw(path))
    X, y = split_xy(df)
    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    logger.info("ROC-AUC on %s: %.4f", Path(path).name, auc)
    return auc


def run_training() -> dict:
    train_df = add_datetime_features(load_raw(TRAIN_FILE))
    X_train, y_train = split_xy(train_df)

    search = build_search()
    search.fit(X_train, y_train)
    logger.info("Best CV ROC-AUC: %.4f", search.best_score_)
    logger.info("Best params: %s", search.best_params_)

    best_model = search.best_estimator_
    test_metrics = {Path(p).name: evaluate(best_model, p) for p in TEST_FILES}

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    logger.info("Saved model to %s", MODEL_PATH)

    metrics = {
        "cv_best_roc_auc": float(search.best_score_),
        "best_params": search.best_params_,
        "test_roc_auc": test_metrics,
    }
    (MODEL_PATH.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    run_training()
