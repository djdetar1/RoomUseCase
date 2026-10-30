"""Train on a tiny synthetic set, then score a reading through predict_one."""
import pandas as pd
import pytest

xgboost = pytest.importorskip("xgboost")

from occupancy.features import add_datetime_features, make_feature_pipeline
from occupancy.predict import predict_one


def test_train_and_predict_roundtrip():
    # Minimal synthetic set with both classes across different hours.
    rows = []
    for h, occ in [(9, 1), (10, 1), (2, 0), (3, 0)] * 5:
        rows.append(
            {
                "date": f"2015-02-04 {h:02d}:00:00",
                "Temperature": 23.0 + occ,
                "Humidity": 27.0,
                "Light": 400.0 if occ else 0.0,
                "CO2": 700.0 if occ else 400.0,
                "HumidityRatio": 0.0047,
                "Occupancy": occ,
            }
        )
    df = add_datetime_features(pd.DataFrame(rows))
    y = df["Occupancy"]
    X = df.drop(columns=["Occupancy"])

    model = make_feature_pipeline(
        xgboost.XGBClassifier(n_estimators=10, max_depth=2, eval_metric="auc")
    )
    model.fit(X, y)

    result = predict_one(
        {
            "date": "2015-02-04 09:00:00",
            "Temperature": 24.0,
            "Humidity": 27.0,
            "Light": 400.0,
            "CO2": 700.0,
            "HumidityRatio": 0.0047,
        },
        model=model,
    )
    assert set(result) == {"occupancy", "probability"}
    assert result["occupancy"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
