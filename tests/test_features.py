import pandas as pd

from occupancy.features import add_datetime_features, build_preprocessor


def test_add_datetime_features_adds_hour_and_weekday():
    df = pd.DataFrame(
        {
            "date": ["2015-02-04 17:51:00"],
            "Temperature": [23.18],
            "Humidity": [27.27],
            "Light": [426.0],
            "CO2": [721.25],
            "HumidityRatio": [0.0047929],
        }
    )
    out = add_datetime_features(df)
    assert out.loc[0, "hour"] == 17
    # 2015-02-04 is a Wednesday -> weekday() == 2
    assert out.loc[0, "weekday"] == 2


def test_preprocessor_builds():
    pre = build_preprocessor()
    assert pre is not None
