import pandas as pd

from occupancy.data import split_xy


def test_split_xy_separates_target():
    df = pd.DataFrame({"Temperature": [1.0, 2.0], "Occupancy": [0, 1]})
    X, y = split_xy(df)
    assert "Occupancy" not in X.columns
    assert list(y) == [0, 1]
