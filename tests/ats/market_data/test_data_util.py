import pandas as pd

from ats.market_data import data_util
from ats.util import logging_utils


def test_group_features():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    raw_data = {
        "ticker": ["ES", "ES", "ES"],
        "open": [1, 2, 3],
        "high": [3, 4, 5],
        "low": [1, 2, 3],
        "close": [3, 4, 5],
        "volume": [1, 2, 3],
        "dv": [1, 2, 3],
        "timestamp": [1325689200, 1325691000, 1349357400],
    }
    raw_data = pd.DataFrame(data=raw_data)
    raw_data = data_util.add_group_features(raw_data, 30)
    row_two = raw_data.iloc[2]
    assert row_two["ticker"] == "ES"
    assert row_two["close_back"] == 0.25


if __name__ == "__main__":
    logging_utils.init_logging()
