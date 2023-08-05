import datetime
import logging
import math

from hydra import initialize, compose
import numpy as np
import pandas as pd


from empyrical import (
    sharpe_ratio,
    calmar_ratio,
    sortino_ratio,
    max_drawdown,
    downside_risk,
    annual_return,
    annual_volatility,
    # cum_returns,                                                                                                                                                )
)

from ats.app.env_mgr import EnvMgr
from ats.event.macro_indicator import MacroDataBuilder
from ats.market_data import data_util
from ats.market_data import market_data_mgr
from ats.util import logging_utils

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.options.display.float_format = "{:.5f}".format

def test_add_highs_trending_no_high():
    start_timestamp = 1325689200
    delta = 30*60
    timestamps = [start_timestamp + i*delta for i in range(7)]
    raw_data = {
        "ticker": ["ES", "ES", "ES", "ES", "ES", "ES", "ES"],
        "open": [1, 2, 3, 4, 5, 6, 7],
        "high": [3, 4, 5, 6, 7, 8, 9],
        "low": [1, 2, 3, 4, 5, 6, 7],
        "close": [1, 2, 3, 4, 5, 6, 7],
        "volume": [1, 3, 2, 1, 2, 3, 4],
        "dv": [1, 2, 3, 1, 2, 3, 1],
        "timestamp": timestamps
    }
    raw_data = pd.DataFrame(data=raw_data)
    raw_data = data_util.add_group_features(raw_data, 30)
    row_two = raw_data.iloc[2]
    assert row_two["ticker"] == "ES"
    assert row_two["close"] == 3
    assert row_two["volume"] == 2
    assert row_two["dv"] == 3
    assert row_two["cum_volume"] == 6
    assert row_two["cum_dv"] == 6
    assert math.isclose(row_two["close_back"], 0.0019900504080103687, rel_tol=0.01)
    assert math.isclose(row_two["volume_back"], -0.2231435513142097, rel_tol=0.01) # 3=>2
    assert pd.isna(row_two["close_high_5_ff"])
    assert pd.isna(row_two["time_high_5_ff"])
    assert pd.isna(row_two["close_low_5_ff"])
    assert pd.isna(row_two["time_low_5_ff"])
