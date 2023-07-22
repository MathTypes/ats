import datetime
import logging
import math

from hydra import initialize, compose
import numpy as np
import pandas as pd

from ats.app.env_mgr import EnvMgr
from ats.event.macro_indicator import MacroDataBuilder
from ats.market_data import data_util
from ats.market_data import market_data_mgr
from ats.util import logging_utils

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


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
    assert row_two["close_back"] == 0.5 # 2=>3
    assert math.isclose(row_two["volume_back"], -0.3333, rel_tol=0.01) # 3=>2
    assert np.isnan(row_two["close_high_5_ff"])
    assert np.isnan(row_two["time_high_5_ff"])
    assert np.isnan(row_two["close_low_5_ff"])
    assert np.isnan(row_two["time_low_5_ff"])


def test_with_high():
    start_timestamp = 1325689200
    delta = 30*60
    timestamps = [start_timestamp + i*delta for i in range(8)]
    raw_data = {
        "ticker": ["ES", "ES", "ES", "ES", "ES", "ES", "ES", "ES"],
        "open": [1, 2, 3, 4, 5, 6, 7, 8],
        "high": [3, 4, 5, 6, 7, 8, 9, 10],
        "low": [1, 2, 3, 4, 5, 6, 7, 8],
        "close": [1, 2, 3, 4, 150, 6, 7, 8],
        "volume": [1, 3, 2, 1, 2, 3, 4, 5],
        "dv": [1, 2, 3, 1, 2, 3, 1, 2],
        "timestamp": timestamps
    }
    raw_data = pd.DataFrame(data=raw_data)
    # fake the time interval to one day so that we can have 5 day high with 5
    # intervals
    raw_data = data_util.add_group_features(raw_data, 30*23*2)
    row_two = raw_data.iloc[2]
    assert row_two["ticker"] == "ES"
    assert row_two["close"] == 3
    assert row_two["volume"] == 2
    assert row_two["dv"] == 3
    assert row_two["cum_volume"] == 6
    assert row_two["cum_dv"] == 6
    assert math.isclose(row_two["close_back"], 0.405465, rel_tol=0.01)
    assert math.isclose(row_two["volume_back"], -0.405465, rel_tol=0.01)
    peak_timestamp = start_timestamp + delta * 4
    np.testing.assert_array_almost_equal(
        raw_data["close_high_5_ff"],
        [np.nan, np.nan, np.nan, np.nan, 4.317488, 4.317488, 4.317488, 4.317488],
        decimal=3,
        err_msg="can not match close_high_5_ff")
    np.testing.assert_array_almost_equal(
        raw_data["time_high_5_ff"],
        [np.nan, np.nan, np.nan, np.nan, peak_timestamp, peak_timestamp, peak_timestamp, peak_timestamp],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        raw_data["close_high_5_bf"],
        [4.317488, 4.317488, 4.317488, 4.317488, 4.317488, np.nan, np.nan, np.nan],
        decimal=3,
    )
    np.testing.assert_array_almost_equal(
        raw_data["close_low_5_ff"],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        decimal=3,
    )


def test_with_low():
    start_timestamp = 1325689200
    delta = 30*60
    timestamps = [start_timestamp + i*delta for i in range(8)]
    raw_data = {
        "ticker": ["ES", "ES", "ES", "ES", "ES", "ES", "ES", "ES"],
        "open": [1, 2, 3, 4, 5, 6, 7, 8],
        "high": [3, 4, 5, 6, 7, 8, 9, 10],
        "low": [1, 2, 3, 4, 5, 6, 7, 8],
        "close": [1, 2, 3, 4, 0.5, 6, 7, 8],
        "volume": [1, 3, 2, 1, 2, 3, 4, 5],
        "dv": [1, 2, 3, 1, 2, 3, 1, 2],
        "timestamp": timestamps
    }
    raw_data = pd.DataFrame(data=raw_data)
    # fake the time interval to one day so that we can have 5 day high with 5
    # intervals
    raw_data = data_util.add_group_features(raw_data, 30*23*2)
    row_two = raw_data.iloc[2]
    assert row_two["ticker"] == "ES"
    assert row_two["close"] == 3
    assert row_two["volume"] == 2
    assert row_two["dv"] == 3
    assert row_two["cum_volume"] == 6
    assert row_two["cum_dv"] == 6
    assert math.isclose(row_two["close_back"], 0.405465, rel_tol=0.01)
    assert math.isclose(row_two["volume_back"], -0.405465, rel_tol=0.01)
    peak_timestamp = start_timestamp + delta * 4
    np.testing.assert_array_almost_equal(
        raw_data["close_high_5_ff"],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        decimal=3, verbose=True,
        err_msg="can not match close_high_5_ff")
    np.testing.assert_array_almost_equal(
        raw_data["time_high_5_ff"],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        decimal=3, verbose=True, err_msg="can not match time_high_5_ff",
    )
    np.testing.assert_array_almost_equal(
        raw_data["close_high_5_bf"],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        decimal=3, verbose=True, err_msg="can not match close_high_5_bf"
    )
    np.testing.assert_array_almost_equal(
        raw_data["close_low_5_ff"],
        [np.nan, np.nan, np.nan, np.nan, -1.386294, -1.386294, -1.386294, -1.386294],
        decimal=3, verbose=True, err_msg="can not match close_low_5_ff",
    )
    np.testing.assert_array_almost_equal(
        raw_data["time_low_5_ff"],
        [np.nan, np.nan, np.nan, np.nan, peak_timestamp, peak_timestamp, peak_timestamp, peak_timestamp],
        decimal=3, verbose=True, err_msg="can not match time_low_5_ff",
    )

def test_group_features():
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


def test_add_example_features():
    with initialize(version_base=None, config_path="../../../conf"):
        cfg = compose(
            config_name="test",
            overrides=[
	        "dataset.read_snapshot=False",
            ],
            return_hydra_config=True
        )
        env_mgr = EnvMgr(cfg)
        market_cal = env_mgr.market_cal
        macro_data_builder = MacroDataBuilder(cfg)
        start_timestamp = 1325689200
        delta = 30*60
        timestamps = [start_timestamp + i*delta for i in range(8)]
        raw_data = {
            "ticker": ["ES", "ES", "ES", "ES", "ES", "ES", "ES", "ES"],
            "open": [1, 2, 3, 4, 5, 6, 7, 8],
            "high": [3, 4, 5, 6, 7, 8, 9, 10],
            "low": [1, 2, 3, 4, 5, 6, 7, 8],
            "close": [1, 2, 3, 4, 150, 6, 7, 8],
            "volume": [1, 3, 2, 1, 2, 3, 4, 5],
            "dv": [1, 2, 3, 1, 2, 3, 1, 2],
            "timestamp": timestamps
        }
        raw_data = pd.DataFrame(data=raw_data)
        raw_data["time"] = raw_data.timestamp.apply(lambda x:datetime.datetime.fromtimestamp(x))
        # fake the time interval to one day so that we can have 5 day high with 5
        # intervals
        raw_data = data_util.add_group_features(raw_data, 30*23*2)
        raw_data = data_util.add_example_level_features(raw_data, market_cal, macro_data_builder)
        row_two = raw_data.iloc[2]
        assert row_two["ticker"] == "ES"
        assert row_two["close_back"] ==  0.4054651081081643
        data_len = len(raw_data.timestamp)
        np.testing.assert_array_almost_equal(
            raw_data["weekly_close_time"],
            [1325887200] * data_len,
            decimal=3, verbose=True, err_msg="can not match weekly_close_time",
        )
        np.testing.assert_array_almost_equal(
            raw_data["monthly_close_time"],
            [1328047200] * data_len,
            decimal=3, verbose=True, err_msg="can not match monthly_close_time",
        )
        np.testing.assert_array_almost_equal(
            raw_data["new_york_open_time"],
            [1325631600] * data_len,
            decimal=3, verbose=True, err_msg="can not match new_york_open_time",
        )
        np.testing.assert_array_almost_equal(
            raw_data["london_open_time"],
            [1325664000] * data_len,
            decimal=3, verbose=True, err_msg="can not match london_open_time",
        )
        np.testing.assert_array_almost_equal(
            raw_data["london_close_time"],
            [1325694600] * data_len,
            decimal=3, verbose=True, err_msg="can not match london_close_time",
        )
        np.testing.assert_array_almost_equal(
            raw_data["option_expiration_time"],
            [1328047200] * data_len,
            decimal=3, verbose=True, err_msg="can not match option_expiration_time",
        )

if __name__ == "__main__":
    logging_utils.init_logging()
