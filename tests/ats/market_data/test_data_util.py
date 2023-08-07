import datetime
import logging
import math
from functools import cached_property, partial

from hydra import initialize, compose
import numpy as np
import pandas as pd
import ray

from ats.app.env_mgr import EnvMgr
from ats.event.macro_indicator import MacroDataBuilder
from ats.market_data import data_util
from ats.market_data import market_data_mgr
from ats.util import logging_utils

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.options.display.float_format = "{:.5f}".format

def test_rolling():
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
    raw_data['close_rolling_3d_max'] = raw_data.close.rolling(3).max()
    np.testing.assert_array_almost_equal(
        raw_data["close_rolling_3d_max"],
        [np.nan, np.nan, 3, 4, 5, 6, 7],
        decimal=3
    )
    raw_data['volume_rolling_3d_max'] = raw_data.volume.rolling(3).max()
    np.testing.assert_array_almost_equal(
        raw_data["volume_rolling_3d_max"],
        [np.nan, np.nan, 3.,  3.,  2.,  3.,  4],
        decimal=3
    )
    
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
    full_ds = ray.data.from_pandas(raw_data)
    add_group_features = partial(data_util.add_group_features, 30*23*2)
    full_ds = full_ds.groupby("ticker").map_groups(add_group_features)
    raw_data = full_ds.to_pandas()
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


def test_with_high():
    start_timestamp = 1325689200
    delta = 30*60
    timestamps = [start_timestamp + i*delta for i in range(11)]
    raw_data = {
        "ticker": ["ES", "ES", "ES", "ES", "ES", "ES", "ES", "ES", "ES", "ES", "ES"],
        "open": [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8],
        "high": [1, 1, 1, 3, 4, 5, 6, 7, 8, 9, 10],
        "low": [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8],
        "close": [1, 1, 1, 1, 2, 3, 4, 15, 6, 7, 8],
        "volume": [1, 1, 1, 1, 3, 2, 1, 2, 3, 4, 5],
        "dv": [1, 1, 1, 1, 2, 3, 1, 2, 3, 1, 2],
        "timestamp": timestamps
    }
    raw_data = pd.DataFrame(data=raw_data)
    # fake the time interval to one day so that we can have 5 day high with 5
    # intervals
    full_ds = ray.data.from_pandas(raw_data)
    add_group_features = partial(data_util.add_group_features, 30*23*2)
    full_ds = full_ds.groupby("ticker").map_groups(add_group_features)
    raw_data = full_ds.to_pandas()
    row_two = raw_data.iloc[5]
    assert row_two["ticker"] == "ES"
    assert row_two["close"] == 3
    assert row_two["volume"] == 2
    assert row_two["dv"] == 3
    assert row_two["cum_volume"] == 9
    assert row_two["cum_dv"] == 9
    assert math.isclose(row_two["close_back"], 0.0019900504080103687, rel_tol=0.01)
    assert math.isclose(row_two["volume_back"], -0.2231435513142097, rel_tol=0.01)
    peak_timestamp = start_timestamp + delta * 4
    np.testing.assert_array_almost_equal(
        raw_data["close_high_5_ff"],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.006, 0.028, 0.028, 0.028, 0.028],
        decimal=3,
        err_msg="can not match close_high_5_ff")
    np.testing.assert_array_almost_equal(
        raw_data["time_high_5_ff"],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1325700000, 1325701800, 1325701800, 1325701800, 1325701800],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        raw_data["close_high_5_bf"],
        [0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.028, 0.028, 0.028, 0.028],
        decimal=3,
    )
    np.testing.assert_array_almost_equal(
        raw_data["close_low_5_ff"],
        [ np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.   , 0.   , 0.002, 0.004, 0.006],
        decimal=3,
    )


def test_with_negative_price():
    start_timestamp = 1325689200
    delta = 30*60
    timestamps = [start_timestamp + i*delta for i in range(8)]
    raw_data = {
        "ticker": ["CL", "CL", "CL", "CL", "CL", "CL", "CL", "CL"],
        "open": [-1, -2, -3, -4, -5, -6, -7, -8],
        "high": [-3, -4, -5, -6, -7, -8, -9, -10],
        "low": [-1, -2, -3, -4, -5, -6, -7, -8],
        "close": [-1, -2, -3, -4, -150, -6, -7, -8],
        "volume": [1, 3, 2, 1, 2, 3, 4, 5],
        "dv": [1, 2, 3, 1, 2, 3, 1, 2],
        "timestamp": timestamps
    }
    raw_data = pd.DataFrame(data=raw_data)
    # fake the time interval to one day so that we can have 5 day high with 5
    # intervals
    full_ds = ray.data.from_pandas(raw_data)
    add_group_features = partial(data_util.add_group_features, 30*23*2)
    full_ds = full_ds.groupby("ticker").map_groups(add_group_features)
    raw_data = full_ds.to_pandas()
    row_two = raw_data.iloc[2]
    assert row_two["ticker"] == "CL"
    assert row_two["close"] == -3
    assert row_two["volume"] == 2
    assert row_two["dv"] == 3
    assert row_two["cum_volume"] == 6
    assert row_two["cum_dv"] == 6
    assert math.isclose(row_two["close_back"], -0.0020100509280238654, rel_tol=0.01)
    assert math.isclose(row_two["volume_back"], -0.2231435513142097, rel_tol=0.01)
    peak_timestamp = start_timestamp + delta * 4
    np.testing.assert_array_almost_equal(
        raw_data["close_high_5_ff"],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, -0.002, -0.004],
        decimal=3,
        err_msg="can not match close_high_5_ff")

def test_with_low():
    start_timestamp = 1325689200
    delta = 30*60
    timestamps = [start_timestamp + i*delta for i in range(11)]
    raw_data = {
        "ticker": ["ES", "ES", "ES", "ES", "ES", "ES", "ES", "ES", "ES", "ES", "ES"],
        "open": [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8],
        "high": [1, 1, 1, 3, 4, 5, 6, 7, 8, 9, 10],
        "low": [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8],
        "close": [1, 1, 1, 1, 2, 3, 4, 0.5, 6, 7, 8],
        "volume": [1, 1, 1, 1, 3, 2, 1, 2, 3, 4, 5],
        "dv": [1, 1, 1, 1, 2, 3, 1, 2, 3, 1, 2],
        "timestamp": timestamps
    }
    raw_data = pd.DataFrame(data=raw_data)
    # fake the time interval to one day so that we can have 5 day high with 5
    # intervals
    full_ds = ray.data.from_pandas(raw_data)
    add_group_features = partial(data_util.add_group_features, 30*23*2)
    full_ds = full_ds.groupby("ticker").map_groups(add_group_features)
    raw_data = full_ds.to_pandas()
    row_two = raw_data.iloc[5]
    assert row_two["ticker"] == "ES"
    assert row_two["close"] == 3
    assert row_two["volume"] == 2
    assert row_two["dv"] == 3
    assert row_two["cum_volume"] == 9
    assert row_two["cum_dv"] == 9
    assert math.isclose(row_two["close_back"], 0.0019900504080103687, rel_tol=0.01)
    assert math.isclose(row_two["volume_back"], -0.2231435513142097, rel_tol=0.01)
    peak_timestamp = start_timestamp + delta * 4
    np.testing.assert_array_almost_equal(
        raw_data["close_high_5_ff"],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.006, 0.006, 0.01, 0.012, 0.014],
        decimal=3, verbose=True,
        err_msg="can not match close_high_5_ff")
    np.testing.assert_array_almost_equal(
        raw_data["time_high_5_ff"],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1325700000, 1325700000, 1325703600, 1325705400, 1325707200],
        decimal=3, verbose=True, err_msg="can not match time_high_5_ff",
    )
    np.testing.assert_array_almost_equal(
        raw_data["close_high_5_bf"],
        [0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.01, 0.012, 0.014],
        decimal=3, verbose=True, err_msg="can not match close_high_5_bf"
    )
    np.testing.assert_array_almost_equal(
        raw_data["close_low_5_ff"],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0., -0.001, -0.001, -0.001, -0.001],
        decimal=3, verbose=True, err_msg="can not match close_low_5_ff",
    )
    np.testing.assert_array_almost_equal(
        raw_data["time_low_5_ff"],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1325701800, 1325701800, 1325701800, 1325701800],
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
    full_ds = ray.data.from_pandas(raw_data)
    add_group_features = partial(data_util.add_group_features, 30)
    full_ds = full_ds.groupby("ticker").map_groups(add_group_features)
    raw_data = full_ds.to_pandas()
    row_two = raw_data.iloc[2]
    assert row_two["ticker"] == "ES"
    # base price 500 is used as denominator
    assert math.isclose(row_two["close_back"], 0.001982161203990529, rel_tol=0.01)


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
        macro_data_builder = MacroDataBuilder(env_mgr)
        start_timestamp = 1325689200
        delta = 30*60
        timestamps = [start_timestamp + i*delta for i in range(8)]
        raw_data = {
            "ticker": ["ES", "ES", "ES", "ES", "ES", "ES", "ES", "ES"],
            "open": [1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048],
            "high": [1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050],
            "low": [1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048],
            "close": [1041, 1042, 1043, 1044, 1540, 1046, 1047, 1048],
            "volume": [1, 3, 2, 1, 2, 3, 4, 5],
            "dv": [1, 2, 3, 1, 2, 3, 1, 2],
            "timestamp": timestamps
        }
        raw_data = pd.DataFrame(data=raw_data)
        raw_data["time"] = raw_data.timestamp.apply(lambda x:datetime.datetime.fromtimestamp(x))
        # fake the time interval to one day so that we can have 5 day high with 5
        # intervals
        full_ds = ray.data.from_pandas(raw_data)
        add_group_features = partial(data_util.add_group_features, 30*23*2)
        full_ds = full_ds.groupby("ticker").map_groups(add_group_features)
        add_example_features = partial(
            data_util.add_example_level_features, market_cal, macro_data_builder)
        full_ds = full_ds.map_batches(add_example_features)
        raw_data = full_ds.to_pandas()
        row_two = raw_data.iloc[2]
        assert row_two["ticker"] == "ES"
        assert row_two["close_back"] ==  0.0006482982398861026
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
            [1325687400] * data_len,
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
            [1327096800] * data_len,
            decimal=3, verbose=True, err_msg="can not match option_expiration_time",
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
    # base price 500 is used as denominator
    assert math.isclose(row_two["close_back"], 0.001982161203990529, rel_tol=0.01)


def test_add_example_features_vwap_before_new_york_open():
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
        macro_data_builder = MacroDataBuilder(env_mgr)
        start_timestamp = 1325689200
        delta = 30*60
        timestamps = [start_timestamp + i*delta for i in range(8)]
        raw_data = {
            "ticker": ["ES", "ES", "ES", "ES", "ES", "ES", "ES", "ES"],
            "open": [1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048],
            "high": [1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050],
            "low": [1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048],
            "close": [1041, 1042, 1043, 1044, 1540, 1046, 1047, 1048],
            "volume": [1, 3, 2, 1, 2, 3, 4, 5],
            "dv": [1, 2, 3, 1, 2, 3, 1, 2],
            "timestamp": timestamps
        }
        raw_data = pd.DataFrame(data=raw_data)
        data_len = len(raw_data.timestamp)
        raw_data["time"] = raw_data.timestamp.apply(lambda x:datetime.datetime.fromtimestamp(x))
        # fake the time interval to one day so that we can have 5 day high with 5
        # intervals
        full_ds = ray.data.from_pandas(raw_data)
        add_group_features = partial(data_util.add_group_features, 30*23*2)
        full_ds = full_ds.groupby("ticker").map_groups(add_group_features)
        add_example_features = partial(
            data_util.add_example_level_features, market_cal, macro_data_builder)
        full_ds = full_ds.map_batches(add_example_features)
        raw_data = full_ds.to_pandas()
        logging.error(f"raw_data:{raw_data}")
        np.testing.assert_array_almost_equal(
            raw_data["vwap_since_new_york_open"],
            [np.nan] * data_len,
            decimal=3, verbose=True, err_msg="can not match weekly_close_time",
        )

def test_add_example_features_vwap_around_new_york_open_no_event():
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
        macro_data_builder = MacroDataBuilder(env_mgr)
        start_timestamp = datetime.datetime(2023,8,3,12,0,0).timestamp()
        delta = 30*60
        timestamps = [start_timestamp + i*delta for i in range(8)]
        raw_data = {
            "ticker": ["ES", "ES", "ES", "ES", "ES", "ES", "ES", "ES"],
            "open": [1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048],
            "high": [1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050],
            "low": [1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048],
            "close": [1041, 1042, 1043, 1044, 1540, 1046, 1047, 1048],
            "volume": [1, 3, 2, 1, 2, 3, 4, 5],
            "dv": [1, 2, 3, 1, 2, 3, 1, 2],
            "timestamp": timestamps
        }
        raw_data = pd.DataFrame(data=raw_data)
        data_len = len(raw_data.timestamp)
        raw_data["time"] = raw_data.timestamp.apply(lambda x:datetime.datetime.fromtimestamp(x))
        # fake the time interval to one day so that we can have 5 day high with 5
        # intervals
        full_ds = ray.data.from_pandas(raw_data)
        add_group_features = partial(data_util.add_group_features, 30*23*2)
        full_ds = full_ds.groupby("ticker").map_groups(add_group_features)
        add_example_features = partial(
            data_util.add_example_level_features, market_cal, macro_data_builder)
        full_ds = full_ds.map_batches(add_example_features)
        raw_data = full_ds.to_pandas()
        logging.error(f"raw_data:{raw_data}")
        np.testing.assert_array_almost_equal(
            raw_data["vwap_since_new_york_open"],
            [np.nan, np.nan, np.nan, 0., 1.404, 1.127, 1.128, 1.129],
            decimal=3, verbose=True, err_msg="can not match vwap_since_new_york_open",
        )
        np.testing.assert_array_almost_equal(
            raw_data["ret_from_vwap_around_new_york_open"],
            [np.nan, np.nan, np.nan, np.nan, 1.404, 1.127, 1.127, 1.128],
            decimal=3, verbose=True, err_msg="can not match vwap_since_new_york_open",
        )


def test_add_example_features_vwap_around_new_york_open():
    with initialize(version_base=None, config_path="../../../conf"):
        cfg = compose(
            config_name="test",
            overrides=[
	        "dataset.read_snapshot=False",
                "model.features.add_macro_event=True",
                "job.test_start_date=2023-07-01",
                "job.test_end_date=2023-07-28",
            ],
            return_hydra_config=True
        )
        env_mgr = EnvMgr(cfg)
        market_cal = env_mgr.market_cal
        macro_data_builder = MacroDataBuilder(env_mgr)
        start_timestamp = datetime.datetime(2023,7,27,11,0,0).timestamp()
        delta = 30*60
        timestamps = [start_timestamp + i*delta for i in range(10)]
        raw_data = {
            "ticker": ["ES", "ES", "ES", "ES", "ES", "ES", "ES", "ES", "ES", "ES"],
            "open": [1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050],
            "high": [1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052],
            "low": [1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050],
            "close": [1041, 1042, 1043, 1044, 1540, 1046, 1047, 1048, 1149, 1050],
            "volume": [1, 1, 1, 1, 1, 1, 1, 1, 300, 7],
            "dv": [1041, 1042, 1043, 1044, 1540, 1046, 1047, 1048, 300*1149, 1050*7],
            "timestamp": timestamps
        }
        raw_data = pd.DataFrame(data=raw_data)
        data_len = len(raw_data.timestamp)
        raw_data["time"] = raw_data.timestamp.apply(lambda x:datetime.datetime.fromtimestamp(x))
        # fake the time interval to one day so that we can have 5 day high with 5
        # intervals
        full_ds = ray.data.from_pandas(raw_data)
        add_group_features = partial(data_util.add_group_features, 30*23*2)
        full_ds = full_ds.groupby("ticker").map_groups(add_group_features)
        add_example_features = partial(
            data_util.add_example_level_features, market_cal, macro_data_builder)
        full_ds = full_ds.map_batches(add_example_features)
        raw_data = full_ds.to_pandas()

        logging.error(f"raw_data:{raw_data}")
        np.testing.assert_array_almost_equal(
            raw_data["vwap_since_new_york_open"],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 0.00000, 0.00000, 0.00032, 0.00041, -0.06015],
            decimal=3, verbose=True, err_msg="can not match weekly_close_time",
        )
        np.testing.assert_array_almost_equal(
            raw_data["vwap_since_last_macro_event"],
            [np.nan, np.nan, np.nan, 0.00000, 0.00000, -0.14822, -0.10076, -0.07601, -0.00017, -0.06073],
            decimal=3, verbose=True, err_msg="can not match weekly_close_time",
        )
        np.testing.assert_array_almost_equal(
            raw_data["ret_from_vwap_around_macro_event"],
            [np.nan, np.nan, np.nan, np.nan, 0.00000, -0.27728, -0.27663, -0.27599, -0.21278, -0.27469],
            decimal=3, verbose=True, err_msg="can not match weekly_close_time",
        )
        np.testing.assert_array_almost_equal(
            raw_data["ret_from_vwap_around_new_york_open"],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.00000, 0.00065, 0.06385, 0.00194],
            decimal=3, verbose=True, err_msg="can not match weekly_close_time",
        )


def test_add_example_features_vwap_around_london_open():
    with initialize(version_base=None, config_path="../../../conf"):
        cfg = compose(
            config_name="test",
            overrides=[
	        "dataset.read_snapshot=False",
                "model.features.add_macro_event=True",
                "job.test_start_date=2023-07-01",
                "job.test_end_date=2023-07-28",
            ],
            return_hydra_config=True
        )
        env_mgr = EnvMgr(cfg)
        market_cal = env_mgr.market_cal
        macro_data_builder = MacroDataBuilder(env_mgr)
        start_timestamp = datetime.datetime(2023,7,27,6,0,0).timestamp()
        delta = 30*60
        timestamps = [start_timestamp + i*delta for i in range(10)]
        raw_data = {
            "ticker": ["ES", "ES", "ES", "ES", "ES", "ES", "ES", "ES", "ES", "ES"],
            "open": [1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050],
            "high": [1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052],
            "low": [1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050],
            "close": [1041, 1042, 1043, 1044, 1540, 1046, 1047, 1048, 1149, 1050],
            "volume": [1, 1, 1, 1, 1, 1, 1, 1, 300, 7],
            "dv": [1041, 1042, 1043, 1044, 1540, 1046, 1047, 1048, 300*1149, 1050*7],
            "timestamp": timestamps
        }
        raw_data = pd.DataFrame(data=raw_data)
        data_len = len(raw_data.timestamp)
        raw_data["time"] = raw_data.timestamp.apply(lambda x:datetime.datetime.fromtimestamp(x))
        # fake the time interval to one day so that we can have 5 day high with 5
        # intervals
        full_ds = ray.data.from_pandas(raw_data)
        add_group_features = partial(data_util.add_group_features, 30*23*2)
        full_ds = full_ds.groupby("ticker").map_groups(add_group_features)
        add_example_features = partial(
            data_util.add_example_level_features, market_cal, macro_data_builder)
        full_ds = full_ds.map_batches(add_example_features)
        raw_data = full_ds.to_pandas()
        np.testing.assert_array_almost_equal(
            raw_data["vwap_since_london_open"],
            [np.nan, np.nan, 0.00000, 0.00000, 0.12962, -0.10082, -0.07606, -0.06078, 0.00004, -0.06053],
            decimal=3, verbose=True, err_msg="can not match vwap_since_london_open",
        )
        np.testing.assert_array_almost_equal(
            raw_data["ret_from_vwap_around_london_open"],
            [np.nan, np.nan, np.nan, 0.00000, 0.27857, 0.00129, 0.00194, 0.00259, 0.06579, 0.00388],
            decimal=3, verbose=True, err_msg="can not match ret_from_vwap_around_london_open",
        )

if __name__ == "__main__":
    logging_utils.init_logging()
