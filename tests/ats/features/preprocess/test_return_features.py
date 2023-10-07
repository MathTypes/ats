import logging

import datetime
import numpy as np
import pandas as pd
from hamilton import base, driver, log_setup
from hamilton.experimental import h_ray
from hamilton.experimental import h_cache
from hydra import initialize, compose
import ray
from ray import workflow

from ats.app.env_mgr import EnvMgr
from ats.features.preprocess.test_utils import run_features
from ats.market_data import market_data_mgr
from ats.util import logging_utils

def test_ret_from_high_1d():
    result = run_features("ret_from_high_1d", 5)
    print(f"result:{result}")
    np.testing.assert_array_almost_equal(
        result["ret_from_high_1d"],
        [-0.0010, -0.0010, -0.0027, -0.0015, 0.0000],
        decimal=3
    )

def test_ret_from_low_21():
    result = run_features("ret_from_low_21", 5)
    print(f"result:{result}")
    np.testing.assert_array_almost_equal(
        result["ret_from_low_21"],
        [0.0098, 0.0098, 0.0081, 0.0093, 0.0108],
        decimal=3
    )

def test_ret_from_high_21():
    result = run_features("ret_from_high_21", 5)
    print(f"result:{result}")
    np.testing.assert_array_almost_equal(
        result["ret_from_high_21"],
        [-0.0010, -0.0010, -0.0027, -0.0015, 0.0000],
        decimal=3
    )

# test_ret_from_low_1d_ff_shift_1d
def test_ret_from_low_1d_ff_shift_1d():
    result = run_features("ret_from_low_1d_shift_1d", 100)
    print(f"result:{result}")
    np.testing.assert_array_almost_equal(
        result['ret_from_low_1d_shift_1d'][10:15],
        [0.002, 0.003, 0.003, 0.002, 0.001],
        decimal=3
    )
    
def test_ret_from_sma_5():
    result = run_features("ret_from_sma_5", 5)
    print(f"result:{result}")
    np.testing.assert_array_almost_equal(
        result["ret_from_sma_5"],
        [0.0018, 0.0011, -0.0009, 0.0002, 0.0013],
        decimal=3
    )

def test_ret_from_vwap_around_london_close():
    result = run_features("ret_from_vwap_around_london_close", 100)
    print(f"result:{result}")
    np.testing.assert_array_almost_equal(
        result['ret_from_vwap_around_london_close'][10:15],
        [0.02 , 0.02 , 0.02 , 0.019, 0.019],
        decimal=3
    )

def test_ret_from_vwap_around_london_close_20230411():
    timestamp = 1681192800
    result = run_features("ret_from_vwap_around_london_close", 100000, timestamp)
    #print(f"result:{result}")
    close_list = result.query(f"(timestamp=={timestamp})")['ret_from_vwap_around_london_close']
    print(f"close_list:{close_list}")
    close_time_list = close_list.index.get_level_values(level=0)[:10].to_list()
    np.testing.assert_array_almost_equal(
        result['ret_from_vwap_around_london_close'][10:15],
        [0.02 , 0.02 , 0.02 , 0.019, 0.019],
        decimal=3
    )
    
#                              ret_from_vwap_around_london_close  \
#idx_ticker idx_timestamp                                      
#ES         1681192800                                2.2263   
#           1681194600                                2.2259   
#           1681196400                                2.2260
           
def test_ret_from_vwap_since_new_york_open():
    result = run_features("ret_from_vwap_since_new_york_open", 100)
    print(f"result:{result}")
    np.testing.assert_array_almost_equal(
        result['ret_from_vwap_since_new_york_open'][10:15],
        [0.002, 0.003, 0.003, 0.002, 0.001],
        decimal=3
    )

    
def test_ret_from_vwap_pre_new_york_open():
    result = run_features("ret_from_vwap_pre_new_york_open", 100)
    print(f"result:{result}")
    np.testing.assert_array_almost_equal(
        result['ret_from_vwap_pre_new_york_open'][10:15],
        [0.01 , 0.01 , 0.011, 0.009, 0.009],
        decimal=3
    )

def test_ret_from_last_daily_close_0():
    result = run_features("ret_from_last_daily_close_0", 100)
    print(f"result:{result}")
    np.testing.assert_array_almost_equal(
        result['ret_from_last_daily_close_0'][10:15],
        [-0.001, -0.001, -0.002, -0.002, -0.002],
        decimal=3
    )

def test_ret_from_last_daily_close_1():
    result = run_features("ret_from_last_daily_close_1", 100)
    print(f"result:{result}")
    np.testing.assert_array_almost_equal(
        result['ret_from_last_daily_close_1'][10:15],
        [0.02 , 0.02 , 0.019, 0.018, 0.018],
        decimal=3
    )

def test_ret_from_last_daily_close_2():
    result = run_features("ret_from_last_daily_close_2", 100)["ret_from_last_daily_close_2"][10:15]
    print(f"result:{result.to_list()}")
    np.testing.assert_array_almost_equal(
        result,
        [0.025, 0.026, 0.024, 0.024, 0.024],
        decimal=3
    )

def test_ret_from_last_weekly_close_1():
    result = run_features("ret_from_last_weekly_close_1", 100)
    print(f"result:{result}")
    np.testing.assert_array_almost_equal(
        result['ret_from_last_weekly_close_1'][10:15],
        [0.007, 0.007, 0.006, 0.005, 0.005],
        decimal=3
    )


def test_ret_from_last_monthly_close_1():
    result = run_features("ret_from_last_monthly_close_1", 100)
    print(f"result:{result}")
    np.testing.assert_array_almost_equal(
        result['ret_from_last_monthly_close_1'][10:15],
        [-0.013, -0.013, -0.015, -0.015, -0.015],
        decimal=3
    )

def test_ret_from_last_monthly_close_9():
    result = run_features("ret_from_last_monthly_close_9", 100)['ret_from_last_monthly_close_9'][10:15]
    print(f"result:{result.to_list()}")
    np.testing.assert_array_almost_equal(
        result,
        [0.001,  0.001, -0.   , -0.001, -0.001],
        decimal=3
    )

def test_ret_from_kc_10d_05_high():
    result = run_features("ret_from_kc_10d_05_high", 100)['ret_from_kc_10d_05_high'][10:15]
    print(f"result:{result.to_list()}")
    np.testing.assert_array_almost_equal(
        result,
        [0.004, 0.004, 0.003, 0.002, 0.002],
        decimal=3
    )

def test_ret_from_vwap_pre_new_york_close():
    result = run_features("ret_from_vwap_pre_new_york_close", 100)['ret_from_vwap_pre_new_york_close'][10:15]
    print(f"result:{result.to_list()}")
    np.testing.assert_array_almost_equal(
        result,
        [-0.0012162281057896962, -0.0008685833951798116, -0.0006948063506895252, -0.00191188030775713, -0.0024339372238166845],
        decimal=3
    )

def test_ret_from_vwap_pre_london_open():
    result = run_features("ret_from_vwap_pre_london_open", 100)['ret_from_vwap_pre_london_open'][10:15]
    print(f"result:{result}")
    np.testing.assert_array_almost_equal(
        result,
        [0.019, 0.02 , 0.02 , 0.019, 0.018],
        decimal=3
    )

def test_ret_from_vwap_pre_london_close():
    result = run_features("ret_from_vwap_pre_london_close", 100)['ret_from_vwap_pre_london_close'][10:15]
    print(f"result:{result}")
    np.testing.assert_array_almost_equal(
        result,
        [0.0195, 0.0198, 0.0200, 0.0188, 0.0183],
        decimal=3
    )

def test_example_group_features():
    result = run_features("example_group_features", 100)
    print(f"result:{result}")
    np.testing.assert_array_almost_equal(
        result,
        [0.0195, 0.0198, 0.0200, 0.0188, 0.0183],
        decimal=3
    )

def test_ret_velocity_from_high_5():
    result = run_features("ret_velocity_from_high_5", 100)['ret_velocity_from_high_5'][10:15]
    print(f"result:{result.to_list()}")
    np.testing.assert_array_almost_equal(
        result,
        [-1.351e-07, -1.609e-08,        np.nan, -6.762e-07, -4.831e-07],
        decimal=8
    )

def test_ret_from_high():
    result = run_features("ret_from_high", 100)['ret_from_high'][10:15]
    print(f"result:{result.to_list()}")
    np.testing.assert_array_almost_equal(
        result,
        [-0.00052142, -0.00017378, -0.00086843, -0.00156454, -0.00069602],
        decimal=8
    )

def test_ret_from_low():
    result = run_features("ret_from_low", 100)['ret_from_low'][10:15]
    print(f"result:{result.to_list()}")
    np.testing.assert_array_almost_equal(
        result,
        [0.        , 0.00034764, 0.00069529, 0.        , 0.00017408],
        decimal=8
    )

def test_ret_from_open():
    result = run_features("ret_from_open", 100)['ret_from_open'][10:15]
    close_time_list = result.index.get_level_values(level=0).to_list()
    np.testing.assert_array_almost_equal(
        result,
        [-0.00034764,  0.00017381,  0.00017378, -0.00139082, -0.00052206],
        decimal=8
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1283380200, 1283382000, 1283383800, 1283385600, 1283387400],
        decimal=8
    )

