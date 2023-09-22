import logging

import numpy as np
import pandas as pd
from hamilton import base, driver, log_setup
from hamilton.experimental import h_ray
from hamilton.experimental import h_cache
from hydra import initialize, compose
import ray
from ray import workflow

from ats.app.env_mgr import EnvMgr
from ats.market_data import market_data_mgr
from ats.util import logging_utils

# THe config path is relative to the file calling initialize (this file)
def run_features(feature_name, k=10):
    logging_utils.init_logging()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.options.display.float_format = "{:,.4f}".format
    with initialize(version_base=None, config_path="../../../../conf"):
        cfg = compose(
            config_name="test",
            overrides=[
                "dataset.snapshot=''",
                "dataset.write_snapshot=False",
            ],
            return_hydra_config=True
        )
        env_mgr = EnvMgr(cfg)
        md_mgr = market_data_mgr.MarketDataMgr(env_mgr)
        #log_setup.setup_logging()
        ray.shutdown()
        ray.init(object_store_memory=30*1024*1024*1024,
                 storage=f"{cfg.dataset.base_dir}/cache",
                 log_to_driver=True)
        workflow.init()
        # You can also script module import loading by knowing the module name
        # See run.py for an example of doing it that way.
        from ats.features.data_loaders import load_data_parquet
        from ats.features.preprocess import price_features, time_features, return_features 
        modules = [load_data_parquet, price_features, time_features, return_features]
        initial_columns = {  # could load data here via some other means, or delegate to a module as we have done.
            "config": env_mgr.config,
            "env_mgr": env_mgr,
            "cal": env_mgr.market_cal,
            "macro_data_builder": md_mgr.macro_data_builder,
            "feast_repository_path":".",
            "feast_config":{},
            "ret_std":env_mgr.config.dataset.ret_std,
            "vol_threshold":5.0,
            "base_price":float(env_mgr.config.dataset.base_price),
            "interval_mins": env_mgr.config.dataset.interval_mins,
            "interval_per_day":int(23 * 60 / env_mgr.config.dataset.interval_mins)
        }
        rga = h_ray.RayWorkflowGraphAdapter(
            result_builder=base.PandasDataFrameResult(),
        #    # Ray will resume a run if possible based on workflow id
            workflow_id=f"wf-{env_mgr.run_id}",
        )
        dr = driver.Driver(initial_columns, *modules, adapter=rga)
        logging.error(f"feature_name:{feature_name}")
        full_data = dr.execute([feature_name])[-k:]
        return full_data
    
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
