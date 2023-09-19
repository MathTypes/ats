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
    
def test_time_low_21d_ff():
    full_data = run_features("time_low_21d_ff")
    print(f"full_data:{full_data}")
    np.testing.assert_array_almost_equal(
        full_data["close_timestamp"],
        [1283252400, 1283252400, 1283252400, 1283252400, 1283252400,
         1283252400, 1283252400, 1283252400, 1283252400, 1283252400],
        decimal=3
    )

def test_time_low_21_ff():
    full_data = run_features("time_low_21_ff")
    print(f"full_data:{full_data}")
    np.testing.assert_array_almost_equal(
        full_data["close_timestamp"],
        [1283252400, 1283252400, 1283252400, 1283252400, 1283252400,
         1283252400, 1283252400, 1283252400, 1283252400, 1283252400],
        decimal=3
    )

def test_new_york_close_time():
    result = run_features("new_york_close_time", 5)
    #print(f"result:{result}")
    np.testing.assert_array_almost_equal(
        result["new_york_close_time"],
        [1283544000, 1283544000, 1283544000, 1283544000, 1283544000],
        decimal=3
    )
    
def test_time_low_5_ff():
    result = run_features("time_low_5_ff", 5)
    #print(f"result:{result['timestamp']}")
    np.testing.assert_array_almost_equal(
        result["close_timestamp"],
        [1283524200, 1283524200, 1283538600, 1283538600, 1283538600],
        decimal=3
    )

def test_extract_time_feature():
    result = run_features("time", 5)
    timestamps = [x.timestamp() for x in result["time"]]
    np.testing.assert_array_almost_equal(
        timestamps,
        [1283535000.0, 1283536800.0, 1283538600.0, 1283540400.0, 1283542200.0],
        decimal=3
    )

def test_extract_week_of_year():
    result = run_features("week_of_year", 500)["week_of_year"]
    np.testing.assert_array_almost_equal(
        result[:3], [33, 33, 33],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        result[-3:], [35, 35, 35],
        decimal=3
    )

def test_extract_month_of_year():
    result = run_features("month_of_year", 500)["month_of_year"]
    np.testing.assert_array_almost_equal(
        result[:3],
        [8,8,8],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        result[-3:],
        [9,9,9],
        decimal=3
    )

def test_time_to_low_5_ff():
    result = run_features("time_to_low_5_ff", 5)
    #print(f"result:{result}")
    np.testing.assert_array_almost_equal(
        result["time_to_low_5_ff"],
        [10800, 12600, 0, 1800, 3600],
        decimal=3
    )

def test_new_york_last_open_time():
    result = run_features("new_york_last_open_time", 5)
    #print(f"result:{result}")
    np.testing.assert_array_almost_equal(
        result["new_york_last_open_time"],
        [1283520600, 1283520600, 1283520600, 1283520600, 1283520600],
        decimal=3
    )
    
def test_next_daily_close_time():
    result = run_features("next_daily_close_time", 100)[10:60]
    close_time_list = result.index.get_level_values(level=0).to_list()
    res = result["next_daily_close_time"].to_list()
    print(f"res:{res}")
    print(f"close_time_list:{close_time_list}")
    np.testing.assert_array_almost_equal(
        res,
        [1283457600, 1283457600, 1283457600, 1283457600, 1283457600,
         1283457600, 1283457600, 1283457600, 1283457600, 1283457600,
         1283457600, 1283457600, 1283457600, 1283457600, 1283457600,
         1283457600, 1283457600, 1283457600, 1283457600, 1283457600,
         1283457600, 1283457600, 1283457600, 1283457600, 1283457600,
         1283457600, 1283457600, 1283457600, 1283457600, 1283457600,
         1283457600, 1283457600, 1283457600, 1283457600, 1283457600,
         1283457600, 1283457600, 1283457600, 1283457600, 1283457600,
         1283457600, 1283457600, 1283457600, 1283457600, 1283544000,
         1283544000, 1283544000, 1283544000, 1283544000, 1283544000],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1283380200, 1283382000, 1283383800, 1283385600, 1283387400,
         1283389200, 1283391000, 1283392800, 1283394600, 1283396400,
         1283398200, 1283400000, 1283401800, 1283403600, 1283405400,
         1283407200, 1283409000, 1283410800, 1283412600, 1283414400,
         1283416200, 1283418000, 1283419800, 1283421600, 1283423400,
         1283425200, 1283427000, 1283428800, 1283430600, 1283432400,
         1283434200, 1283436000, 1283437800, 1283439600, 1283441400,
         1283443200, 1283445000, 1283446800, 1283448600, 1283450400,
         1283452200, 1283454000, 1283455800, 1283457600, 1283459400,
         1283461200, 1283464800, 1283466600, 1283468400, 1283470200],
        decimal=3
    )

def test_next_weekly_close_time_1254513600():
    result = run_features("next_weekly_close_time", 100)[10:60]
    print(f"result:{result}")
    res = result["next_weekly_close_time"].to_list()
    np.testing.assert_array_almost_equal(
        res,
        [1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000],
        decimal=3
    )

def test_next_weekly_close_time():
    result = run_features("next_weekly_close_time", 100)[10:60]
    close_time_list = result.index.get_level_values(level=0).to_list()
    res = result["next_weekly_close_time"].to_list()
    print(f"res:{res}")
    print(f"close_time_list:{close_time_list}")
    np.testing.assert_array_almost_equal(
        res,
        [1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000, 1283544000],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1283380200, 1283382000, 1283383800, 1283385600, 1283387400, 1283389200, 1283391000, 1283392800, 1283394600, 1283396400, 1283398200, 1283400000, 1283401800, 1283403600, 1283405400, 1283407200, 1283409000, 1283410800, 1283412600, 1283414400, 1283416200, 1283418000, 1283419800, 1283421600, 1283423400, 1283425200, 1283427000, 1283428800, 1283430600, 1283432400, 1283434200, 1283436000, 1283437800, 1283439600, 1283441400, 1283443200, 1283445000, 1283446800, 1283448600, 1283450400, 1283452200, 1283454000, 1283455800, 1283457600, 1283459400, 1283461200, 1283464800, 1283466600, 1283468400, 1283470200],
        decimal=3
    )

def test_next_monthly_close_time():
    result = run_features("next_monthly_close_time", 100)[10:60]
    close_time_list = result.index.get_level_values(level=0).to_list()
    res = result["next_monthly_close_time"].to_list()
    print(f"res:{res}")
    print(f"close_time_list:{close_time_list}")
    np.testing.assert_array_almost_equal(
        res,
        [1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800, 1285876800],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1283380200, 1283382000, 1283383800, 1283385600, 1283387400, 1283389200, 1283391000, 1283392800, 1283394600, 1283396400, 1283398200, 1283400000, 1283401800, 1283403600, 1283405400, 1283407200, 1283409000, 1283410800, 1283412600, 1283414400, 1283416200, 1283418000, 1283419800, 1283421600, 1283423400, 1283425200, 1283427000, 1283428800, 1283430600, 1283432400, 1283434200, 1283436000, 1283437800, 1283439600, 1283441400, 1283443200, 1283445000, 1283446800, 1283448600, 1283450400, 1283452200, 1283454000, 1283455800, 1283457600, 1283459400, 1283461200, 1283464800, 1283466600, 1283468400, 1283470200],
        decimal=3
    )

def test_time_high_1d_ff_shift_1d():
    result = run_features("time_high_1d_ff_shift_1d", 100).iloc[:10]
    np.testing.assert_array_almost_equal(
        result["close_timestamp"],
        [1283437800.0, 1283437800.0, 1283437800.0, 1283450400.0, 1283452200.0, 1283454000.0, 1283455800.0, 1283457600.0, 1283457600.0, 1283457600.0],
        decimal=3
    )


def test_time_to_high_51_ff():
    result = run_features("time_to_high_51_ff", 500).iloc[100:110]
    logging.error(f"result:{result['time_to_high_51_ff'].to_list()}")
    np.testing.assert_array_almost_equal(
        result["time_to_high_51_ff"],
        [64800.0, 66600.0, 68400.0, 70200.0, 72000.0, 73800.0, 75600.0, 77400.0, 79200.0, 81000.0],
        decimal=3
    )

def test_time_to_high_5_ff():
    result = run_features("time_to_high_5_ff", 500).iloc[100:120]
    logging.error(f"result:{result['time_to_high_5_ff'].to_list()}")
    np.testing.assert_array_almost_equal(
        result["time_to_high_5_ff"],
        [5400.0, 7200.0, 9000.0, 10800.0, 12600.0, 14400.0, 16200.0, 0.0,
         1800.0, 3600.0, 5400.0, 7200.0, 9000.0, 10800.0, 12600.0, 0.0,
         0.0, 1800.0, 3600.0, 0.0],
        decimal=3
    )

def test_time_to_low_5_ff():
    result = run_features("time_to_low_5_ff", 500).iloc[100:120]
    logging.error(f"result:{result['time_to_low_5_ff'].to_list()}")
    np.testing.assert_array_almost_equal(
        result["time_to_low_5_ff"],
        [0.0, 0.0, 1800.0, 3600.0, 0.0, 1800.0, 3600.0, 5400.0, 0.0, 0.0, 1800.0, 3600.0, 0.0, 0.0, 1800.0, 3600.0, 5400.0, 7200.0, 0.0, 1800.0],
        decimal=3
    )

def test_is_new_york_close_time():
    result = run_features("is_new_york_close_time", 50)
    print(f"result:{result['is_new_york_close_time'].to_list()}")
    np.testing.assert_array_almost_equal(
        result['is_new_york_close_time'],
        [False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        decimal=3
    )

