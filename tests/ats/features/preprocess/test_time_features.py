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
    
def test_time_low_21d():
    full_data = run_features("time_low_21d_ff")
    print(f"full_data:{full_data}")
    np.testing.assert_array_almost_equal(
        full_data["timestamp"],
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
        result["timestamp"],
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

def test_time_to_low_5_ff():
    result = run_features("time_to_low_5_ff", 5)
    #print(f"result:{result}")
    np.testing.assert_array_almost_equal(
        result["time_to_low_5_ff"],
        [10800, 12600, 0, 1800, 3600],
        decimal=3
    )

def test_time_high_1d_ff_shift_1d():
    result = run_features("time_high_1d_ff_shift_1d", 100).iloc[:10]
    #print(f"result:{result['timestamp'].to_list()}")
    np.testing.assert_array_almost_equal(
        result["timestamp"],
        [1283437800.0, 1283437800.0, 1283437800.0, 1283450400.0, 1283452200.0, 1283454000.0, 1283455800.0, 1283457600.0, 1283457600.0, 1283457600.0],
        decimal=3
    )
