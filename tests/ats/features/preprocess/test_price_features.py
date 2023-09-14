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
        print(f"full_data:{full_data}")
        return full_data

def test_close_low_5_ff():
    result = run_features("close_low_5_ff", 50)
    print(f"result:{result['close_low_5_ff'].to_list()}")
    np.testing.assert_array_almost_equal(
        result["close_low_5_ff"],
        [940.25, 940.25, 941.75, 942.75, 943.5, 944.5, 946.75, 946.75, 946.0, 945.75,
         945.75, 945.75, 945.75, 945.25, 945.25, 944.75, 944.75, 944.75, 944.75,
         944.75, 944.5, 944.5, 944.5, 944.5, 944.5, 945.25, 945.5, 945.75, 945.75,
         945.75, 945.75, 947.0, 947.25, 945.75, 945.75, 945.75, 945.75, 945.75,
         949.25, 953.75, 953.75, 953.75, 953.75, 953.75, 954.5, 955.0, 955.5, 957.5, 957.5, 957.5],
        decimal=3
    )

def test_close_high_5_ff():
    result = run_features("close_high_5_ff", 50)
    print(f"result:{result['close_high_5_ff'].to_list()}")
    np.testing.assert_array_almost_equal(
        result["close_high_5_ff"],
        [943.5, 944.5, 947.0, 947.75, 947.75, 947.75, 947.75, 947.75, 947.0, 946.75,
         947.25, 947.25, 947.25, 947.25, 947.25, 946.75, 946.0, 945.75, 945.75, 945.5,
         945.5, 945.5, 945.5, 945.75, 946.0, 947.5, 947.5, 947.5, 948.25, 948.25, 949.0,
         949.0, 949.0, 949.0, 949.25, 959.25, 959.25, 961.5, 961.5, 961.5, 961.5, 961.5,
         955.5, 958.25, 958.25, 960.0, 960.0, 960.0, 960.0, 961.5],
        decimal=3
    )

def test_close_low_21_ff():
    result = run_features("close_low_21_ff", 50)
    print(f"result:{result['close_low_21_ff'].to_list()}")
    np.testing.assert_array_almost_equal(
        result["close_low_21_ff"],
        [936.5, 937.75, 937.75, 937.75, 937.75, 939.0, 939.0, 939.0, 939.5, 939.5,
         940.25, 940.25, 940.25, 940.25, 940.25, 940.25, 940.25, 940.25, 941.75,
         942.75, 943.5, 944.5, 944.5, 944.5, 944.5, 944.5, 944.5, 944.5, 944.5,
         944.5, 944.5, 944.5, 944.5, 944.5, 944.5, 944.5, 944.5, 944.5, 944.5,
         944.5, 944.5, 945.25, 945.5, 945.75, 945.75, 945.75, 945.75, 945.75, 945.75, 945.75],
        decimal=3
    )

def test_close_high_21_ff():
    result = run_features("close_high_21_ff", 50)
    print(f"result:{result['close_high_21_ff'].to_list()}")
    np.testing.assert_array_almost_equal(
        result["close_high_21_ff"],
        [943.5, 944.5, 947.0, 947.75, 947.75, 947.75, 947.75, 947.75, 947.75, 947.75,
         947.75, 947.75, 947.75, 947.75, 947.75, 947.75, 947.75, 947.75, 947.75,
         947.75, 947.75, 947.75, 947.75, 947.75, 947.25, 947.5, 947.5, 947.5,
         948.25, 948.25, 949.0, 949.0, 949.0, 949.0, 949.25, 959.25, 959.25, 961.5,
         961.5, 961.5, 961.5, 961.5, 961.5, 961.5, 961.5, 961.5, 961.5, 961.5, 961.5, 961.5],
        decimal=3
    )

def test_close():
    result = run_features("close", 50)
    print(f"result:{result['close'].to_list()}")
    np.testing.assert_array_almost_equal(
        result["close"],
        [943.5, 944.5, 947.0, 947.75, 947.0, 946.75, 946.75, 946.75, 946.0, 945.75,
         947.25, 946.75, 946.0, 945.25, 945.75, 944.75, 945.5, 945.0, 945.5, 945.0,
         944.5, 945.25, 945.5, 945.75, 946.0, 947.5, 945.75, 947.0, 948.25, 948.25,
         949.0, 948.25, 947.25, 945.75, 949.25, 959.25, 957.5, 961.5, 954.0, 953.75,
         954.5, 955.0, 955.5, 958.25, 958.0, 960.0, 960.0, 957.5, 959.25, 961.],
        decimal=3
    )

