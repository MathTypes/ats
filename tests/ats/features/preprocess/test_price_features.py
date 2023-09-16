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
def run_features(features, k=10):
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
                "job.eval_start_date=2009-08-01",
                "job.eval_end_date=2009-10-01",                
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
        logging.error(f"features:{features}")
        if not isinstance(features, list):
            features = [features]
        full_data = dr.execute(features)[-k:]
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

def test_daily_open():
    result = run_features("daily_open", 50)
    open_list = result['daily_open'][:10]
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    np.testing.assert_array_almost_equal(
        open_list,
        [927.25, 932.  , 927.75, 894.75, 883.75, 880.75, 875.  , 881.25,
         913.75, 924.25],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1277409600, 1277496000, 1277755200, 1277841600, 1277928000,
         1278014400, 1278100800, 1278446400, 1278532800, 1278619200],
        decimal=3
    )

def test_daily_close():
    result = run_features("daily_close", 50)
    close_list = result['daily_close'][:10]
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    np.testing.assert_array_almost_equal(
        close_list,
        [931.25, 929.25, 892.25, 890.5 , 879.75, 878.  , 874.75, 909.25,
         920.25, 928.75],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1277409600, 1277496000, 1277755200, 1277841600, 1277928000,
         1278014400, 1278100800, 1278446400, 1278532800, 1278619200],
        decimal=3
    )

def test_daily_high():
    result = run_features("daily_high", 50)
    high_list = result['daily_high'][:10]
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    np.testing.assert_array_almost_equal(
        high_list,
        [935.5,936.5 , 932.25, 901.  , 885.25, 887.  , 893.25, 909.25,
         922.25, 930.75],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1277409600, 1277496000, 1277755200, 1277841600, 1277928000,
         1278014400, 1278100800, 1278446400, 1278532800, 1278619200],
        decimal=3
    )

def test_daily_low():
    result = run_features("daily_low", 50)
    low_list = result['daily_low'][:10]
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    np.testing.assert_array_almost_equal(
        low_list,
        [921.75, 926.5 , 892.25, 890.5 , 867.5 , 870.5 , 862.  , 875.  ,
         913.75, 922],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1277409600, 1277496000, 1277755200, 1277841600, 1277928000,
         1278014400, 1278100800, 1278446400, 1278532800, 1278619200],
        decimal=3
    )

def test_close_low_1d_ff_shift_1d():
    result = run_features("close_low_1d_ff_shift_1d", 50)
    close = result["close"][:4]
    timestamp = result["timestamp"][:4]
    np.testing.assert_array_almost_equal(
        close,
        [944.5, 944.5, 944.5, 944.5],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        timestamp,
        [1283536800, 1283538600, 1283540400, 1283542200],
        decimal=3
    )

def test_close_low_1d_ff():
    result = run_features("close_low_1d_ff", 50)
    close = result[:4]
    np.testing.assert_array_almost_equal(
        close,
        [0.0018, 0.0011, -0.0009, 0.0002, 0.0013],
        decimal=3
    )


