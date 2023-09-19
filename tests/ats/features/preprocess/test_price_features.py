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

def test_joined_last_daily_close_0():
    result = run_features("joined_last_daily_close_0", 200)
    logging.error(f"result:{result.iloc[10:110]}")
    close_list = result['joined_last_daily_close_0'][10:110]
    logging.error(f"close:{close_list.to_list()}")
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    np.testing.assert_array_almost_equal(
        close_list,
        [931.25, 929.25, 892.25, 890.5, 879.75, 878.0, 874.75, 909.25, 920.25, 928.75],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1277409600, 1277496000, 1277755200, 1277841600, 1277928000,
         1278014400, 1278100800, 1278446400, 1278532800, 1278619200],
        decimal=3
    )
    
def test_last_daily_close_0():
    result = run_features("last_daily_close_0", 50)
    close_list = result['last_daily_close_0'][:10]
    logging.error(f"close:{close_list.to_list()}")
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    np.testing.assert_array_almost_equal(
        close_list,
        [931.25, 929.25, 892.25, 890.5, 879.75, 878.0, 874.75, 909.25, 920.25, 928.75],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1277409600, 1277496000, 1277755200, 1277841600, 1277928000,
         1278014400, 1278100800, 1278446400, 1278532800, 1278619200],
        decimal=3
    )

def test_last_daily_close_1():
    result = run_features("last_daily_close_1", 50)
    logging.error(f"result:{result}")
    open_list = result['last_daily_close_1'][:10]
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    np.testing.assert_array_almost_equal(
        open_list,
        [928.5 , 931.25, 929.25, 892.25, 890.5 , 879.75, 878.  , 874.75,
         909.25, 920.25],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1277409600, 1277496000, 1277755200, 1277841600, 1277928000,
         1278014400, 1278100800, 1278446400, 1278532800, 1278619200],
        decimal=3
    )

def test_last_daily_close_2():
    result = run_features("last_daily_close_2", 50)
    logging.error(f"result:{result}")
    open_list = result['last_daily_close_2'][:10]
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    np.testing.assert_array_almost_equal(
        open_list,
        [946.25, 928.5 , 931.25, 929.25, 892.25, 890.5 , 879.75, 878.  ,
         874.75, 909.25],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1277409600, 1277496000, 1277755200, 1277841600, 1277928000,
         1278014400, 1278100800, 1278446400, 1278532800, 1278619200],
        decimal=3
    )

def test_daily_close_df():
    result = run_features("daily_close_df", 50)
    logging.error(f"result:{result}")
    daily_close_0_list = result['daily_close_0'][:10]
    daily_close_1_list = result['daily_close_1'][:10]
    daily_close_2_list = result['daily_close_2'][:10]
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    np.testing.assert_array_almost_equal(
        daily_close_0_list,
        [931.25, 929.25, 892.25, 890.5 , 879.75, 878.  , 874.75, 909.25,
         920.25, 928.75],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        daily_close_1_list,
        [928.5 , 931.25, 929.25, 892.25, 890.5 , 879.75, 878.  , 874.75,
         909.25, 920.25],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        daily_close_2_list,
        [946.25, 928.5 , 931.25, 929.25, 892.25, 890.5 , 879.75, 878.  ,
         874.75, 909.25],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1277409600, 1277496000, 1277755200, 1277841600, 1277928000,
         1278014400, 1278100800, 1278446400, 1278532800, 1278619200],
        decimal=3
    )

def test_daily_open():
    result = run_features("daily_open", 50)
    close_list = result.query('(close_time>=1277409600) and (close_time<=1278619200)')['daily_open']
    close_time_list = close_list.index.get_level_values(level=0)[:10].to_list()
    np.testing.assert_array_almost_equal(
        close_list,
        [928.  , 932.75, 928.25, 893.  , 882.25, 880.  , 872.75, 881.75, 916.5], 
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1277496000, 1277755200, 1277841600, 1277928000, 1278014400,
         1278100800, 1278446400, 1278532800, 1278619200],
        decimal=3
    )


def test_daily_close_series():
    result = run_features("daily_close", 50)
    close_list = result.query('(close_time>=1277409600) and (close_time<=1278619200)')['daily_close']
    close_time_list = close_list.index.get_level_values(level=0)[:10].to_list()
    np.testing.assert_array_almost_equal(
        close_list,
        [932.5, 928.25, 892.75, 882.25, 879.75, 872.0, 881.75, 917.0, 924.75],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1277496000, 1277755200, 1277841600, 1277928000, 1278014400,
         1278100800, 1278446400, 1278532800, 1278619200],
        decimal=3
    )

def test_daily_high_series():
    result = run_features("daily_high", 50)
    close_list = result.query('(close_time>=1277409600) and (close_time<=1278619200)')['daily_high']
    close_time_list = close_list.index.get_level_values(level=0)[:10].to_list()
    np.testing.assert_array_almost_equal(
        close_list,
        [937.25, 937.5 , 932.5 , 902.25, 887.5 , 890.25, 896.25, 917.5 ,
         926.25],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1277496000, 1277755200, 1277841600, 1277928000, 1278014400,
         1278100800, 1278446400, 1278532800, 1278619200],
        decimal=3
    )

def test_daily_low_series():
    result = run_features("daily_low", 50)
    close_list = result.query('(close_time>=1277409600) and (close_time<=1278619200)')['daily_low']
    close_time_list = close_list.index.get_level_values(level=0)[:10].to_list()
    np.testing.assert_array_almost_equal(
        close_list,
        [920.5 , 924.25, 888.  , 880.75, 863.75, 868.5 , 860.5 , 874.  ,
         911.75],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1277496000, 1277755200, 1277841600, 1277928000, 1278014400,
         1278100800, 1278446400, 1278532800, 1278619200],
        decimal=3
    )

def test_weekly_open():
    result = run_features("weekly_open", 50)
    open_list = result['weekly_open'][:10]
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    np.testing.assert_array_almost_equal(
        open_list,
        [903.5 , 888.  , 866.  , 912.  , 924.5 , 920.75, 875.5 , 910.  ,
         936.25, 934.],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1253908800, 1254513600, 1255118400, 1255723200, 1256328000,
         1256932800, 1257541200, 1258146000, 1258750800, 1259344800],
        decimal=3
    )

def test_weekly_close():
    result = run_features("weekly_close", 50)
    close_list = result['weekly_close'][:10]
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    # TODO: there seems to be a bug with first Friday.
    # 2009-10-02 15:30:00  865.75  868.00  863.25  865.25  161658  1.399255e+08   
    # 2009-10-02 16:00:00  865.25  866.75  864.25  866.25   90128  7.802734e+07
    # 15:30:00 is treated as last instead of 16:00:00.
    np.testing.assert_array_almost_equal(
        close_list,
        [885.5 , 865.25, 912.5 , 926.5 , 921.  , 876.5 , 910.5 , 935.75,
         934.5 , 933.5],
        decimal=3
    )
    # 2009-09-25 16:00:00
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1253908800, 1254513600, 1255118400, 1255723200, 1256328000,
         1256932800, 1257541200, 1258146000, 1258750800, 1259344800],
        decimal=3
    )

def test_weekly_high():
    result = run_features("weekly_high", 50)
    high_list = result['weekly_high'][:10]
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    np.testing.assert_array_almost_equal(
        high_list,
        [920.  , 910.  , 912.75, 939.75, 943.25, 932.75, 913.75, 947.5 ,
         956.5 , 955.75],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1253908800, 1254513600, 1255118400, 1255723200, 1256328000,
         1256932800, 1257541200, 1258146000, 1258750800, 1259344800],
        decimal=3
    )

def test_weekly_low():
    result = run_features("weekly_low", 50)
    low_list = result['weekly_low'][:10]
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    np.testing.assert_array_almost_equal(
        low_list,
        [880.5 , 856.25, 863.75, 907.25, 914.5 , 873.75, 870.25, 909.25,
         927.75, 911.25],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1253908800, 1254513600, 1255118400, 1255723200, 1256328000,
         1256932800, 1257541200, 1258146000, 1258750800, 1259344800],
        decimal=3
    )

def test_monthly_open():
    result = run_features("monthly_open", 50)
    open_list = result['monthly_open'][:10]
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    np.testing.assert_array_almost_equal(
        open_list,
        [1173.25, 1157.75, 1220.25, 1234.25, 1114.  , 1100.  , 1116.75,
         1005.75,  803.  ,  726.25],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1204318800, 1206993600, 1209585600, 1212177600, 1214856000,
         1217534400, 1220040000, 1222804800, 1225483200, 1227895200],
        decimal=3
    )

def test_monthly_close():
    result = run_features("monthly_close", 50)
    close_list = result['monthly_close'][:10]
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    np.testing.assert_array_almost_equal(
        close_list,
        [1157.  , 1221.25, 1236.25, 1118.  , 1100.75, 1116.75, 1000.  ,
         798.5 ,  726.75,  724.5],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1206993600, 1209585600, 1212177600, 1214856000, 1217534400,
         1220040000, 1222804800, 1225483200, 1227895200, 1230757200],
        decimal=3
    )

def test_monthly_high():
    result = run_features("monthly_high", 50)
    high_list = result['monthly_high'][:10]
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    np.testing.assert_array_almost_equal(
        high_list,
        [1175.5 , 1239.  , 1276.  , 1244.5 , 1127.25, 1144.25, 1133.  ,
         1005.75,  839.  ,  748.25],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1204318800, 1206993600, 1209585600, 1212177600, 1214856000,
         1217534400, 1220040000, 1222804800, 1225483200, 1227895200],
        decimal=3
    )

def test_monthly_low():
    result = run_features("monthly_low", 50)
    low_list = result['monthly_low'][:10]
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    np.testing.assert_array_almost_equal(
        low_list,
        [1147.  , 1152.  , 1207.25, 1109.25, 1038.75, 1080.75,  947.75,
         661.  ,  577.  ,  646.25],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1204318800, 1206993600, 1209585600, 1212177600, 1214856000,
         1217534400, 1220040000, 1222804800, 1225483200, 1227895200],
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


