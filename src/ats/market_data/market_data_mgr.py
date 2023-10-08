import datetime
from functools import cached_property, partial
import logging
import pathlib
import numpy as np
import pandas as pd
#import modin.pandas as pd
import os
import ray
from ray import workflow
import time
import traceback

from hamilton import base, driver, log_setup
from hamilton.experimental import h_ray
from hamilton.experimental import h_cache

from numba import njit
import wandb

from ats.calendar import market_time
from ats.event.macro_indicator import MacroDataBuilder
from ats.market_data import data_util
from ats.market_data.data_module import TimeSeriesDataModule


#@ray.remote
def read_snapshot(input_dirs) -> pd.DataFrame:
    ds = ray.data.read_parquet(input_dirs, parallelism=10).sort("time_idx")
    ds = ds.to_pandas(10000000)
    #ds = pd.read_parquet(input_dirs)
    ds = ds.sort_index()
    ds = ds[~ds.index.duplicated(keep="first")]
    ds_dup = ds[ds.index.duplicated()]
    if not ds_dup.empty:
        logging.info(f"ds_dup:{ds_dup}")
        # exit(0)
    # ds = ds.dropna()
    return ds


class MarketDataMgr(object):
    def __init__(self, env_mgr):
        super().__init__()
        self.env_mgr = env_mgr
        self.config = env_mgr.config
        self.market_cal = env_mgr.market_cal
        self.macro_data_builder = MacroDataBuilder(self.env_mgr)
        self._FULL_DATA = None
        self._DATA_MODULE = None

    
    #@cached_property
    def full_data(self):
        logging.info(f"self._FULL_DATA:{self._FULL_DATA}")
        if self._FULL_DATA is None:
            logging.info("costly full_data creation")
            self._FULL_DATA = self._get_snapshot()
            if wandb.run:
                run_id = wandb.run.id
                data_artifact = wandb.Artifact(f"run_{run_id}_data_viz", type="data_viz")
                data_table = wandb.Table(dataframe=self._FULL_DATA.sample(frac=0.01, replace=False, random_state=1))
                data_artifact.add(data_table, "raw_data")
                wandb.run.use_artifact(data_artifact)
        return self._FULL_DATA

    #@cached_property
    def data_module(self):
        logging.info(f"self._DATA_MODULE:{self._DATA_MODULE}")
        if self._DATA_MODULE is None:
            logging.info("costly data_module creation")
            self._DATA_MODULE = self.create_data_module()
        logging.info(f"return_data_module:{self._DATA_MODULE}")
        return self._DATA_MODULE
    
    def create_data_module(self):
        env_mgr = self.env_mgr
        config = env_mgr.config
        start = time.time()
        full_data = self.full_data()
        full_data.replace([np.inf, -np.inf], np.nan,inplace=True)
        full_data = full_data.fillna(-1)
        logging.info(f"full_data, max_time:{full_data.timestamp.max()}, min_time:{full_data.timestamp.min()}")
        logging.info(f"env_mgr.train_start_timestamp:{env_mgr.train_start_timestamp}, test_start:{env_mgr.test_start_timestamp}, test_end:{env_mgr.test_end_timestamp}")
        # TODO: full_data might need to extend beyond test_end_timestamp to pick up
        # lead data
        full_data = full_data[
            (full_data.timestamp >= env_mgr.train_start_timestamp)
            & (full_data.timestamp <= env_mgr.test_end_timestamp)
        ]
        # TODO: it is a hack to add env_mgr.test_start_timestamp to train_data.
        # The reason is that without it, train_data would end at 16:30 EST prior day and
        # eval data would start at 18:30 EST prior day. So we have a 18:00 missing.
        train_data = full_data[
            (full_data.timestamp >= env_mgr.train_start_timestamp)
            & (full_data.timestamp <= env_mgr.test_start_timestamp)
        ]
        eval_data = full_data[
            (full_data.timestamp >= env_mgr.eval_start_timestamp)
            & (full_data.timestamp <= env_mgr.eval_end_timestamp)
        ]
        test_data = full_data[
            (full_data.timestamp >= env_mgr.test_start_timestamp)
            & (full_data.timestamp <= env_mgr.test_end_timestamp)
        ]
        train_time_idx = train_data["time_idx"]    
        eval_time_idx = eval_data["time_idx"]    
        test_time_idx = test_data["time_idx"]    
        logging.info(f"full data after filtering: {full_data.iloc[-3:]}")
        logging.info(f"full_data:{len(full_data)}, train:{len(train_time_idx)}, eval:{len(eval_time_idx)}, test:{len(test_time_idx)}")
        logging.info(f"full_data:{full_data.describe()}")
        full_data = full_data.sort_values(["ticker", "timestamp"])
        data_module = TimeSeriesDataModule(
            config,
            full_data, train_time_idx, eval_time_idx, test_time_idx,
            env_mgr.targets,
            transformed_full=self.transformed_full
        )
        if self.config.dataset.write_snapshot and self.config.dataset.snapshot:
            data_start_date_str = env_mgr.data_start_date.strftime('%Y%m%d')
            data_end_date_str = env_mgr.data_end_date.strftime('%Y%m%d')
            snapshot_dir = f"{self.config.dataset.snapshot}/{env_mgr.run_id}/{data_start_date_str}_{data_end_date_str}_transformed_full"
            ds = ray.data.from_pandas(data_module.full.transformed_data)
            os.makedirs(snapshot_dir, exist_ok=True)
            ds.write_parquet(snapshot_dir)
        
        return data_module

    
    def _get_snapshot(self):
        full_data = None
        env_mgr = self.env_mgr
        data_start_date_str = env_mgr.data_start_date.strftime('%Y%m%d')
        data_end_date_str = env_mgr.data_end_date.strftime('%Y%m%d')
        snapshot_dir = f"{self.config.dataset.snapshot}/{data_start_date_str}_{data_end_date_str}"
        logging.info(f"checking snapshot:{snapshot_dir}")
        self.transformed_full = None
        try:
            if self.config.dataset.read_snapshot and os.listdir(f"{snapshot_dir}"):
                logging.info(f"reading snapshot from {snapshot_dir}")
                full_data = read_snapshot(snapshot_dir)
                snapshot_dir = f"{self.config.dataset.snapshot}/{data_start_date_str}_{data_end_date_str}_transformed_full"
                self.transformed_full = read_snapshot(snapshot_dir)        
        except Exception:
            # Will try regenerating when reading fails
            pass

        if not full_data is None:
            return full_data

        log_setup.setup_logging()
        workflow.init()
        # You can also script module import loading by knowing the module name
        # See run.py for an example of doing it that way.
        from ats.features.data_loaders import load_data_parquet
        from ats.features.preprocess import price_features, time_features, return_features 
        modules = [load_data_parquet, price_features, time_features, return_features]
        initial_columns = {  # could load data here via some other means, or delegate to a module as we have done.
            "config": self.config,
            "env_mgr": env_mgr,
            "cal": self.market_cal,
            "macro_data_builder": self.macro_data_builder,
            "feast_repository_path":".",
            "feast_config":{},
            "ret_std":self.config.dataset.ret_std,
            "vol_threshold":5.0,
            "base_price":float(self.config.dataset.base_price),
            "interval_mins": self.config.dataset.interval_mins,
            "interval_per_day":int(23 * 60 / self.config.dataset.interval_mins)
        }
        cache_path = "/tmp/hamilton_cache"
        pathlib.Path(cache_path).mkdir(exist_ok=True)
        rga = h_ray.RayWorkflowGraphAdapter(
            result_builder=base.PandasDataFrameResult(),
            workflow_id=f"wf-{env_mgr.run_id}",
        )
        dr = driver.Driver(initial_columns, *modules, adapter=rga)
    
        output_columns = self.config.model.features.time_varying_known_reals
        full_data = dr.execute(["example_group_features"])
        logging.error(f"full_data before filtering:{full_data.describe()}")
        #ret_from_vwap_around_london_close
        logging.error(f"full_data.ret_from_vwap_around_london_close>0.15:{full_data[full_data.ret_from_vwap_around_london_close>0.5].iloc[-3:]}")
        logging.error(f"full_data.ret_from_vwap_around_london_open>0.15:{full_data[full_data.ret_from_vwap_around_london_open>0.15].iloc[-3:]}")
        logging.error(f"full_data.ret_from_close_cumsum_low_51d<-0.24:{full_data[full_data.ret_from_close_cumsum_low_51d<-0.24].iloc[-10:]}") 
        logging.error(f"full_data.ret_from_close_cumsum_low_201d<-0.3:{full_data[full_data.ret_from_close_cumsum_low_201d<-0.3].iloc[-10:]}")
        logging.error(f"full_data.ret_from_close_cumsum_high_201d>0.3:{full_data[full_data.ret_from_close_cumsum_high_201d>0.3].iloc[-10:]}")
        logging.error(f"full_data.ret_from_vwap_since_last_macro_event_imp1>0.2:{full_data[full_data.ret_from_vwap_since_last_macro_event_imp1>0.2].iloc[-10:]}") 
        logging.error(f"full_data.ret_from_vwap_since_last_macro_event_imp2<0:{full_data[full_data.ret_from_vwap_since_last_macro_event_imp2<-0.15].iloc[-10:]}") 
        logging.error(f"full_data.ret_from_vwap_since_last_macro_event_imp2>0.2:{full_data[full_data.ret_from_vwap_since_last_macro_event_imp2>0.2].iloc[-10:]}") 
        logging.error(f"full_data.ret_from_vwap_around_london_open>0.2:{full_data[full_data.ret_from_vwap_around_london_open>0.2].iloc[-10:]}") 
        logging.error(f"full_data.ret_from_vwap_around_london_close>0.3:{full_data[full_data.ret_from_vwap_around_london_close>0.3].iloc[-10:]}") 
        logging.error(f"full_data.ret_from_vwap_around_new_york_close>0.3:{full_data[full_data.ret_from_vwap_around_new_york_close>0.3].iloc[-10:]}") 
        logging.error(f"full_data.ret_from_vwap_around_new_york_open>0.3:{full_data[full_data.ret_from_vwap_around_new_york_open>0.3].iloc[-10:]}")
        logging.error(f"full_data.ret_from_vwap_since_london_open>0.3:{full_data[full_data.ret_from_vwap_since_london_open>0.3].iloc[-10:]}")
        logging.error(f"full_data.ret_from_vwap_since_last_macro_event_imp3<0:{full_data[full_data.ret_from_vwap_since_last_macro_event_imp3<-0.15].iloc[-10:]}") 
        logging.error(f"full_data.ret_from_vwap_since_last_macro_event_imp3>0.2:{full_data[full_data.ret_from_vwap_since_last_macro_event_imp3>0.2].iloc[-10:]}") 

        logging.info(f"full_data:{full_data.describe()}")
        if self.config.dataset.write_snapshot and self.config.dataset.snapshot:
            ds = ray.data.from_pandas(full_data)
            snapshot_dir = f"{self.config.dataset.snapshot}/{env_mgr.run_id}/{data_start_date_str}_{data_end_date_str}"
            logging.error(f"writing snapshot to {snapshot_dir}")
            os.makedirs(snapshot_dir, exist_ok=True)
            ds.write_parquet(snapshot_dir)
        return full_data
