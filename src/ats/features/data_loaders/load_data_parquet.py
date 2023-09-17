import datetime
from functools import cached_property, partial
import logging
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

from numba import njit
import wandb
from omegaconf.dictconfig import DictConfig
from ats.calendar import market_time
from ats.event.macro_indicator import MacroDataBuilder
from ats.market_data import data_util
from ats.market_data.data_module import TimeSeriesDataModule
from ats.app.env_mgr import EnvMgr
from ats.util import time_util
from ats.util import profile_util
from ats.features.data_loaders.input_data_utils import *

def sorted_data(config : DictConfig, env_mgr : EnvMgr) -> pd.DataFrame:
    train_data_vec = []
    for ticker in env_mgr.model_tickers:
        ticker_train_data = get_input_for_ticker(
            env_mgr.dataset_base_dir,
            env_mgr.data_start_date,
            env_mgr.data_end_date,
            ticker,
            "FUT",
            env_mgr.time_interval,
        )
        if ticker_train_data is None or ticker_train_data.empty:
            continue
        ticker_train_data["new_idx"] = ticker_train_data.apply(
            lambda x: x.ticker + "_" + str(x.series_idx), axis=1
        )
        ticker_train_data = ticker_train_data.set_index("new_idx")
        train_data_vec.append(ticker_train_data)
    full_data = pd.concat(train_data_vec)
    # logging.info(f"full_data:{full_data.iloc[:2]}")
    # TODO: the original time comes from aggregated time is UTC, but actually
    # has New York time in it.
    full_data["time"] = full_data.time.apply(
        market_time.utc_to_nyse_time,
        interval_minutes=config.dataset.interval_mins,
    )
    full_data["timestamp"] = full_data.time.apply(lambda x: int(x.timestamp()))
    full_data["idx_timestamp"] = full_data["timestamp"]
    # Do not use what are in serialized files as we need to recompute across different months.
    full_data = full_data.drop(
        columns=[
            "close_back",
            "volume_back",
            "dv_back",
            "close_fwd",
            "volume_fwd",
            "dv_fwd",
            "cum_volume",
            "cum_dv",
        ]
    )
    full_data = full_data.reset_index()
    full_data["new_idx"] = full_data.apply(
        lambda x: x.ticker + "_" + str(x.timestamp), axis=1
    )
    full_data = full_data.set_index("new_idx")
    full_data = full_data.sort_index()
    full_data["time_idx"] = range(0, len(full_data))
    full_data = full_data.reset_index()
    full_data = full_data.set_index(["timestamp","ticker"])
    #full_data["timestamp"] = full_data["idx_timestamp"]
    full_data = full_data.sort_index()
    return full_data

@profile_util.profile
def get_processed_data(
    base_dir, start_date, end_date, ticker: str, asset_type: str, time_interval
) -> pd.DataFrame:
    input_dirs = get_input_dirs(
        base_dir, start_date, end_date, ticker, asset_type, time_interval
    )
    ds = ray.data.read_parquet(input_dirs, parallelism=100)
    ds = ds.to_pandas(10000000)
    ds = ds.sort_index()
    ds = ds[~ds.index.duplicated(keep="first")]
    # logging.info(f"ds before filter:{ds.head()}")
    # Filter out between london open until new york close
    # ds = ds[(ds.hour_of_day > 1) & (ds.hour_of_day < 17)]
    # logging.info(f"ds after filter:{ds.head()}")
    # Need to recompute close_back after filtering
    ds_dup = ds[ds.index.duplicated()]
    if not ds_dup.empty:
        logging.info(f"ds_dup:{ds_dup}")
        # exit(0)
    # .join(df_pct_forward, rsuffix='_fwd')
    # logging.info(f"ds:{ds.head()}")
    ds = ds.dropna()
    # logging.info(f"ds:{ds.info()}")
    return ds
