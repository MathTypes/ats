from collections import defaultdict
import datetime
import logging

import numpy as np
import pandas as pd
import pytz
from pytorch_forecasting.utils import create_mask, detach, to_list
import ray
import torch

from ats.calendar import market_time
from ats.event.macro_indicator import MacroDataBuilder
from ats.market_data.data_module import (
    TransformerDataModule,
    LSTMDataModule,
    TimeSeriesDataModule,
)
from ats.model.models import AttentionEmbeddingLSTM
from ats.market_data import data_util
from ats.model import model_utils
from ats.model.utils import Pipeline
from ats.model import viz_utils
from ats.optimizer import position_utils
from ats.prediction import prediction_utils
from ats.util.profile import profile

@ray.remote
def get_input_for_ticker(
    base_dir, start_date, end_date, ticker, asset_type, time_interval
):
    try:
        from ats.market_data import data_util
        logging.info(f"reading from base_dir:{base_dir}, ticker:{ticker}")
        all_data = data_util.get_processed_data(
            base_dir, start_date, end_date, ticker, asset_type, time_interval
        )
        all_data = all_data.replace([np.inf, -np.inf], np.nan)
        all_data = all_data.dropna()
        all_data = all_data.drop(columns=["time_idx"])
        #logging.info(f"all_data:all_data.head()")
        return all_data
    except Exception as e:
        print(f"can not get input for {ticker}, {e}")
        logging.info(f"can not get input for {ticker}, {e}")
        return None

def get_input_dirs(base_dir, start_date, end_date, ticker, asset_type, time_interval):
    base_dir = f"{base_dir}/{asset_type}/{time_interval}/{ticker}"
    input_dirs = []
    start_date = start_date + datetime.timedelta(days=-60)
    start_date = start_date.replace(day=1)
    # end_date = datetime.datetime.strptime(config.job.test_end_date,"%Y-%m-%d")
    # logging.info(f"looking for {base_dir}, start_date:{start_date}, end_date:{end_date}, {type(end_date)}")
    for cur_date in time_util.monthlist(start_date, end_date):
        for_date = cur_date[0]
        # logging.info(f"checking {for_date}")
        date_dir = os.path.join(base_dir, for_date.strftime("%Y%m%d"))
        # logging.info(f"listing:{date_dir}")
        try:
            files = os.listdir(date_dir)
            files = [
                date_dir + "/" + f for f in files if os.path.isfile(date_dir + "/" + f)
            ]  # Filtering only the files.
            input_dirs.extend(files)
        except:
            logging.warn(f"can not find files under {date_dir}")
    # logging.info(f"reading files:{input_dirs}")
    return input_dirs


def get_snapshot(input_dirs) -> pd.DataFrame:
    ds = ray.data.read_parquet(input_dirs, parallelism=100)
    ds = ds.to_pandas(10000000)
    ds = ds.sort_index()
    ds = ds[~ds.index.duplicated(keep="first")]
    ds_dup = ds[ds.index.duplicated()]
    if not ds_dup.empty:
        logging.info(f"ds_dup:{ds_dup}")
        # exit(0)
    #ds = ds.dropna()
    return ds

class MarketDataMgr(object):
    def __init__(
        self,
        env_mgr
    ):
        super().__init__()
        self.env_mgr = env_mgr
        self.config = env_mgr.config
        self.market_cal = env_mgr.market_cal
        self.macro_data_builder = MacroDataBuilder(self.config)
        self.raw_data = None

    def get_raw_data(self):
        if "initial_snapshot" in self.config.dataset:
            try:
                self.raw_data = get_snapshot(self.config.dataset.initial_snapshot)
            except Exception as e:
                # Will try regenerating when reading fails
                pass
        if not self.raw_data is None:
            return self.raw_data
        
        train_data_vec = []
        refs = []
        env_mgr = self.env_mgr
        for ticker in env_mgr.model_tickers:
            ticker_train_data = get_input_for_ticker.remote(
                env_mgr.dataset_base_dir, env_mgr.train_start_date,
                env_mgr.test_end_date, ticker, "FUT", env_mgr.time_interval
            )
        refs.append(ticker_train_data)
        all_results = ray.get(refs)
        for result in all_results:
            ticker_train_data = result
            if ticker_train_data is None or ticker_train_data.empty:
                continue
            ticker_train_data["new_idx"] = ticker_train_data.apply(
                lambda x: x.ticker + "_" + str(x.series_idx), axis=1
            )
            ticker_train_data = ticker_train_data.set_index("new_idx")
            train_data_vec.append(ticker_train_data)
        raw_data = pd.concat(train_data_vec)
        # TODO: the original time comes from aggregated time is UTC, but actually
        # has New York time in it.
        raw_data["time"] = raw_data.time.apply(market_time.utc_to_nyse_time,
                                               interval_minutes=self.config.job.time_interval_minutes)
        raw_data["timestamp"] = raw_data.time.apply(lambda x: int(x.timestamp()))
        # Do not use what are in serialized files as we need to recompute across different months.
        raw_data = raw_data.drop(columns=["close_back", "volume_back", "dv_back", "close_fwd",
                                          "volume_fwd", "dv_fwd", "cum_volume", "cum_dv"])
        raw_data = raw_data.reset_index()
        raw_data["new_idx"] = raw_data.apply(lambda x: x.ticker + "_" + str(x.timestamp), axis=1)
        raw_data = raw_data.set_index("new_idx")
        raw_data = raw_data.sort_index()
        raw_data["time_idx"] = range(0, len(raw_data))

        raw_data = data_util.add_group_features(raw_data, self.config.job.time_interval_minutes)
        raw_data = data_util.add_example_level_features(raw_data, self.market_cal, self)
        if self.config.dataset.write_snapshot and self.config.dataset.initial_snapshot:
            ds = ray.data.from_pandas(raw_data)
            ds.write_parquet(self.config.dataset.initial_snapshot)
        self.raw_data = raw_data
        return self.raw_data
