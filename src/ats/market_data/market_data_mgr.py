import datetime
from functools import cached_property, partial
import logging
import numpy as np
import pandas as pd
#import modin.pandas as pd
import os
import ray
import time
import traceback

from ats.calendar import market_time
from ats.event.macro_indicator import MacroDataBuilder
from ats.market_data import data_util
from ats.market_data.data_module import TimeSeriesDataModule


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
        # logging.info(f"all_data:all_data.head()")
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
        logging.info(f"train data after filtering: {train_data.iloc[-30:][['time_idx','ticker','time']]}")
        logging.info(f"eval data after filtering: {eval_data.iloc[-30:][['time_idx','ticker','time']]}")
        logging.info(f"test data after filtering: {test_data.iloc[-30:][['time_idx','ticker','time']]}")
        logging.info(f"full_data:{len(full_data)}, train:{len(train_time_idx)}, eval:{len(eval_time_idx)}, test:{len(test_time_idx)}")
        logging.info(f"full_data:{full_data.describe()}")
        full_data = full_data.sort_values(["ticker", "time"])
        data_module = TimeSeriesDataModule(
            config,
            full_data, train_time_idx, eval_time_idx, test_time_idx,
            env_mgr.targets,
            transformed_full=self.transformed_full
        )
        if self.config.dataset.write_snapshot and self.config.dataset.snapshot:
            data_start_date_str = env_mgr.data_start_date.strftime('%Y%m%d')
            data_end_date_str = env_mgr.data_end_date.strftime('%Y%m%d')
            snapshot_dir = f"{self.config.dataset.snapshot}/{data_start_date_str}_{data_end_date_str}_transformed_full"
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
        except Exception as e:
            logging.error(f"can not read snapshot:{e}")
            # Will try regenerating when reading fails
            pass

        if not full_data is None:
            return full_data

        train_data_vec = []
        refs = []
        for ticker in env_mgr.model_tickers:
            logging.info(f"adding {ticker} from {env_mgr.dataset_base_dir}")
            logging.info(f"{env_mgr.data_start_date}, {env_mgr.data_end_date}")
            ticker_train_data = get_input_for_ticker.remote(
                env_mgr.dataset_base_dir,
                env_mgr.data_start_date,
                env_mgr.data_end_date,
                ticker,
                "FUT",
                env_mgr.time_interval,
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
        full_data = pd.concat(train_data_vec)
        # TODO: the original time comes from aggregated time is UTC, but actually
        # has New York time in it.
        full_data["time"] = full_data.time.apply(
            market_time.utc_to_nyse_time,
            interval_minutes=self.config.job.time_interval_minutes,
        )
        full_data["timestamp"] = full_data.time.apply(lambda x: int(x.timestamp()))
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
        full_data = data_util.add_group_features(self.config.job.time_interval_minutes, full_data)
        full_ds = ray.data.from_pandas(full_data)
        add_example_features = partial(
            data_util.add_example_level_features, self.market_cal, self.macro_data_builder)
        full_ds = full_ds.repartition(100).map_batches(add_example_features, batch_size=4096)
        full_data = full_ds.to_pandas(limit=10000000).sort_index()
        #full_data = data_util.add_example_level_features(self.market_cal, self.macro_data_builder, full_data)
        full_data = data_util.add_example_group_features(self.market_cal, self.macro_data_builder, full_data)

        if self.config.dataset.write_snapshot and self.config.dataset.snapshot:
            ds = ray.data.from_pandas(full_data)
            snapshot_dir = f"{self.config.dataset.snapshot}/{env_mgr.run_id}/{data_start_date_str}_{data_end_date_str}"
            logging.error(f"writing snapshot to {snapshot_dir}")
            os.makedirs(snapshot_dir, exist_ok=True)
            ds.write_parquet(snapshot_dir)
        return full_data
