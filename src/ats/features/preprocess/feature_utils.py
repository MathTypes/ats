import datetime
import logging
import math
import os
from typing import Dict

from numba import njit
import numpy as np
import pandas as pd
from pyarrow import csv
import ray
import ta

import pandas_market_calendars as mcal
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler

from ats.calendar import market_time
from ats.model.mom_trans.classical_strategies import (
    MACDStrategy,
    calc_returns,
    calc_daily_vol,
    calc_skew,
    calc_kurt
)
from ats.util import time_util
from ats.util import profile_util

VOL_THRESHOLD = 5  # multiple to winsorise by
HALFLIFE_WINSORISE = 252


def get_tick_data_with_ray(
    ticker: str, asset_type: str, start_date, end_date, raw_dir
) -> pd.DataFrame:
    # ticker = ticker.replace("CME_","")
    names = ["Time", "Open", "High", "Low", "Close", "Volume"]
    if asset_type in ["FUT"]:
        file_path = os.path.join(
            f"{raw_dir}/futures", f"{ticker}_1min_continuous_adjusted.txt"
        )
    else:
        file_path = os.path.join(
            f"{raw_dir}/stock", f"{ticker}_full_1min_adjsplitdiv.txt"
        )
    read_options = csv.ReadOptions(column_names=names, skip_rows=1)
    parse_options = csv.ParseOptions(delimiter=",")
    ds = ray.data.read_csv(
        file_path, parse_options=parse_options, read_options=read_options
    )
    ds = ds.sort("Time")
    return ds


def get_tick_data(
    ticker: str, asset_type: str, start_date, end_date, raw_dir
) -> pd.DataFrame:
    # ticker = ticker.replace("CME_","")
    names = ["Time", "Open", "High", "Low", "Close", "Volume"]
    if asset_type in ["FUT"]:
        file_path = os.path.join(
            f"{raw_dir}/futures", f"{ticker}_1min_continuous_adjusted.txt"
        )
    else:
        file_path = os.path.join(
            f"{raw_dir}/stock", f"{ticker}_full_1min_adjsplitdiv.txt"
        )
    ds = pd.read_csv(file_path, delimiter=",", header=0, names=names)
    ds["Time"] = ds.Time.apply(
        lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    )
    ds.set_index("Time")
    # logging.info(f"ds:{ds.head()}")
    # logging.info(f"ds:{ds.info()}")
    return ds


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

def get_close_time(x, close_col):
    #logging.error(f"get_close_time_x:{x}, x_shape:{x.shape}")
    ticker = x["ticker"]
    timestamp = x["timestamp"]
    close_timestamp = None
    if x[close_col] == x["close"]:
        close_timestamp = int(x["timestamp"])
    else:
        close_timestamp = None
    series = pd.Series([timestamp,ticker,close_timestamp],index=['timestamp','ticker','close_timestamp'])
    return series

def get_time(x, close_col):
    if x[close_col] == x["close_back_cumsum"]:
        return int(x["timestamp"])
    else:
        return None

@profile_util.profile
#@njit(parallel=True)
def ticker_transform(raw_data, config, base_price=500):
    interval_minutes = config.dataset.interval_mins
    add_daily_rolling_features = config.model.features.add_daily_rolling_features
    ret_std = config.dataset.ret_std
    ewm = raw_data["close"].ewm(halflife=HALFLIFE_WINSORISE)
    means = ewm.mean()
    stds = ewm.std()
    del ewm
    raw_data["close"] = np.minimum(raw_data["close"], means + VOL_THRESHOLD * stds)
    raw_data["close"] = np.maximum(raw_data["close"], means - VOL_THRESHOLD * stds)
    raw_data["cum_volume"] = raw_data.volume.cumsum()
    raw_data["cum_dv"] = raw_data.dv.cumsum()
    squash_factor = 6
    
    #raw_data['close_back'] = squash_factor * np.tanh((np.log(raw_data.close+base_price) - np.log(raw_data.close.shift(1)+base_price))/squash_factor)
    raw_data['close_back'] = np.log(raw_data.close+base_price) - np.log(raw_data.close.shift(1)+base_price)
    raw_data['high_back'] = np.log(raw_data.high+base_price) - np.log(raw_data.high.shift(1)+base_price)
    raw_data['open_back'] = np.log(raw_data.open+base_price) - np.log(raw_data.open.shift(1)+base_price)
    raw_data['low_back'] = np.log(raw_data.low+base_price) - np.log(raw_data.low.shift(1)+base_price)

    raw_data['close_velocity_back'] = raw_data.close_back - raw_data.close_back.shift(1)
    #logging.info(f"raw_data:{raw_data.describe()}")
    raw_data['close_back'] = np.minimum(raw_data["close_back"], VOL_THRESHOLD * ret_std)
    raw_data['close_back'] = np.maximum(raw_data["close_back"], -VOL_THRESHOLD * ret_std)
    raw_data['high_back'] = np.minimum(raw_data["high_back"], VOL_THRESHOLD * ret_std)
    raw_data['high_back'] = np.maximum(raw_data["high_back"], -VOL_THRESHOLD * ret_std)
    raw_data['open_back'] = np.minimum(raw_data["open_back"], VOL_THRESHOLD * ret_std)
    raw_data['open_back'] = np.maximum(raw_data["open_back"], -VOL_THRESHOLD * ret_std)
    raw_data['low_back'] = np.minimum(raw_data["low_back"], VOL_THRESHOLD * ret_std)
    raw_data['low_back'] = np.maximum(raw_data["low_back"], -VOL_THRESHOLD * ret_std)
    #logging.info(f"raw_data after min/max:{raw_data.describe()}")
    
    #raw_data['close_back'] = squash_factor * np.tanh(raw_data['close_back']/squash_factor)
    #raw_data['high_back'] = squash_factor * np.tanh(raw_data['high_back']/squash_factor)
    #raw_data['open_back'] = squash_factor * np.tanh(raw_data['open_back']/squash_factor)
    #raw_data['low_back'] = squash_factor * np.tanh(raw_data['low_back']/squash_factor)

    # Avoid inf
    raw_data['volume_back'] = np.log(raw_data.volume+2) - np.log(raw_data.volume.shift(1)+2)
    raw_data['dv_back'] = np.log(raw_data.dv) - np.log(raw_data.dv.shift(1))
    
    raw_data["close_back_cumsum"] = raw_data["close_back"].cumsum()
    raw_data["volume_back_cumsum"] = raw_data["volume_back"].cumsum()

    close_back_cumsum = raw_data["close_back_cumsum"]
    timestamp = raw_data["timestamp"]

    if add_daily_rolling_features:
        interval_per_day = int(23 * 60 / interval_minutes)
        # find_peaks only with prominance which needs to be set to half of the width.
        # in case of high among 5 days, the high needs to be higher than 4 points around
        # it, 2 to the left and 2 to the right.
        raw_data['close_back_cumsum_rolling_1d_max'] = raw_data.close_back_cumsum.rolling(interval_per_day).max()
        raw_data["close_back_cumsum_high_1d_ff"] = raw_data["close_back_cumsum_rolling_1d_max"].ffill()
        raw_data['close_rolling_1d_max'] = raw_data.close.rolling(interval_per_day).max()
        raw_data["close_high_1d_ff"] = raw_data["close_rolling_1d_max"].ffill()
        raw_data["time_high_1d_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_1d_ff"), axis=1)
        raw_data["time_high_1d_ff"]  = raw_data["time_high_1d_ff"].ffill()
        raw_data["close_high_1d_ff_shift_1d"] = raw_data["close_high_1d_ff"].shift(-1*interval_per_day)        
        raw_data["time_high_1d_ff_shift_1d"] = raw_data["time_high_1d_ff"].shift(-1*interval_per_day)

        raw_data['close_back_cumsum_rolling_1d_min'] = raw_data.close_back_cumsum.rolling(interval_per_day).min()
        raw_data["close_back_cumsum_low_1d_ff"] = raw_data["close_back_cumsum_rolling_1d_min"].ffill()
        raw_data['close_rolling_1d_min'] = raw_data.close.rolling(interval_per_day).min()
        raw_data["close_low_1d_ff"] = raw_data["close_rolling_1d_min"].ffill()
        raw_data["time_low_1d_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_1d_ff"), axis=1)
        raw_data["time_low_1d_ff"]  = raw_data["time_high_1d_ff"].ffill()
        raw_data["close_low_1d_ff_shift_1d"] = raw_data["close_low_1d_ff"].shift(-1*interval_per_day)        
        raw_data["time_low_1d_ff_shift_1d"] = raw_data["time_low_1d_ff"].shift(-1*interval_per_day)

        raw_data['close_back_cumsum_rolling_5d_max'] = raw_data.close_back_cumsum.rolling(5*interval_per_day).max()
        raw_data["close_back_cumsum_high_5d_ff"] = raw_data["close_back_cumsum_rolling_5d_max"].ffill()
        raw_data['close_rolling_5d_max'] = raw_data.close.rolling(5*interval_per_day).max()
        raw_data["close_high_5d_ff"] = raw_data["close_rolling_5d_max"].ffill()
        raw_data["time_high_5d_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_5d_ff"), axis=1)
        raw_data["time_high_5d_ff"]  = raw_data["time_high_5d_ff"].ffill()
        raw_data["time_close_back_cumsum_high_5d_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_high_5d_ff"), axis=1)
        raw_data["time_close_back_cumsum_high_5d_ff"]  = raw_data["time_close_back_cumsum_high_5d_ff"].ffill()
        raw_data["close_high_5d_bf"] = raw_data["close_rolling_5d_max"].bfill()
        raw_data["time_high_5d_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_5d_bf"), axis=1)
        raw_data["time_high_5d_bf"]  = raw_data["time_high_5d_bf"].bfill()
        raw_data["close_high_5d_ff_shift_5d"] = raw_data["close_high_5d_ff"].shift(-5*interval_per_day)        
        raw_data["time_high_5d_ff_shift_5d"] = raw_data["time_high_5d_ff"].shift(-5*interval_per_day)

        raw_data['close_back_cumsum_rolling_5d_min'] = raw_data.close_back_cumsum.rolling(5*interval_per_day).min()
        raw_data["close_back_cumsum_low_5d_ff"] = raw_data["close_back_cumsum_rolling_5d_min"].ffill()
        raw_data['close_rolling_5d_min'] = raw_data.close.rolling(5*interval_per_day).min()
        raw_data["close_low_5d_ff"] = raw_data["close_rolling_5d_min"].ffill()
        raw_data["time_low_5d_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_5d_ff"), axis=1)
        raw_data["time_low_5d_ff"]  = raw_data["time_low_5d_ff"].ffill()
        raw_data["time_close_back_cumsum_low_5d_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_low_5d_ff"), axis=1)
        raw_data["time_close_back_cumsum_low_5d_ff"]  = raw_data["time_close_back_cumsum_low_5d_ff"].ffill()
        raw_data["close_low_5d_bf"] = raw_data["close_rolling_5d_min"].bfill()
        raw_data["time_low_5d_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_5d_bf"), axis=1)
        raw_data["time_low_5d_bf"]  = raw_data["time_low_5d_bf"].bfill()
        raw_data["close_low_5d_ff_shift_5d"] = raw_data["close_low_5d_ff"].shift(-5*interval_per_day)        
        raw_data["time_low_5d_ff_shift_5d"] = raw_data["time_low_5d_ff"].shift(-5*interval_per_day)

        raw_data['close_back_cumsum_rolling_11d_max'] = raw_data.close_back_cumsum.rolling(11*interval_per_day).max()
        raw_data["close_back_cumsum_high_11d_ff"] = raw_data["close_back_cumsum_rolling_11d_max"].ffill()
        raw_data['close_rolling_11d_max'] = raw_data.close.rolling(11*interval_per_day).max()
        raw_data["close_high_11d_ff"] = raw_data["close_rolling_11d_max"].ffill()
        raw_data["time_high_11d_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_11d_ff"), axis=1)
        raw_data["time_high_11d_ff"]  = raw_data["time_high_11d_ff"].ffill()
        raw_data["time_close_back_cumsum_high_11d_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_high_11d_ff"), axis=1)
        raw_data["time_close_back_cumsum_high_11d_ff"]  = raw_data["time_close_back_cumsum_high_11d_ff"].ffill()
        raw_data["close_high_11d_bf"] = raw_data["close_rolling_11d_max"].bfill()
        raw_data["time_high_11d_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_11d_bf"), axis=1)
        raw_data["time_high_11d_bf"]  = raw_data["time_high_11d_bf"].bfill()
        raw_data["close_high_11d_ff_shift_11d"] = raw_data["close_high_11d_ff"].shift(-11*interval_per_day)        
        raw_data["time_high_11d_ff_shift_11d"] = raw_data["time_high_11d_ff"].shift(-11*interval_per_day)

        raw_data['close_back_cumsum_rolling_11d_min'] = raw_data.close_back_cumsum.rolling(11*interval_per_day).min()
        raw_data["close_back_cumsum_low_11d_ff"] = raw_data["close_back_cumsum_rolling_11d_min"].ffill()
        raw_data['close_rolling_11d_min'] = raw_data.close.rolling(11*interval_per_day).min()
        raw_data["close_low_11d_ff"] = raw_data["close_rolling_11d_min"].ffill()
        raw_data["time_low_11d_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_11d_ff"), axis=1)
        raw_data["time_low_11d_ff"]  = raw_data["time_low_11d_ff"].ffill()
        raw_data["time_close_back_cumsum_low_11d_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_low_11d_ff"), axis=1)
        raw_data["time_close_back_cumsum_low_11d_ff"]  = raw_data["time_close_back_cumsum_low_11d_ff"].ffill()
        raw_data["close_low_11d_bf"] = raw_data["close_rolling_11d_min"].bfill()
        raw_data["time_low_11d_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_11d_bf"), axis=1)
        raw_data["time_low_11d_bf"]  = raw_data["time_low_11d_bf"].bfill()
        raw_data["close_low_11d_ff_shift_11d"] = raw_data["close_low_11d_ff"].shift(-11*interval_per_day)        
        raw_data["time_low_11d_ff_shift_11d"] = raw_data["time_low_11d_ff"].shift(-11*interval_per_day)

        raw_data['close_back_cumsum_rolling_21d_max'] = raw_data.close_back_cumsum.rolling(21*interval_per_day).max()
        raw_data["close_back_cumsum_high_21d_ff"] = raw_data["close_back_cumsum_rolling_21d_max"].ffill()
        raw_data['close_rolling_21d_max'] = raw_data.close.rolling(21*interval_per_day).max()
        raw_data["close_high_21d_ff"] = raw_data["close_rolling_21d_max"].ffill()
        raw_data["time_high_21d_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_21d_ff"), axis=1)
        raw_data["time_high_21d_ff"]  = raw_data["time_high_21d_ff"].ffill()
        raw_data["time_close_back_cumsum_high_21d_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_high_21d_ff"), axis=1)
        raw_data["time_close_back_cumsum_high_21d_ff"]  = raw_data["time_close_back_cumsum_high_21d_ff"].ffill()
        raw_data["close_high_21d_bf"] = raw_data["close_rolling_21d_max"].bfill()
        raw_data["time_high_21d_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_21d_bf"), axis=1)
        raw_data["time_high_21d_bf"]  = raw_data["time_high_21d_bf"].bfill()
        raw_data["close_high_21d_ff_shift_21d"] = raw_data["close_high_21d_ff"].shift(-21)        
        raw_data["time_high_21d_ff_shift_21d"] = raw_data["time_high_21d_ff"].shift(-21)

        raw_data['close_back_cumsum_rolling_21d_min'] = raw_data.close_back_cumsum.rolling(21*interval_per_day).min()
        raw_data["close_back_cumsum_low_21d_ff"] = raw_data["close_back_cumsum_rolling_21d_min"].ffill()
        raw_data['close_rolling_21d_min'] = raw_data.close.rolling(21*interval_per_day).min()
        raw_data["close_low_21d_ff"] = raw_data["close_rolling_21d_min"].ffill()
        raw_data["time_low_21d_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_21d_ff"), axis=1)
        raw_data["time_low_21d_ff"]  = raw_data["time_low_21d_ff"].ffill()
        raw_data["time_close_back_cumsum_low_21d_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_low_21d_ff"), axis=1)
        raw_data["time_close_back_cumsum_low_21d_ff"]  = raw_data["time_close_back_cumsum_low_21d_ff"].ffill()
        raw_data["close_low_21d_bf"] = raw_data["close_rolling_21d_min"].bfill()
        raw_data["time_low_21d_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_21d_bf"), axis=1)
        raw_data["time_low_21d_bf"]  = raw_data["time_low_21d_bf"].bfill()
        raw_data["close_low_21d_ff_shift_21d"] = raw_data["close_low_21d_ff"].shift(-21)        
        raw_data["time_low_21d_ff_shift_21d"] = raw_data["time_low_21d_ff"].shift(-21)

        raw_data['close_back_cumsum_rolling_51d_max'] = raw_data.close_back_cumsum.rolling(51*interval_per_day).max()
        raw_data["close_back_cumsum_high_51d_ff"] = raw_data["close_back_cumsum_rolling_51d_max"].ffill()
        raw_data['close_rolling_51d_max'] = raw_data.close.rolling(51*interval_per_day).max()
        raw_data["close_high_51d_ff"] = raw_data["close_rolling_51d_max"].ffill()
        raw_data["time_high_51d_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_51d_ff"), axis=1)
        raw_data["time_high_51d_ff"]  = raw_data["time_high_51d_ff"].ffill()
        raw_data["time_close_back_cumsum_high_51d_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_high_51d_ff"), axis=1)
        raw_data["time_close_back_cumsum_high_51d_ff"]  = raw_data["time_close_back_cumsum_high_51d_ff"].ffill()
        raw_data["close_high_51d_bf"] = raw_data["close_rolling_51d_max"].bfill()
        raw_data["time_high_51d_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_51d_bf"), axis=1)
        raw_data["time_high_51d_bf"]  = raw_data["time_high_51d_bf"].bfill()

        raw_data['close_back_cumsum_rolling_51d_min'] = raw_data.close_back_cumsum.rolling(51*interval_per_day).min()
        raw_data["close_back_cumsum_low_51d_ff"] = raw_data["close_back_cumsum_rolling_51d_min"].ffill()
        raw_data['close_rolling_51d_min'] = raw_data.close.rolling(51*interval_per_day).min()
        raw_data["close_low_51d_ff"] = raw_data["close_rolling_51d_min"].ffill()
        raw_data["time_low_51d_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_51d_ff"), axis=1)
        raw_data["time_low_51d_ff"]  = raw_data["time_low_51d_ff"].ffill()
        raw_data["time_close_back_cumsum_low_51d_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_low_51d_ff"), axis=1)
        raw_data["time_close_back_cumsum_low_51d_ff"]  = raw_data["time_close_back_cumsum_low_51d_ff"].ffill()
        raw_data["close_low_51d_bf"] = raw_data["close_rolling_51d_min"].bfill()
        raw_data["time_low_51d_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_51d_bf"), axis=1)
        raw_data["time_low_51d_bf"]  = raw_data["time_low_51d_bf"].bfill()

        raw_data['close_back_cumsum_rolling_101d_max'] = raw_data.close_back_cumsum.rolling(101*interval_per_day).max()
        raw_data["close_back_cumsum_high_101d_ff"] = raw_data["close_back_cumsum_rolling_101d_max"].ffill()
        raw_data['close_rolling_101d_max'] = raw_data.close.rolling(101*interval_per_day).max()
        raw_data["close_high_101d_ff"] = raw_data["close_rolling_101d_max"].ffill()
        raw_data["time_high_101d_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_101d_ff"), axis=1)
        raw_data["time_high_101d_ff"]  = raw_data["time_high_101d_ff"].ffill()
        raw_data["time_close_back_cumsum_high_101d_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_high_101d_ff"), axis=1)
        raw_data["time_close_back_cumsum_high_101d_ff"]  = raw_data["time_close_back_cumsum_high_101d_ff"].ffill()
        raw_data["close_high_101d_bf"] = raw_data["close_rolling_101d_max"].bfill()
        raw_data["time_high_101d_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_101d_bf"), axis=1)
        raw_data["time_high_101d_bf"]  = raw_data["time_high_101d_bf"].bfill()

        raw_data['close_back_cumsum_rolling_101d_min'] = raw_data.close_back_cumsum.rolling(101*interval_per_day).min()
        raw_data["close_back_cumsum_low_101d_ff"] = raw_data["close_back_cumsum_rolling_101d_min"].ffill()
        raw_data['close_rolling_101d_min'] = raw_data.close.rolling(101*interval_per_day).min()
        raw_data["close_low_101d_ff"] = raw_data["close_rolling_101d_min"].ffill()
        raw_data["time_low_101d_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_101d_ff"), axis=1)
        raw_data["time_low_101d_ff"]  = raw_data["time_low_101d_ff"].ffill()
        raw_data["time_close_back_cumsum_low_101d_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_low_101d_ff"), axis=1)
        raw_data["time_close_back_cumsum_low_101d_ff"]  = raw_data["time_close_back_cumsum_low_101d_ff"].ffill()
        raw_data["close_low_101d_bf"] = raw_data["close_rolling_101d_min"].bfill()
        raw_data["time_low_101d_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_101d_bf"), axis=1)
        raw_data["time_low_101d_bf"]  = raw_data["time_low_101d_bf"].bfill()

        raw_data['close_back_cumsum_rolling_201d_max'] = raw_data.close_back_cumsum.rolling(201*interval_per_day).max()
        raw_data["close_back_cumsum_high_201d_ff"] = raw_data["close_back_cumsum_rolling_201d_max"].ffill()
        raw_data['close_rolling_201d_max'] = raw_data.close.rolling(201*interval_per_day).max()
        raw_data["close_high_201d_ff"] = raw_data["close_rolling_201d_max"].ffill()
        raw_data["time_high_201d_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_201d_ff"), axis=1)
        raw_data["time_high_201d_ff"]  = raw_data["time_high_201d_ff"].ffill()
        raw_data["time_close_back_cumsum_high_201d_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_high_201d_ff"), axis=1)
        raw_data["time_close_back_cumsum_high_201d_ff"]  = raw_data["time_close_back_cumsum_high_201d_ff"].ffill()
        raw_data["close_high_201d_bf"] = raw_data["close_rolling_201d_max"].bfill()
        raw_data["time_high_201d_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_201d_bf"), axis=1)
        raw_data["time_high_201d_bf"]  = raw_data["time_high_201d_bf"].bfill()

        raw_data['close_back_cumsum_rolling_201d_min'] = raw_data.close_back_cumsum.rolling(201*interval_per_day).min()
        raw_data["close_back_cumsum_low_201d_ff"] = raw_data["close_back_cumsum_rolling_201d_min"].ffill()
        raw_data['close_rolling_201d_min'] = raw_data.close.rolling(201*interval_per_day).min()
        raw_data["close_low_201d_ff"] = raw_data["close_rolling_201d_min"].ffill()
        raw_data["time_low_201d_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_201d_ff"), axis=1)
        raw_data["time_low_201d_ff"]  = raw_data["time_low_201d_ff"].ffill()
        raw_data["time_close_back_cumsum_low_201d_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_low_201d_ff"), axis=1)
        raw_data["time_close_back_cumsum_low_201d_ff"]  = raw_data["time_close_back_cumsum_low_201d_ff"].ffill()
        raw_data["close_low_201d_bf"] = raw_data["close_rolling_201d_min"].bfill()
        raw_data["time_low_201d_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_201d_bf"), axis=1)
        raw_data["time_low_201d_bf"]  = raw_data["time_low_201d_bf"].bfill()


    raw_data['close_back_cumsum_rolling_5_max'] = raw_data.close_back_cumsum.rolling(5).max()
    raw_data["close_back_cumsum_high_5_ff"] = raw_data["close_back_cumsum_rolling_5_max"].ffill()
    raw_data['close_rolling_5_max'] = raw_data.close.rolling(5).max()
    raw_data["close_high_5_ff"] = raw_data["close_rolling_5_max"].ffill()
    raw_data["time_high_5_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_5_ff"), axis=1)
    raw_data["time_high_5_ff"]  = raw_data["time_high_5_ff"].ffill()
    raw_data["time_close_back_cumsum_high_5_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_high_5_ff"), axis=1)
    raw_data["time_close_back_cumsum_high_5_ff"]  = raw_data["time_close_back_cumsum_high_5_ff"].ffill()
    raw_data["close_high_5_bf"] = raw_data["close_rolling_5_max"].bfill()
    raw_data["time_high_5_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_5_bf"), axis=1)
    raw_data["time_high_5_bf"]  = raw_data["time_high_5_bf"].bfill()

    raw_data["close_back_cumsum_rolling_5_min"] = raw_data.close_back_cumsum.rolling(5).min()
    raw_data["close_back_cumsum_low_5_ff"] = raw_data["close_back_cumsum_rolling_5_min"].ffill()
    raw_data["close_rolling_5_min"] = raw_data.close.rolling(5).min()
    raw_data["close_low_5_ff"] = raw_data["close_rolling_5_min"].ffill()
    raw_data["time_low_5_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_5_ff"), axis=1)
    raw_data["time_low_5_ff"]  = raw_data["time_low_5_ff"].ffill()
    raw_data["time_close_back_cumsum_low_5_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_low_5_ff"), axis=1)
    raw_data["time_close_back_cumsum_low_5_ff"]  = raw_data["time_close_back_cumsum_low_5_ff"].ffill()
    raw_data["close_low_5_bf"] = raw_data["close_rolling_5_min"].bfill()
    raw_data["time_low_5_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_5_bf"), axis=1)
    raw_data["time_low_5_bf"]  = raw_data["time_low_5_bf"].bfill()

    raw_data['close_back_cumsum_rolling_11_max'] = raw_data.close_back_cumsum.rolling(11).max()
    raw_data["close_back_cumsum_high_11_ff"] = raw_data["close_back_cumsum_rolling_11_max"].ffill()
    raw_data['close_rolling_11_max'] = raw_data.close.rolling(11).max()
    raw_data["close_high_11_ff"] = raw_data["close_rolling_11_max"].ffill()
    raw_data["time_high_11_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_11_ff"), axis=1)
    raw_data["time_high_11_ff"]  = raw_data["time_high_11_ff"].ffill()
    raw_data["time_close_back_cumsum_high_11_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_high_11_ff"), axis=1)
    raw_data["time_close_back_cumsum_high_11_ff"]  = raw_data["time_close_back_cumsum_high_11_ff"].ffill()
    raw_data["close_high_11_bf"] = raw_data["close_rolling_11_max"].bfill()
    raw_data["time_high_11_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_11_bf"), axis=1)
    raw_data["time_high_11_bf"]  = raw_data["time_high_11_bf"].bfill()

    raw_data["close_back_cumsum_rolling_11_min"] = raw_data.close_back_cumsum.rolling(11).min()
    raw_data["close_back_cumsum_low_11_ff"] = raw_data["close_back_cumsum_rolling_11_min"].ffill()
    raw_data["close_rolling_11_min"] = raw_data.close.rolling(11).min()
    raw_data["close_low_11_ff"] = raw_data["close_rolling_11_min"].ffill()
    raw_data["time_low_11_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_11_ff"), axis=1)
    raw_data["time_low_11_ff"]  = raw_data["time_low_11_ff"].ffill()
    raw_data["time_close_back_cumsum_low_11_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_low_11_ff"), axis=1)
    raw_data["time_close_back_cumsum_low_11_ff"]  = raw_data["time_close_back_cumsum_low_11_ff"].ffill()
    raw_data["close_low_11_bf"] = raw_data["close_rolling_11_min"].bfill()
    raw_data["time_low_11_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_11_bf"), axis=1)
    raw_data["time_low_11_bf"]  = raw_data["time_low_11_bf"].bfill()

    raw_data['close_back_cumsum_rolling_21_max'] = raw_data.close_back_cumsum.rolling(21).max()
    raw_data["close_back_cumsum_high_21_ff"] = raw_data["close_back_cumsum_rolling_21_max"].ffill()
    raw_data['close_rolling_21_max'] = raw_data.close.rolling(21).max()
    raw_data["close_high_21_ff"] = raw_data["close_rolling_21_max"].ffill()
    raw_data["time_high_21_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_21_ff"), axis=1)
    raw_data["time_high_21_ff"]  = raw_data["time_high_21_ff"].ffill()
    raw_data["time_close_back_cumsum_high_21_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_high_21_ff"), axis=1)
    raw_data["time_close_back_cumsum_high_21_ff"]  = raw_data["time_close_back_cumsum_high_21_ff"].ffill()
    raw_data["close_high_21_bf"] = raw_data["close_rolling_21_max"].bfill()
    raw_data["time_high_21_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_21_bf"), axis=1)
    raw_data["time_high_21_bf"]  = raw_data["time_high_21_bf"].bfill()

    raw_data["close_back_cumsum_rolling_21_min"] = raw_data.close_back_cumsum.rolling(21).min()
    raw_data["close_back_cumsum_low_21_ff"] = raw_data["close_back_cumsum_rolling_21_min"].ffill()
    raw_data["close_rolling_21_min"] = raw_data.close.rolling(21).min()
    raw_data["close_low_21_ff"] = raw_data["close_rolling_21_min"].ffill()
    raw_data["time_low_21_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_21_ff"), axis=1)
    raw_data["time_low_21_ff"]  = raw_data["time_low_21_ff"].ffill()
    raw_data["time_close_back_cumsum_low_21_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_low_21_ff"), axis=1)
    raw_data["time_close_back_cumsum_low_21_ff"]  = raw_data["time_close_back_cumsum_low_21_ff"].ffill()
    raw_data["close_low_21_bf"] = raw_data["close_rolling_21_min"].bfill()
    raw_data["time_low_21_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_21_bf"), axis=1)
    raw_data["time_low_21_bf"]  = raw_data["time_low_21_bf"].bfill()

    raw_data['close_back_cumsum_rolling_51_max'] = raw_data.close_back_cumsum.rolling(51).max()
    raw_data["close_back_cumsum_high_51_ff"] = raw_data["close_back_cumsum_rolling_51_max"].ffill()
    raw_data['close_rolling_51_max'] = raw_data.close.rolling(51).max()
    raw_data["close_high_51_ff"] = raw_data["close_rolling_51_max"].ffill()
    raw_data["time_high_51_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_51_ff"), axis=1)
    raw_data["time_high_51_ff"]  = raw_data["time_high_51_ff"].ffill()
    raw_data["time_close_back_cumsum_high_51_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_high_51_ff"), axis=1)
    raw_data["time_close_back_cumsum_high_51_ff"]  = raw_data["time_close_back_cumsum_high_51_ff"].ffill()
    raw_data["close_high_51_bf"] = raw_data["close_rolling_51_max"].bfill()
    raw_data["time_high_51_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_51_bf"), axis=1)
    raw_data["time_high_51_bf"]  = raw_data["time_high_51_bf"].bfill()

    raw_data["close_back_cumsum_rolling_51_min"] = raw_data.close_back_cumsum.rolling(51).min()
    raw_data["close_back_cumsum_low_51_ff"] = raw_data["close_back_cumsum_rolling_51_min"].ffill()
    raw_data["close_rolling_51_min"] = raw_data.close.rolling(51).min()
    raw_data["close_low_51_ff"] = raw_data["close_rolling_51_min"].ffill()
    raw_data["time_low_51_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_51_ff"), axis=1)
    raw_data["time_low_51_ff"]  = raw_data["time_low_51_ff"].ffill()
    raw_data["time_close_back_cumsum_low_51_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_low_51_ff"), axis=1)
    raw_data["time_close_back_cumsum_low_51_ff"]  = raw_data["time_close_back_cumsum_low_51_ff"].ffill()
    raw_data["close_low_51_bf"] = raw_data["close_rolling_51_min"].bfill()
    raw_data["time_low_51_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_51_bf"), axis=1)
    raw_data["time_low_51_bf"]  = raw_data["time_low_51_bf"].bfill()

    raw_data['close_back_cumsum_rolling_101_max'] = raw_data.close_back_cumsum.rolling(101).max()
    raw_data["close_back_cumsum_high_101_ff"] = raw_data["close_back_cumsum_rolling_101_max"].ffill()
    raw_data['close_rolling_101_max'] = raw_data.close.rolling(101).max()
    raw_data["close_high_101_ff"] = raw_data["close_rolling_101_max"].ffill()
    raw_data["time_high_101_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_101_ff"), axis=1)
    raw_data["time_high_101_ff"]  = raw_data["time_high_101_ff"].ffill()
    raw_data["time_close_back_cumsum_high_101_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_high_101_ff"), axis=1)
    raw_data["time_close_back_cumsum_high_101_ff"]  = raw_data["time_close_back_cumsum_high_101_ff"].ffill()
    raw_data["close_high_101_bf"] = raw_data["close_rolling_101_max"].bfill()
    raw_data["time_high_101_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_101_bf"), axis=1)
    raw_data["time_high_101_bf"]  = raw_data["time_high_101_bf"].bfill()

    raw_data["close_back_cumsum_rolling_101_min"] = raw_data.close_back_cumsum.rolling(101).min()
    raw_data["close_back_cumsum_low_101_ff"] = raw_data["close_back_cumsum_rolling_101_min"].ffill()
    raw_data["close_rolling_101_min"] = raw_data.close.rolling(101).min()
    raw_data["close_low_101_ff"] = raw_data["close_rolling_101_min"].ffill()
    raw_data["time_low_101_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_101_ff"), axis=1)
    raw_data["time_low_101_ff"]  = raw_data["time_low_101_ff"].ffill()
    raw_data["time_close_back_cumsum_low_101_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_low_101_ff"), axis=1)
    raw_data["time_close_back_cumsum_low_101_ff"]  = raw_data["time_close_back_cumsum_low_101_ff"].ffill()
    raw_data["close_low_101_bf"] = raw_data["close_rolling_101_min"].bfill()
    raw_data["time_low_101_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_101_bf"), axis=1)
    raw_data["time_low_101_bf"]  = raw_data["time_low_101_bf"].bfill()

    raw_data['close_back_cumsum_rolling_201_max'] = raw_data.close_back_cumsum.rolling(201).max()
    raw_data["close_back_cumsum_high_201_ff"] = raw_data["close_back_cumsum_rolling_201_max"].ffill()
    raw_data['close_rolling_201_max'] = raw_data.close.rolling(201).max()
    raw_data["close_high_201_ff"] = raw_data["close_rolling_201_max"].ffill()
    raw_data["time_high_201_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_201_ff"), axis=1)
    raw_data["time_high_201_ff"]  = raw_data["time_high_201_ff"].ffill()
    raw_data["time_close_back_cumsum_high_201_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_high_201_ff"), axis=1)
    raw_data["time_close_back_cumsum_high_201_ff"]  = raw_data["time_close_back_cumsum_high_201_ff"].ffill()
    raw_data["close_high_201_bf"] = raw_data["close_rolling_201_max"].bfill()
    raw_data["time_high_201_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_high_201_bf"), axis=1)
    raw_data["time_high_201_bf"]  = raw_data["time_high_201_bf"].bfill()

    raw_data["close_back_cumsum_rolling_201_min"] = raw_data.close_back_cumsum.rolling(201).min()
    raw_data["close_back_cumsum_low_201_ff"] = raw_data["close_back_cumsum_rolling_201_min"].ffill()
    raw_data["close_rolling_201_min"] = raw_data.close.rolling(201).min()
    raw_data["close_low_201_ff"] = raw_data["close_rolling_201_min"].ffill()
    raw_data["time_low_201_ff"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_201_ff"), axis=1)
    raw_data["time_low_201_ff"]  = raw_data["time_low_201_ff"].ffill()
    raw_data["time_close_back_cumsum_low_201_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_back_cumsum_low_201_ff"), axis=1)
    raw_data["time_close_back_cumsum_low_201_ff"]  = raw_data["time_close_back_cumsum_low_201_ff"].ffill()
    raw_data["close_low_201_bf"] = raw_data["close_rolling_201_min"].bfill()
    raw_data["time_low_201_bf"] = raw_data.apply(lambda x: get_close_time(x, close_col="close_low_201_bf"), axis=1)
    raw_data["time_low_201_bf"]  = raw_data["time_low_201_bf"].bfill()

    del close_back_cumsum
    
    # Compute RSI
    raw_data["rsi_14"] = ta.momentum.RSIIndicator(close=raw_data["close"], window=14).rsi()
    raw_data["rsi_28"] = ta.momentum.RSIIndicator(close=raw_data["close"], window=28).rsi()
    raw_data["rsi_42"] = ta.momentum.RSIIndicator(close=raw_data["close"], window=42).rsi()

    # Compute MACD
    macd = ta.trend.MACD(close=raw_data["close"])
    raw_data["macd"] = macd.macd()
    raw_data["macd_signal"] = macd.macd_signal()

    # Compute Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=5, window_dev=2)
    raw_data["bb_high_5_2"] = bollinger.bollinger_hband()
    raw_data["bb_low_5_2"] = bollinger.bollinger_lband()
    bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=5, window_dev=3)
    raw_data["bb_high_5_3"] = bollinger.bollinger_hband()
    raw_data["bb_low_5_3"] = bollinger.bollinger_lband()

    bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=10, window_dev=2)
    raw_data["bb_high_10_2"] = bollinger.bollinger_hband()
    raw_data["bb_low_10_2"] = bollinger.bollinger_lband()
    bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=10, window_dev=3)
    raw_data["bb_high_10_3"] = bollinger.bollinger_hband()
    raw_data["bb_low_10_3"] = bollinger.bollinger_lband()

    bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=20, window_dev=2)
    raw_data["bb_high_20_2"] = bollinger.bollinger_hband()
    raw_data["bb_low_20_2"] = bollinger.bollinger_lband()

    bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=20, window_dev=3)
    raw_data["bb_high_20_3"] = bollinger.bollinger_hband()
    raw_data["bb_low_20_3"] = bollinger.bollinger_lband()

    bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=50, window_dev=2)
    raw_data["bb_high_50_2"] = bollinger.bollinger_hband()
    raw_data["bb_low_50_2"] = bollinger.bollinger_lband()

    bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=50, window_dev=3)
    raw_data["bb_high_50_3"] = bollinger.bollinger_hband()
    raw_data["bb_low_50_3"] = bollinger.bollinger_lband()

    bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=100, window_dev=2)
    raw_data["bb_high_100_2"] = bollinger.bollinger_hband()
    raw_data["bb_low_100_2"] = bollinger.bollinger_lband()
    bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=100, window_dev=3)
    raw_data["bb_high_100_3"] = bollinger.bollinger_hband()
    raw_data["bb_low_100_3"] = bollinger.bollinger_lband()

    bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=200, window_dev=2)
    raw_data["bb_high_200_2"] = bollinger.bollinger_hband()
    raw_data["bb_low_200_2"] = bollinger.bollinger_lband()
    bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=200, window_dev=3)
    raw_data["bb_high_200_3"] = bollinger.bollinger_hband()
    raw_data["bb_low_200_3"] = bollinger.bollinger_lband()

    # Compute Moving Averages
    if add_daily_rolling_features:
        interval_per_day = int(23 * 60 / interval_minutes)
        raw_data["rsi_14d"] = ta.momentum.RSIIndicator(close=raw_data["close"], window=14*interval_per_day).rsi()
        raw_data["rsi_28d"] = ta.momentum.RSIIndicator(close=raw_data["close"], window=28*interval_per_day).rsi()
        raw_data["rsi_42d"] = ta.momentum.RSIIndicator(close=raw_data["close"], window=42*interval_per_day).rsi()

        raw_data["sma_5d"] = ta.trend.SMAIndicator(
            close=raw_data["close"], window=5*interval_per_day
        ).sma_indicator()
        raw_data["sma_10d"] = ta.trend.SMAIndicator(
            close=raw_data["close"], window=10*interval_per_day
        ).sma_indicator()
        raw_data["sma_20d"] = ta.trend.SMAIndicator(
            close=raw_data["close"], window=20*interval_per_day
        ).sma_indicator()
        raw_data["sma_50d"] = ta.trend.SMAIndicator(
            close=raw_data["close"], window=50*interval_per_day
        ).sma_indicator()
        raw_data["sma_100d"] = ta.trend.SMAIndicator(
            close=raw_data["close"], window=100*interval_per_day
        ).sma_indicator()
        raw_data["sma_200d"] = ta.trend.SMAIndicator(
            close=raw_data["close"], window=200*interval_per_day
        ).sma_indicator()

        bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=5*interval_per_day, window_dev=2)
        raw_data["bb_high_5d_2"] = bollinger.bollinger_hband()
        raw_data["bb_low_5d_2"] = bollinger.bollinger_lband()
        bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=5*interval_per_day, window_dev=3)
        raw_data["bb_high_5d_3"] = bollinger.bollinger_hband()
        raw_data["bb_low_5d_3"] = bollinger.bollinger_lband()

        bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=10*interval_per_day, window_dev=2)
        raw_data["bb_high_10d_2"] = bollinger.bollinger_hband()
        raw_data["bb_low_10d_2"] = bollinger.bollinger_lband()
        bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=10*interval_per_day, window_dev=3)
        raw_data["bb_high_10d_3"] = bollinger.bollinger_hband()
        raw_data["bb_low_10d_3"] = bollinger.bollinger_lband()

        bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=20*interval_per_day, window_dev=2)
        raw_data["bb_high_20d_2"] = bollinger.bollinger_hband()
        raw_data["bb_low_20d_2"] = bollinger.bollinger_lband()

        bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=20*interval_per_day, window_dev=3)
        raw_data["bb_high_20d_3"] = bollinger.bollinger_hband()
        raw_data["bb_low_20d_3"] = bollinger.bollinger_lband()

        bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=50*interval_per_day, window_dev=2)
        raw_data["bb_high_50d_2"] = bollinger.bollinger_hband()
        raw_data["bb_low_50d_2"] = bollinger.bollinger_lband()

        bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=50*interval_per_day, window_dev=3)
        raw_data["bb_high_50d_3"] = bollinger.bollinger_hband()
        raw_data["bb_low_50d_3"] = bollinger.bollinger_lband()

        bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=100*interval_per_day, window_dev=2)
        raw_data["bb_high_100d_2"] = bollinger.bollinger_hband()
        raw_data["bb_low_100d_2"] = bollinger.bollinger_lband()
        bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=100*interval_per_day, window_dev=3)
        raw_data["bb_high_100d_3"] = bollinger.bollinger_hband()
        raw_data["bb_low_100d_3"] = bollinger.bollinger_lband()

        bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=200*interval_per_day, window_dev=2)
        raw_data["bb_high_200d_2"] = bollinger.bollinger_hband()
        raw_data["bb_low_200d_2"] = bollinger.bollinger_lband()
        bollinger = ta.volatility.BollingerBands(close=raw_data["close"], window=200*interval_per_day, window_dev=3)
        raw_data["bb_high_200d_3"] = bollinger.bollinger_hband()
        raw_data["bb_low_200d_3"] = bollinger.bollinger_lband()

    raw_data["sma_5"] = ta.trend.SMAIndicator(
        close=raw_data["close"], window=5
    ).sma_indicator()
    raw_data["sma_10"] = ta.trend.SMAIndicator(
        close=raw_data["close"], window=10
    ).sma_indicator()
    raw_data["sma_20"] = ta.trend.SMAIndicator(
        close=raw_data["close"], window=20
    ).sma_indicator()
    raw_data["sma_50"] = ta.trend.SMAIndicator(
        close=raw_data["close"], window=50
    ).sma_indicator()
    raw_data["sma_100"] = ta.trend.SMAIndicator(
        close=raw_data["close"], window=100
    ).sma_indicator()
    raw_data["sma_200"] = ta.trend.SMAIndicator(
        close=raw_data["close"], window=200
    ).sma_indicator()

    raw_data["daily_returns"] = calc_returns(raw_data["close"], day_offset=1, base_price=base_price)
    raw_data["daily_returns_5"] = calc_returns(raw_data["close"], day_offset=5, base_price=base_price)
    raw_data["daily_returns_10"] = calc_returns(raw_data["close"], day_offset=10, base_price=base_price)
    raw_data["daily_returns_20"] = calc_returns(raw_data["close"], day_offset=20, base_price=base_price)
    raw_data["daily_vol"] = calc_daily_vol(raw_data["daily_returns"])
    raw_data["daily_vol_5"] = calc_daily_vol(raw_data["daily_returns_5"])
    raw_data["daily_vol_10"] = calc_daily_vol(raw_data["daily_returns_10"])
    raw_data["daily_vol_20"] = calc_daily_vol(raw_data["daily_returns_20"])
    raw_data["daily_skew"] = calc_skew(raw_data["daily_returns"])
    raw_data["daily_skew_5"] = calc_skew(raw_data["daily_returns_5"])
    raw_data["daily_skew_10"] = calc_skew(raw_data["daily_returns_10"])
    raw_data["daily_skew_20"] = calc_skew(raw_data["daily_returns_20"])
    raw_data["daily_kurt"] = calc_kurt(raw_data["daily_returns"])
    raw_data["daily_kurt_5"] = calc_kurt(raw_data["daily_returns_5"])
    raw_data["daily_kurt_10"] = calc_kurt(raw_data["daily_returns_10"])
    raw_data["daily_kurt_20"] = calc_kurt(raw_data["daily_returns_20"])

    trend_combinations = [(8, 24), (16, 48), (32, 96)]
    for short_window, long_window in trend_combinations:
        raw_data[f"macd_{short_window}_{long_window}"] = MACDStrategy.calc_signal(
            raw_data["close"], short_window, long_window, interval_per_day=1
        )
    if add_daily_rolling_features:
        interval_per_day = int(23 * 60 / interval_minutes)        
        for short_window, long_window in trend_combinations:
            raw_data[f"macd_{short_window}_{long_window}_day"] = MACDStrategy.calc_signal(
                raw_data["close"], short_window*interval_per_day, long_window*interval_per_day, interval_per_day=46
            )
        raw_data["daily_returns_1d"] = calc_returns(raw_data["close"], day_offset=1*interval_per_day, base_price=base_price)
        raw_data["daily_returns_5d"] = calc_returns(raw_data["close"], day_offset=5*interval_per_day, base_price=base_price)
        raw_data["daily_returns_10d"] = calc_returns(raw_data["close"], day_offset=10*interval_per_day, base_price=base_price)
        raw_data["daily_returns_20d"] = calc_returns(raw_data["close"], day_offset=20*interval_per_day, base_price=base_price)
        raw_data["daily_vol_1d"] = calc_daily_vol(raw_data["daily_returns_1d"])
        raw_data["daily_vol_5d"] = calc_daily_vol(raw_data["daily_returns_5d"])
        raw_data["daily_vol_10d"] = calc_daily_vol(raw_data["daily_returns_10d"])
        raw_data["daily_vol_20d"] = calc_daily_vol(raw_data["daily_returns_20d"])
        raw_data["daily_vol_diff_1_20d"] = raw_data.apply(lambda x: val_diff(x, base_col="daily_vol_20d", diff_col="daily_vol_1d", diff_mul=1), axis=1)
            #std_scaler = StandardScaler()
 
            #raw_data["daily_vol_1d"] = std_scaler.fit_transform(raw_data["daily_vol_1d"].to_numpy())
            #raw_data["daily_vol_5d"] = std_scaler.fit_transform(raw_data["daily_vol_5d"].to_numpy())
            #raw_data["daily_vol_10d"] = std_scaler.fit_transform(raw_data["daily_vol_10d"].to_numpy())
            #raw_data["daily_vol_20d"] = std_scaler.fit_transform(raw_data["daily_vol_20d"].to_numpy())
        raw_data["daily_skew_1d"] = calc_skew(raw_data["daily_returns_1d"])
        raw_data["daily_skew_5d"] = calc_skew(raw_data["daily_returns_5d"])
        raw_data["daily_skew_10d"] = calc_skew(raw_data["daily_returns_10d"])
        raw_data["daily_skew_20d"] = calc_skew(raw_data["daily_returns_20d"])
        raw_data["daily_kurt_1d"] = calc_kurt(raw_data["daily_returns_1d"])
        raw_data["daily_kurt_5d"] = calc_kurt(raw_data["daily_returns_5d"])
        raw_data["daily_kurt_10d"] = calc_kurt(raw_data["daily_returns_10d"])
        raw_data["daily_kurt_20d"] = calc_kurt(raw_data["daily_returns_20d"])

    return raw_data

def time_diff(row, base_col, diff_col):
    if pd.isna(row[diff_col]):
        return np.nan
    else:
        return (row[diff_col] - row[base_col])

def val_diff(row, base_col, diff_col, diff_mul):
    if pd.isna(row[diff_col]):
        return np.nan
    else:
        return (row[diff_col]*diff_mul - row[base_col])

def ret_diff(row, base_col, diff_col):
    if pd.isna(row[base_col]):
        return np.nan
    return (row[diff_col] - row[base_col])

def compute_ret(row, base_col, base_price=500):
    if pd.isna(row[base_col]):
        return np.nan
    return np.log(row["close"]+base_price) - np.log(row[base_col]+base_price)

def compute_ret_velocity(row, ret_col, time_col):
    if pd.isna(row[time_col]) or row[time_col]==0:
        return np.nan
    return row[ret_col]/row[time_col]

def compute_ret_from_vwap(row, dv_col, volume_col, base_price=500):
    if pd.isna(row[dv_col]) or pd.isna(row[volume_col]):
        return np.nan
    else:
        if row["cum_volume"]>row[volume_col]:
            if row["cum_dv"] == row[dv_col]:
                return 0
            else:
                vwap_price = (row["cum_dv"] - row[dv_col])/(row["cum_volume"]-row[volume_col])
                return np.log(row["close"]+base_price) - np.log(vwap_price+base_price)
        else:
            return 0

def fill_cum_dv(row, time_col, pre_interval_mins=30, post_interval_mins=30):
    #if interval_mins == 30 or interval_mins == 60:
    before_mins=pre_interval_mins*60*0.5+1
    after_mins=post_interval_mins*60*0.5+1
    #else:
    #    before_mins=interval_mins*60*0.9+1
    #    after_mins=interval_mins*60*0.9+1

    if pd.isna(row[time_col]):
        #logging.error(f"no time_col:{time_col}")
        return np.nan
    else:
        #logging.error(f"checking time_col:{time_col}, before_mins:{before_mins}, after_mins:{after_mins}, row[time]:{row[time_col]}, row[timestamp]:{row['timestamp']}")
        if ((row["timestamp"] > row[time_col]-before_mins) and
            (row["timestamp"] < row[time_col]+after_mins)):
            #logging.error(f"find cum_dv")
            return row["cum_dv"]
        else:
            # A tricky situation is that when we compute pre_vwap, last_close_time will
            # be one day ahead.
            next_time_check = row[time_col] + 86400
            if ((row["timestamp"] > next_time_check-before_mins) and
                (row["timestamp"] < next_time_check+after_mins)):
                return row["cum_dv"]
            else:
                return np.nan

def fill_cum_volume(row, time_col, pre_interval_mins=30, post_interval_mins=30):
    before_mins=pre_interval_mins*60*.5+1
    after_mins=post_interval_mins*60*.5+1
    if pd.isna(row[time_col]):
        return np.nan
    else:
        if ((row["timestamp"] > row[time_col]-before_mins) and
            (row["timestamp"] < row[time_col]+after_mins)):
            return row["cum_volume"]
        else:
            # A tricky situation is that when we compute pre_vwap, last_close_time will
            # be one day ahead.
            next_time_check = row[time_col] + 86400
            if ((row["timestamp"] > next_time_check-before_mins) and
                (row["timestamp"] < next_time_check+after_mins)):
                return row["cum_dv"]
            else:
                return np.nan

def fill_open(row, time_col, pre_interval_mins=30, post_interval_mins=30):
    before_mins=pre_interval_mins*60*.5+1
    after_mins=post_interval_mins*60*.5+1
    if pd.isna(row[time_col]):
        return np.nan
    else:
        #logging.error(f"open row:{row['timestamp']}, time_col:{time_col}, before:{row[time_col]-before_mins}, after:{row[time_col]+after_mins}")
        if ((row["timestamp"] > row[time_col]-before_mins) and
            (row["timestamp"] < row[time_col]+after_mins)):
            return row["open"]
        else:
            return np.nan

#def fill_close(row, is_col):
#    if not row[is_col]:
#        return np.nan
#    else:
#        return row["close"]

        
def fill_close(row, time_col):
    if pd.isna(row[time_col]) or not row["is_new_york_close"]:
        return np.nan
    else:
        return row["close"]
        #logging.error(f"close row:{row['timestamp']}, time_col:{time_col}, check_time:{row[time_col]}, before:{row[time_col]-before_mins}, after:{row[time_col]+after_mins}")
        check_time = row[time_col]
        if ((row["timestamp"] > check_time-before_mins) and
            (row["timestamp"] < check_time+after_mins)):
            return row["close"]
        else:
            return np.nan

#@profile_util.profile
#@njit(parallel=True)
def group_features(sorted_data, config: dict) -> pd.DataFrame:
    interval_minutes = config.dataset.interval_mins
    add_daily_rolling_features = config.model.features.add_daily_rolling_features
    base_price = config.dataset.base_price
    for column in [
        "close_back", "high_back", "low_back", "open_back", "close_velocity_back",
        "volume_back",
        "dv_back",
        #"close_fwd",
        #"volume_fwd",
        #"dv_fwd",
        "cum_volume",
        "cum_dv",
        "close_back_cumsum",
        "volume_back_cumsum",
        "close_back_cumsum_high_5_ff",
        "time_high_5_ff",
        "close_back_cumsum_low_5_ff",
        "time_low_5_ff",
        "close_back_cumsum_high_11_ff",
        "time_high_11_ff",
        "close_back_cumsum_low_11_ff",
        "time_low_11_ff",
        "close_back_cumsum_high_21_ff",
        "time_high_21_ff",
        "close_back_cumsum_low_21_ff",
        "time_low_21_ff",
        "close_back_cumsum_high_51_ff",
        "time_high_51_ff",
        "close_back_cumsum_low_51_ff",
        "time_low_51_ff",
        "close_back_cumsum_high_201_ff",
        "time_high_201_ff",
        "close_back_cumsum_low_201_ff",
        "time_low_201_ff",
        "close_back_cumsum_high_5d_ff",
        "time_high_5d_ff",
        "close_back_cumsum_low_5d_ff",
        "time_low_5d_ff",
        "close_back_cumsum_high_11d_ff",
        "time_high_11d_ff",
        "close_back_cumsum_low_11d_ff",
        "time_low_11d_ff",
        "close_back_cumsum_high_21d_ff",
        "time_high_21d_ff",
        "close_back_cumsum_low_21d_ff",
        "time_low_21d_ff",
        "close_back_cumsum_high_51d_ff",
        "time_high_51d_ff",
        "close_back_cumsum_low_51d_ff",
        "time_low_51d_ff",
        "close_back_cumsum_high_201d_ff",
        "time_high_201d_ff",
        "close_back_cumsum_low_201d_ff",
        "time_low_201d_ff",
        'close_rolling_5d_max', 'close_rolling_201_min',
        'sma_5d', 'sma_10d', 'sma_20d', 'sma_50d', 'sma_100d', 'sma_200d',
        "rsi",
        "macd",
        "macd_signal",
        "bb_high_5_2",
        "bb_low_5_2",
        "bb_high_5_3",
        "bb_low_5_3",
        "bb_high_10_2",
        "bb_low_10_2",
        "bb_high_10_3",
        "bb_low_10_3",
        "bb_high_10_2",
        "bb_low_20_2",
        "bb_high_20_3",
        "bb_low_20_3",
        "bb_high_50_2",
        "bb_low_50_2",
        "bb_high_50_3",
        "bb_low_50_3",
        "bb_high_100_2",
        "bb_low_100_2",
        "bb_high_100_3",
        "bb_low_100_3",
        "bb_high_200_2",
        "bb_low_200_2",
        "bb_high_200_3",
        "bb_low_200_3",
        "sma_5",
        "sma_10",
        "sma_20",
        "sma_50",
        "sma_100",
        "sma_200",
        'close_rolling_5_max', 'close_rolling_5_min', 'close_rolling_11_max',
        'close_rolling_11_min', 'close_rolling_21_max',
        'close_rolling_21_min', 'close_rolling_51_max',
        'close_rolling_51_min', 'close_rolling_201_max',
        'close_rolling_201_min'
        'close_rolling_5d_max', 'close_rolling_5d_min', 'close_rolling_11d_max',
        'close_rolling_11d_min', 'close_rolling_21d_max',
        'close_rolling_21d_min', 'close_rolling_51d_max',
        'close_rolling_51d_min', 'close_rolling_201d_max',
        'close_rolling_201d_min'
    ]:
        if column in raw_data.columns:
            raw_data = raw_data.drop(columns=[column])
    new_features = raw_data.sort_values(['timestamp']).groupby(["ticker"], group_keys=False)[[
        "volume", "dv", "close", "high", "low", "open", "timestamp"]].apply(ticker_transform, config)
    new_features = new_features.drop(columns=["volume", "dv", "close", "high", "low", "open", "timestamp"])
    raw_data = raw_data.join(new_features)
    raw_data.reset_index(drop=True, inplace=True)
    #del new_features

    return raw_data



#@profile_util.profile
def example_group_features(cal:any, macro_data_builder:any, example_level_features, config:dict) ->pd.DataFrame:
    add_daily_rolling_features=config.model.features.add_daily_rolling_features
    interval_mins = config.dataset.interval_mins
    new_york_cal = mcal.get_calendar("NYSE")
    lse_cal = mcal.get_calendar("LSE")

    def vwap_around(ser, cum_dv_col, cum_volume_col):
        data = raw_data.loc[ser.index]
        #logging.error(f"ser:{ser}, data:{data}, cum_dv_col:{cum_dv_col}")
        cum_volume_diff = data[cum_volume_col].iloc[-1]-data[cum_volume_col].iloc[0]
        cum_dv_diff = data[cum_dv_col].iloc[-1]-data[cum_dv_col].iloc[0]
        if cum_volume_diff>0:
            return cum_dv_diff/cum_volume_diff
        else:
            return data["close"].iloc[-1]
        return 0

    if add_daily_rolling_features:
        raw_data["ret_from_close_cumsum_high_5d"] = raw_data.apply(
            time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_back_cumsum_high_5d_ff"
        )
        raw_data["ret_from_close_cumsum_high_11d"] = raw_data.apply(
            time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_back_cumsum_high_11d_ff"
        )
        raw_data["ret_from_close_cumsum_high_21d"] = raw_data.apply(
            time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_back_cumsum_high_21d_ff"
        )
        raw_data["ret_from_close_cumsum_high_51d"] = raw_data.apply(
            time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_back_cumsum_high_51d_ff"
        )
        raw_data["ret_from_close_cumsum_high_101d"] = raw_data.apply(
            time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_back_cumsum_high_101d_ff"
        )
        raw_data["ret_from_close_cumsum_high_201d"] = raw_data.apply(
            time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_back_cumsum_high_201d_ff"
        )
        raw_data["ret_from_close_cumsum_low_5d"] = raw_data.apply(
            time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_back_cumsum_low_5d_ff"
        )
        raw_data["ret_from_close_cumsum_low_11d"] = raw_data.apply(
            time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_back_cumsum_low_11d_ff"
        )
        raw_data["ret_from_close_cumsum_low_21d"] = raw_data.apply(
            time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_back_cumsum_low_21d_ff"
        )
        raw_data["ret_from_close_cumsum_low_51d"] = raw_data.apply(
            time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_back_cumsum_low_51d_ff"
        )
        raw_data["ret_from_close_cumsum_low_101d"] = raw_data.apply(
            time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_back_cumsum_low_101d_ff"
        )
        raw_data["ret_from_close_cumsum_low_201d"] = raw_data.apply(
            time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_back_cumsum_low_201d_ff"
        )

        raw_data["ret_from_high_1d"] = raw_data.apply(compute_ret, axis=1, base_col="close_high_1d_ff")
        raw_data["ret_from_high_1d_shift_1d"] = raw_data.apply(compute_ret, axis=1, base_col="close_high_1d_ff_shift_1d")
        raw_data["ret_from_high_5d"] = raw_data.apply(compute_ret, axis=1, base_col="close_high_5d_ff")
        raw_data["ret_from_high_5d_shift_5d"] = raw_data.apply(compute_ret, axis=1, base_col="close_high_5d_ff_shift_5d")
        raw_data["ret_from_high_11d"] = raw_data.apply(compute_ret, axis=1, base_col="close_high_11d_ff")
        raw_data["ret_from_high_11d_shift_11d"] = raw_data.apply(compute_ret, axis=1, base_col="close_high_11d_ff_shift_11d")
        raw_data["ret_from_high_21d"] = raw_data.apply(compute_ret, axis=1, base_col="close_high_21d_ff")
        raw_data["ret_from_high_21d_shift_21d"] = raw_data.apply(compute_ret, axis=1, base_col="close_high_21d_ff_shift_21d")
        raw_data["ret_from_high_51d"] = raw_data.apply(compute_ret, axis=1, base_col="close_high_51d_ff")
        raw_data["ret_from_high_101d"] = raw_data.apply(compute_ret, axis=1, base_col="close_high_101d_ff")
        raw_data["ret_from_high_201d"] = raw_data.apply(compute_ret, axis=1, base_col="close_high_201d_ff")
        raw_data["ret_from_low_1d"] = raw_data.apply(compute_ret, axis=1, base_col="close_low_1d_ff")
        raw_data["ret_from_low_1d_shift_1d"] = raw_data.apply(compute_ret, axis=1, base_col="close_low_1d_ff_shift_1d")
        raw_data["ret_from_low_5d"] = raw_data.apply(compute_ret, axis=1, base_col="close_low_5d_ff")
        raw_data["ret_from_low_5d_shift_5d"] = raw_data.apply(compute_ret, axis=1, base_col="close_low_5d_ff_shift_5d")
        raw_data["ret_from_low_11d"] = raw_data.apply(compute_ret, axis=1, base_col="close_low_11d_ff")
        raw_data["ret_from_low_11d_shift_11d"] = raw_data.apply(compute_ret, axis=1, base_col="close_low_11d_ff_shift_11d")
        raw_data["ret_from_low_21d"] = raw_data.apply(compute_ret, axis=1, base_col="close_low_21d_ff")
        raw_data["ret_from_low_21d_shift_21d"] = raw_data.apply(compute_ret, axis=1, base_col="close_low_21d_ff_shift_21d")
        raw_data["ret_from_low_51d"] = raw_data.apply(compute_ret, axis=1, base_col="close_low_51d_ff")
        raw_data["ret_from_low_101d"] = raw_data.apply(compute_ret, axis=1, base_col="close_low_101d_ff")
        raw_data["ret_from_low_201d"] = raw_data.apply(compute_ret, axis=1, base_col="close_low_201d_ff")

        raw_data["ret_from_high_5d_bf"] = raw_data.apply(compute_ret, axis=1, base_col="close_high_5d_bf")
        raw_data["ret_from_high_11d_bf"] = raw_data.apply(compute_ret, axis=1, base_col="close_high_11d_bf")
        raw_data["ret_from_high_21d_bf"] = raw_data.apply(compute_ret, axis=1, base_col="close_high_21d_bf")
        raw_data["ret_from_high_51d_bf"] = raw_data.apply(compute_ret, axis=1, base_col="close_high_51d_bf")
        raw_data["ret_from_low_5d_bf"] = raw_data.apply(compute_ret, axis=1, base_col="close_low_5d_bf")
        raw_data["ret_from_low_11d_bf"] = raw_data.apply(compute_ret, axis=1, base_col="close_low_11d_bf")
        raw_data["ret_from_low_21d_bf"] = raw_data.apply(compute_ret, axis=1, base_col="close_low_21d_bf")
        raw_data["ret_from_low_51d_bf"] = raw_data.apply(compute_ret, axis=1, base_col="close_low_51d_bf")

    
    raw_data["ret_from_close_cumsum_high_5"] = raw_data.apply(
        time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_back_cumsum_high_5_ff"
    )
    raw_data["ret_from_close_cumsum_high_11"] = raw_data.apply(
        time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_back_cumsum_high_11_ff"
    )
    raw_data["ret_from_close_cumsum_high_21"] = raw_data.apply(
        time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_back_cumsum_high_21_ff"
    )
    raw_data["ret_from_close_cumsum_high_51"] = raw_data.apply(
        time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_back_cumsum_high_51_ff"
    )
    raw_data["ret_from_close_cumsum_high_201"] = raw_data.apply(
        time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_back_cumsum_high_201_ff"
    )
    raw_data["ret_from_close_cumsum_low_5"] = raw_data.apply(
        time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_back_cumsum_low_5_ff"
    )
    raw_data["ret_from_close_cumsum_low_11"] = raw_data.apply(
        time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_back_cumsum_low_11_ff"
    )
    raw_data["ret_from_close_cumsum_low_21"] = raw_data.apply(
        time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_back_cumsum_low_21_ff"
    )
    raw_data["ret_from_close_cumsum_low_51"] = raw_data.apply(
        time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_back_cumsum_low_51_ff"
    )
    raw_data["ret_from_close_cumsum_low_201"] = raw_data.apply(
        time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_back_cumsum_low_201_ff"
    )

    raw_data["ret_from_high_5"] = raw_data.apply(compute_ret, axis=1, base_col="close_high_5_ff")
    raw_data["ret_velocity_from_high_5"] = raw_data.apply(compute_ret_velocity, axis=1, ret_col="ret_from_high_5", time_col="time_to_high_5_ff")
    raw_data["ret_from_high_11"] = raw_data.apply(compute_ret, axis=1, base_col="close_high_11_ff")
    raw_data["ret_velocity_from_high_11"] = raw_data.apply(compute_ret_velocity, axis=1, ret_col="ret_from_high_11", time_col="time_to_high_11_ff")
    raw_data["ret_from_high_21"] = raw_data.apply(compute_ret, axis=1, base_col="close_high_21_ff")
    raw_data["ret_velocity_from_high_21"] = raw_data.apply(compute_ret_velocity, axis=1, ret_col="ret_from_high_21", time_col="time_to_high_21_ff")
    raw_data["ret_from_high_51"] = raw_data.apply(compute_ret, axis=1, base_col="close_high_51_ff")
    raw_data["ret_velocity_from_high_51"] = raw_data.apply(compute_ret_velocity, axis=1, ret_col="ret_from_high_51", time_col="time_to_high_51_ff")
    raw_data["ret_from_high_101"] = raw_data.apply(compute_ret, axis=1, base_col="close_high_101_ff")
    raw_data["ret_velocity_from_high_101"] = raw_data.apply(compute_ret_velocity, axis=1, ret_col="ret_from_high_101", time_col="time_to_high_101_ff")
    raw_data["ret_from_high_201"] = raw_data.apply(compute_ret, axis=1, base_col="close_high_201_ff")
    raw_data["ret_velocity_from_high_201"] = raw_data.apply(compute_ret_velocity, axis=1, ret_col="ret_from_high_201", time_col="time_to_high_201_ff")
    raw_data["ret_from_low_5"] = raw_data.apply(compute_ret, axis=1, base_col="close_low_5_ff")
    raw_data["ret_velocity_from_low_5"] = raw_data.apply(compute_ret_velocity, axis=1, ret_col="ret_from_low_5", time_col="time_to_low_5_ff")
    raw_data["ret_from_low_11"] = raw_data.apply(compute_ret, axis=1, base_col="close_low_11_ff")
    raw_data["ret_velocity_from_low_11"] = raw_data.apply(compute_ret_velocity, axis=1, ret_col="ret_from_low_11", time_col="time_to_low_11_ff")
    raw_data["ret_from_low_21"] = raw_data.apply(compute_ret, axis=1, base_col="close_low_21_ff")
    raw_data["ret_velocity_from_low_21"] = raw_data.apply(compute_ret_velocity, axis=1, ret_col="ret_from_low_21", time_col="time_to_low_21_ff")
    raw_data["ret_from_low_51"] = raw_data.apply(compute_ret, axis=1, base_col="close_low_51_ff")
    raw_data["ret_velocity_from_low_51"] = raw_data.apply(compute_ret_velocity, axis=1, ret_col="ret_from_low_51", time_col="time_to_low_51_ff")
    raw_data["ret_from_low_101"] = raw_data.apply(compute_ret, axis=1, base_col="close_low_101_ff")
    raw_data["ret_velocity_from_low_101"] = raw_data.apply(compute_ret_velocity, axis=1, ret_col="ret_from_low_101", time_col="time_to_low_101_ff")
    raw_data["ret_from_low_201"] = raw_data.apply(compute_ret, axis=1, base_col="close_low_201_ff")
    raw_data["ret_velocity_from_low_201"] = raw_data.apply(compute_ret_velocity, axis=1, ret_col="ret_from_low_201", time_col="time_to_low_201_ff")

    if macro_data_builder.add_macro_event:
        raw_data["last_macro_event_cum_dv_imp1"] = raw_data.apply(fill_cum_dv, time_col="last_macro_event_time_imp1",
                                                                  pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3, axis=1)
        raw_data["last_macro_event_cum_volume_imp1"] = raw_data.apply(fill_cum_volume, time_col="last_macro_event_time_imp1",
                                                                      pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3, axis=1)
        raw_data["last_macro_event_cum_dv_imp1"] = raw_data.last_macro_event_cum_dv_imp1.ffill()
        raw_data["last_macro_event_cum_volume_imp1"] = raw_data.last_macro_event_cum_volume_imp1.ffill()
        raw_data["ret_from_vwap_since_last_macro_event_imp1"] = raw_data.apply(compute_ret_from_vwap, dv_col="last_macro_event_cum_dv_imp1", volume_col="last_macro_event_cum_volume_imp1", axis=1)

        raw_data["pre_macro_event_cum_dv_imp1"] = raw_data.apply(fill_cum_dv, time_col="last_macro_event_time_imp1",
                                                                 pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)
        raw_data["pre_macro_event_cum_volume_imp1"] = raw_data.apply(fill_cum_volume, time_col="last_macro_event_time_imp1",
                                                                     pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)
        rol = raw_data.pre_macro_event_cum_dv_imp1.rolling(window=2)
        raw_data["vwap_pre_macro_event_imp1"] = rol.apply(vwap_around, args=("pre_macro_event_cum_dv_imp1","pre_macro_event_cum_volume_imp1"), raw=False)
        raw_data["vwap_pre_macro_event_imp1"] = raw_data.vwap_pre_macro_event_imp1.ffill()
        raw_data["ret_from_vwap_pre_macro_event_imp1"] = raw_data.apply(compute_ret, base_col="vwap_pre_macro_event_imp1", axis=1)

        raw_data["post_macro_event_cum_dv_imp1"] = raw_data.apply(fill_cum_dv, time_col="last_macro_event_time_imp1",
                                                                  pre_interval_mins=0, post_interval_mins=interval_mins*3, axis=1)
        raw_data["post_macro_event_cum_volume_imp1"] = raw_data.apply(fill_cum_volume, time_col="last_macro_event_time_imp1",
                                                                      pre_interval_mins=0, post_interval_mins=interval_mins*3, axis=1)
        rol = raw_data.post_macro_event_cum_dv_imp1.rolling(window=2)
        raw_data["vwap_post_macro_event_imp1"] = rol.apply(vwap_around, args=("post_macro_event_cum_dv_imp1","post_macro_event_cum_volume_imp1"), raw=False)
        raw_data["vwap_post_macro_event_imp1"] = raw_data.vwap_post_macro_event_imp1.ffill()
        raw_data["ret_from_vwap_post_macro_event_imp1"] = raw_data.apply(compute_ret, base_col="vwap_post_macro_event_imp1", axis=1)

        raw_data["around_macro_event_cum_dv_imp1"] = raw_data.apply(fill_cum_dv, time_col="last_macro_event_time_imp1",
                                                                    pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3, axis=1)
        raw_data["around_macro_event_cum_volume_imp1"] = raw_data.apply(fill_cum_volume, time_col="last_macro_event_time_imp1",
                                                                        pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3, axis=1)
        rol = raw_data.around_macro_event_cum_dv_imp1.rolling(window=2)
        raw_data["vwap_around_macro_event_imp1"] = rol.apply(vwap_around, args=("around_macro_event_cum_dv_imp1","around_macro_event_cum_volume_imp1"), raw=False)
        raw_data["vwap_around_macro_event_imp1"] = raw_data.vwap_around_macro_event_imp1.ffill()
        raw_data["ret_from_vwap_around_macro_event_imp1"] = raw_data.apply(compute_ret, base_col="vwap_around_macro_event_imp1", axis=1)

        #raw_data = raw_data.drop(columns=["around_macro_event_cum_dv_imp1", "around_macro_event_cum_volume_imp1",
        #                                  "last_macro_event_cum_dv_imp1", "last_macro_event_cum_volume_imp1",
        #                                  "around_macro_event_cum_dv_imp1", "around_macro_event_cum_volume_imp1"])

        raw_data["last_macro_event_cum_dv_imp2"] = raw_data.apply(fill_cum_dv, time_col="last_macro_event_time_imp2", axis=1,
                                                                  pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3)
        raw_data["last_macro_event_cum_volume_imp2"] = raw_data.apply(fill_cum_volume, time_col="last_macro_event_time_imp2", axis=1,
                                                                      pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3,)
        raw_data["last_macro_event_cum_dv_imp2"] = raw_data.last_macro_event_cum_dv_imp2.ffill()
        raw_data["last_macro_event_cum_volume_imp2"] = raw_data.last_macro_event_cum_volume_imp2.ffill()
        raw_data["ret_from_vwap_since_last_macro_event_imp2"] = raw_data.apply(compute_ret_from_vwap, dv_col="last_macro_event_cum_dv_imp2", volume_col="last_macro_event_cum_volume_imp2", axis=1)

        raw_data["pre_macro_event_cum_dv_imp2"] = raw_data.apply(fill_cum_dv, time_col="last_macro_event_time_imp2",
                                                                 pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)
        raw_data["pre_macro_event_cum_volume_imp2"] = raw_data.apply(fill_cum_volume, time_col="last_macro_event_time_imp2",
                                                                     pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)
        rol = raw_data.pre_macro_event_cum_dv_imp2.rolling(window=2)
        raw_data["vwap_pre_macro_event_imp2"] = rol.apply(vwap_around, args=("pre_macro_event_cum_dv_imp2","pre_macro_event_cum_volume_imp2"), raw=False)
        raw_data["vwap_pre_macro_event_imp2"] = raw_data.vwap_pre_macro_event_imp2.ffill()
        raw_data["ret_from_vwap_pre_macro_event_imp2"] = raw_data.apply(compute_ret, base_col="vwap_pre_macro_event_imp2", axis=1)

        raw_data["post_macro_event_cum_dv_imp2"] = raw_data.apply(fill_cum_dv, time_col="last_macro_event_time_imp2",
                                                                  pre_interval_mins=0, post_interval_mins=interval_mins*3, axis=1)
        raw_data["post_macro_event_cum_volume_imp2"] = raw_data.apply(fill_cum_volume, time_col="last_macro_event_time_imp2",
                                                                      pre_interval_mins=0, post_interval_mins=interval_mins*3, axis=1)
        rol = raw_data.post_macro_event_cum_dv_imp2.rolling(window=2)
        raw_data["vwap_post_macro_event_imp2"] = rol.apply(vwap_around, args=("post_macro_event_cum_dv_imp2","post_macro_event_cum_volume_imp2"), raw=False)
        raw_data["vwap_post_macro_event_imp2"] = raw_data.vwap_post_macro_event_imp1.ffill()
        raw_data["ret_from_vwap_post_macro_event_imp2"] = raw_data.apply(compute_ret, base_col="vwap_post_macro_event_imp2", axis=1)

        raw_data["around_macro_event_cum_dv_imp2"] = raw_data.apply(fill_cum_dv, time_col="last_macro_event_time_imp2", axis=1,
                                                                    pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3,)
        raw_data["around_macro_event_cum_volume_imp2"] = raw_data.apply(fill_cum_volume, time_col="last_macro_event_time_imp2", axis=1,
                                                                        pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3,)
        rol = raw_data.around_macro_event_cum_dv_imp2.rolling(window=2)
        raw_data["vwap_around_macro_event_imp2"] = rol.apply(vwap_around, args=("around_macro_event_cum_dv_imp2","around_macro_event_cum_volume_imp2"), raw=False)
        raw_data["vwap_around_macro_event_imp2"] = raw_data.vwap_around_macro_event_imp2.ffill()
        raw_data["ret_from_vwap_around_macro_event_imp2"] = raw_data.apply(compute_ret, base_col="vwap_around_macro_event_imp2", axis=1)
        #logging.error(f"after raw_data_with_macro:{raw_data[(raw_data.timestamp>1643797800) & (raw_data.timestamp<1643814000)]}")
        #raw_data = raw_data.drop(columns=["around_macro_event_cum_dv_imp2", "around_macro_event_cum_volume_imp2",
        #                                  "last_macro_event_cum_dv_imp2", "last_macro_event_cum_volume_imp2",
        #                                  "around_macro_event_cum_dv_imp2", "around_macro_event_cum_volume_imp2"])

        raw_data["last_macro_event_cum_dv_imp3"] = raw_data.apply(
            fill_cum_dv, time_col="last_macro_event_time_imp3", pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3, axis=1)
        raw_data["last_macro_event_cum_volume_imp3"] = raw_data.apply(
            fill_cum_volume, time_col="last_macro_event_time_imp3", pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3, axis=1)
        raw_data["last_macro_event_cum_dv_imp3"] = raw_data.last_macro_event_cum_dv_imp3.ffill()
        raw_data["last_macro_event_cum_volume_imp3"] = raw_data.last_macro_event_cum_volume_imp3.ffill()
        raw_data["ret_from_vwap_since_last_macro_event_imp3"] = raw_data.apply(
            compute_ret_from_vwap, dv_col="last_macro_event_cum_dv_imp3",
            volume_col="last_macro_event_cum_volume_imp3", axis=1)

        raw_data["last_macro_event_cum_dv_imp3"] = raw_data.apply(fill_cum_dv, time_col="last_macro_event_time_imp3", axis=1,
                                                                  pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3)
        raw_data["last_macro_event_cum_volume_imp3"] = raw_data.apply(fill_cum_volume, time_col="last_macro_event_time_imp3", axis=1,
                                                                      pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3)
        raw_data["last_macro_event_cum_dv_imp3"] = raw_data.last_macro_event_cum_dv_imp3.ffill()
        raw_data["last_macro_event_cum_volume_imp3"] = raw_data.last_macro_event_cum_volume_imp3.ffill()
        raw_data["ret_from_vwap_since_last_macro_event_imp3"] = raw_data.apply(compute_ret_from_vwap, dv_col="last_macro_event_cum_dv_imp3", volume_col="last_macro_event_cum_volume_imp3", axis=1)

        raw_data["pre_macro_event_cum_dv_imp3"] = raw_data.apply(fill_cum_dv, time_col="last_macro_event_time_imp3",
                                                                 pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)
        raw_data["pre_macro_event_cum_volume_imp3"] = raw_data.apply(fill_cum_volume, time_col="last_macro_event_time_imp3",
                                                                     pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)
        rol = raw_data.pre_macro_event_cum_dv_imp3.rolling(window=2)
        raw_data["vwap_pre_macro_event_imp3"] = rol.apply(vwap_around, args=("pre_macro_event_cum_dv_imp3","pre_macro_event_cum_volume_imp3"), raw=False)
        raw_data["vwap_pre_macro_event_imp3"] = raw_data.vwap_pre_macro_event_imp3.ffill()
        raw_data["ret_from_vwap_pre_macro_event_imp3"] = raw_data.apply(compute_ret, base_col="vwap_pre_macro_event_imp3", axis=1)

        raw_data["post_macro_event_cum_dv_imp3"] = raw_data.apply(fill_cum_dv, time_col="last_macro_event_time_imp3",
                                                                  pre_interval_mins=0, post_interval_mins=interval_mins*3, axis=1)
        raw_data["post_macro_event_cum_volume_imp3"] = raw_data.apply(fill_cum_volume, time_col="last_macro_event_time_imp3",
                                                                      pre_interval_mins=0, post_interval_mins=interval_mins*3, axis=1)
        rol = raw_data.post_macro_event_cum_dv_imp3.rolling(window=2)
        raw_data["vwap_post_macro_event_imp3"] = rol.apply(vwap_around, args=("post_macro_event_cum_dv_imp3","post_macro_event_cum_volume_imp3"), raw=False)
        raw_data["vwap_post_macro_event_imp3"] = raw_data.vwap_post_macro_event_imp3.ffill()
        raw_data["ret_from_vwap_post_macro_event_imp3"] = raw_data.apply(compute_ret, base_col="vwap_post_macro_event_imp3", axis=1)

        raw_data["around_macro_event_cum_dv_imp3"] = raw_data.apply(
            fill_cum_dv, time_col="last_macro_event_time_imp3", pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3,
            axis=1)
        raw_data["around_macro_event_cum_volume_imp3"] = raw_data.apply(
            fill_cum_volume, time_col="last_macro_event_time_imp3",
            pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3,
            axis=1)
        rol = raw_data.around_macro_event_cum_dv_imp3.rolling(window=2)
        raw_data["vwap_around_macro_event_imp3"] = rol.apply(
            vwap_around,
            args=("around_macro_event_cum_dv_imp3","around_macro_event_cum_volume_imp3"), raw=False)
        raw_data["vwap_around_macro_event_imp3"] = raw_data.vwap_around_macro_event_imp3.ffill()
        raw_data["ret_from_vwap_around_macro_event_imp3"] = raw_data.apply(
            compute_ret, base_col="vwap_around_macro_event_imp3", axis=1)

        #raw_data = raw_data.drop(columns=["around_macro_event_cum_dv_imp3", "around_macro_event_cum_volume_imp3",
        #                                  "last_macro_event_cum_dv_imp3", "last_macro_event_cum_volume_imp3",
        #                                  "around_macro_event_cum_dv_imp3", "around_macro_event_cum_volume_imp3"])

    raw_data["new_york_open_cum_dv"] = raw_data.apply(fill_cum_dv, time_col="new_york_last_open_time",
                                                      pre_interval_mins=interval_mins*2, post_interval_mins=interval_mins*2, axis=1)
    raw_data["new_york_open_cum_volume"] = raw_data.apply(fill_cum_volume, time_col="new_york_last_open_time",
                                                          pre_interval_mins=interval_mins*2, post_interval_mins=interval_mins*2, axis=1)
    raw_data["new_york_open_cum_dv"] = raw_data.new_york_open_cum_dv.ffill()
    raw_data["new_york_open_cum_volume"] = raw_data.new_york_open_cum_volume.ffill()
    raw_data["ret_from_vwap_since_new_york_open"] = raw_data.apply(compute_ret_from_vwap, dv_col="new_york_open_cum_dv", volume_col="new_york_open_cum_volume", axis=1)

    raw_data["london_open_cum_dv"] = raw_data.apply(fill_cum_dv, time_col="london_last_open_time", pre_interval_mins=interval_mins*2,
                                                    post_interval_mins=interval_mins*2, axis=1)
    raw_data["london_open_cum_dv"] = raw_data.london_open_cum_dv.ffill()
    raw_data["london_open_cum_volume"] = raw_data.apply(fill_cum_volume, time_col="london_last_open_time", pre_interval_mins=interval_mins*2,
                                                        post_interval_mins=interval_mins*2, axis=1)
    raw_data["london_open_cum_volume"] = raw_data.london_open_cum_volume.ffill()
    raw_data["ret_from_vwap_since_london_open"] = raw_data.apply(compute_ret_from_vwap, dv_col="london_open_cum_dv", volume_col="london_open_cum_volume", axis=1)    
    
    raw_data["pre_new_york_open_cum_dv"] = raw_data.apply(fill_cum_dv, time_col="new_york_last_open_time",
                                                             pre_interval_mins=interval_mins*2, post_interval_mins=0, axis=1)
    raw_data["pre_new_york_open_cum_volume"] = raw_data.apply(fill_cum_volume, time_col="new_york_last_open_time",
                                                              pre_interval_mins=interval_mins*2, post_interval_mins=0, axis=1)
    rol = raw_data.pre_new_york_open_cum_dv.rolling(window=2)
    raw_data["vwap_pre_new_york_open"] = rol.apply(vwap_around, args=("pre_new_york_open_cum_dv","pre_new_york_open_cum_volume"), raw=False)
    raw_data["vwap_pre_new_york_open"] = raw_data.vwap_pre_new_york_open.ffill()
    raw_data["ret_from_vwap_pre_new_york_open"] = raw_data.apply(compute_ret, base_col="vwap_pre_new_york_open", axis=1)
    raw_data = raw_data.drop(columns=["pre_new_york_open_cum_dv", "pre_new_york_open_cum_volume"])

    raw_data["post_new_york_open_cum_dv"] = raw_data.apply(fill_cum_dv, time_col="new_york_last_open_time",
                                                             pre_interval_mins=0, post_interval_mins=interval_mins*2, axis=1)
    raw_data["post_new_york_open_cum_volume"] = raw_data.apply(fill_cum_volume, time_col="new_york_last_open_time",
                                                               pre_interval_mins=0, post_interval_mins=interval_mins*2, axis=1)
    rol = raw_data.post_new_york_open_cum_dv.rolling(window=2)
    raw_data["vwap_post_new_york_open"] = rol.apply(vwap_around, args=("post_new_york_open_cum_dv","post_new_york_open_cum_volume"), raw=False)
    raw_data["vwap_post_new_york_open"] = raw_data.vwap_post_new_york_open.ffill()
    raw_data["ret_from_vwap_post_new_york_open"] = raw_data.apply(compute_ret, base_col="vwap_post_new_york_open", axis=1)
    raw_data = raw_data.drop(columns=["post_new_york_open_cum_dv", "post_new_york_open_cum_volume"])

    for idx in config.model.daily_lookback:
        raw_data[f"new_york_last_daily_open_{idx}"] = raw_data.apply(fill_close, time_col=f"new_york_last_open_time_{idx}",
                                                                     pre_interval_mins=0, post_interval_mins=interval_mins, axis=1)
        raw_data[f"new_york_last_daily_close_{idx}"] = raw_data.apply(fill_close, time_col=f"new_york_last_close_time_{idx}",
                                                                      pre_interval_mins=0, post_interval_mins=interval_mins, axis=1)
        raw_data[f"london_last_daily_open_{idx}"] = raw_data.apply(fill_close, time_col=f"london_last_open_time_{idx}",
                                                                   pre_interval_mins=0, post_interval_mins=interval_mins*2, axis=1)
        raw_data[f"london_last_daily_close_{idx}"] = raw_data.apply(fill_close, time_col=f"new_york_last_close_time_{idx}",
                                                                    pre_interval_mins=0, post_interval_mins=interval_mins*2, axis=1)    
        raw_data[f"new_york_last_daily_open_{idx}"] = raw_data[f"new_york_last_daily_open_{idx}"].ffill()
        raw_data[f"new_york_last_daily_close_{idx}"] = raw_data[f"new_york_last_daily_close_{idx}"].ffill()
        raw_data[f"london_last_daily_open_{idx}"] = raw_data[f"london_last_daily_open_{idx}"].ffill()
        raw_data[f"london_last_daily_close_{idx}"] = raw_data[f"london_last_daily_close_{idx}"].ffill()

        raw_data[f"ret_from_new_york_last_daily_open_{idx}"] = raw_data.apply(compute_ret, base_col=f"new_york_last_daily_open_{idx}", axis=1)
        raw_data[f"ret_from_new_york_last_daily_close_{idx}"] = raw_data.apply(compute_ret, base_col=f"new_york_last_daily_close_{idx}", axis=1)
        raw_data[f"ret_from_london_last_daily_open_{idx}"] = raw_data.apply(compute_ret, base_col=f"london_last_daily_open_{idx}", axis=1)
        raw_data[f"ret_from_london_last_daily_close_{idx}"] = raw_data.apply(compute_ret, base_col=f"london_last_daily_close_{idx}", axis=1)

    for idx in range(config.model.weekly_lookback):
        raw_data[f"last_weekly_close_{idx}"] = raw_data.apply(fill_close, time_col=f"last_weekly_close_time_{idx}",
                                                              pre_interval_mins=0, post_interval_mins=interval_mins, axis=1)
        raw_data[f"last_weekly_close_{idx}"] = raw_data[f"last_weekly_close_{idx}"].ffill()
        raw_data[f"ret_from_last_weekly_close_{idx}"] = raw_data.apply(compute_ret, base_col=f"last_weekly_close_{idx}", axis=1)

    for idx in range(config.model.monthly_lookback):
        raw_data[f"last_monthly_close_{idx}"] = raw_data.apply(fill_close, time_col=f"last_monthly_close_time_{idx}",
                                                               pre_interval_mins=0, post_interval_mins=interval_mins, axis=1)
        raw_data[f"last_monthly_close_{idx}"] = raw_data[f"last_monthly_close_{idx}"].ffill()
        raw_data[f"ret_from_last_monthly_close_{idx}"] = raw_data.apply(compute_ret, base_col=f"last_monthly_close_{idx}", axis=1)

    raw_data["next_new_york_close"] = raw_data.apply(fill_close, time_col="new_york_close_time",
                                                     pre_interval_mins=0, post_interval_mins=interval_mins, axis=1)    
    raw_data["next_weekly_close"] = raw_data.apply(fill_close, time_col="weekly_close_time",
                                                   pre_interval_mins=0, post_interval_mins=interval_mins, axis=1)    
    raw_data["next_monthly_close"] = raw_data.apply(fill_close, time_col="monthly_close_time",
                                                    pre_interval_mins=0, post_interval_mins=interval_mins, axis=1)
    raw_data["ret_to_next_new_york_close"] = raw_data.apply(compute_ret, base_col="next_new_york_close", axis=1)
    raw_data["ret_to_next_weekly_close"] = raw_data.apply(compute_ret, base_col="next_weekly_close", axis=1)
    raw_data["ret_to_next_monthly_close"] = raw_data.apply(compute_ret, base_col="next_monthly_close", axis=1)

    raw_data["last_option_expiration_close"] = raw_data.apply(fill_close, time_col="last_option_expiration_time",
                                                              pre_interval_mins=0, post_interval_mins=interval_mins, axis=1)    
    raw_data["last_option_expiration_close"] = raw_data.last_option_expiration_close.ffill()

                                                        
    raw_data["around_new_york_open_cum_dv"] = raw_data.apply(fill_cum_dv, time_col="new_york_last_open_time",
                                                             pre_interval_mins=interval_mins*2, post_interval_mins=interval_mins*2, axis=1)
    raw_data["around_new_york_open_cum_volume"] = raw_data.apply(fill_cum_volume, time_col="new_york_last_open_time",
                                                                 pre_interval_mins=interval_mins*2, post_interval_mins=interval_mins*2, axis=1)
    rol = raw_data.around_new_york_open_cum_dv.rolling(window=2)
    raw_data["vwap_around_new_york_open"] = rol.apply(vwap_around, args=("around_new_york_open_cum_dv","around_new_york_open_cum_volume"), raw=False)
    raw_data["vwap_around_new_york_open"] = raw_data.vwap_around_new_york_open.ffill()
    raw_data["ret_from_vwap_around_new_york_open"] = raw_data.apply(compute_ret, base_col="vwap_around_new_york_open", axis=1)
    raw_data = raw_data.drop(columns=["around_new_york_open_cum_dv", "around_new_york_open_cum_volume"])

    raw_data["pre_new_york_close_cum_dv"] = raw_data.apply(fill_cum_dv, time_col="new_york_last_close_time",
                                                           pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)
    raw_data["pre_new_york_close_cum_volume"] = raw_data.apply(fill_cum_volume, time_col="new_york_last_close_time",
                                                               pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)
    rol = raw_data.pre_new_york_close_cum_dv.rolling(window=2)
    raw_data["vwap_pre_new_york_close"] = rol.apply(vwap_around, args=("pre_new_york_close_cum_dv","pre_new_york_close_cum_volume"), raw=False)
    raw_data["vwap_pre_new_york_close"] = raw_data.vwap_pre_new_york_close.ffill()
    raw_data["ret_from_vwap_pre_new_york_close"] = raw_data.apply(compute_ret, base_col="vwap_pre_new_york_close", axis=1)
    raw_data = raw_data.drop(columns=["pre_new_york_close_cum_dv", "pre_new_york_close_cum_volume"])

    raw_data["post_new_york_close_cum_dv"] = raw_data.apply(fill_cum_dv, time_col="new_york_last_close_time",
                                                              pre_interval_mins=0, post_interval_mins=interval_mins*3, axis=1)
    raw_data["post_new_york_close_cum_volume"] = raw_data.apply(fill_cum_volume, time_col="new_york_last_close_time",
                                                                pre_interval_mins=0, post_interval_mins=interval_mins*3, axis=1)
    rol = raw_data.post_new_york_close_cum_dv.rolling(window=2)
    raw_data["vwap_post_new_york_close"] = rol.apply(vwap_around, args=("post_new_york_close_cum_dv","post_new_york_close_cum_volume"), raw=False)
    raw_data["vwap_post_new_york_close"] = raw_data.vwap_post_new_york_close.ffill()
    raw_data["ret_from_vwap_post_new_york_close"] = raw_data.apply(compute_ret, base_col="vwap_post_new_york_close", axis=1)
    raw_data = raw_data.drop(columns=["post_new_york_close_cum_dv", "post_new_york_close_cum_volume"])


    raw_data["around_new_york_close_cum_dv"] = raw_data.apply(fill_cum_dv, time_col="new_york_last_close_time",
                                                              pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3, axis=1)
    raw_data["around_new_york_close_cum_volume"] = raw_data.apply(fill_cum_volume, time_col="new_york_last_close_time",
                                                                  pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3, axis=1)
    rol = raw_data.around_new_york_close_cum_dv.rolling(window=2)
    raw_data["vwap_around_new_york_close"] = rol.apply(vwap_around, args=("around_new_york_close_cum_dv","around_new_york_close_cum_volume"), raw=False)
    raw_data["vwap_around_new_york_close"] = raw_data.vwap_around_new_york_close.ffill()
    raw_data["ret_from_vwap_around_new_york_close"] = raw_data.apply(compute_ret, base_col="vwap_around_new_york_close", axis=1)
    raw_data = raw_data.drop(columns=["around_new_york_close_cum_dv", "around_new_york_close_cum_volume"])

    raw_data["pre_london_open_cum_dv"] = raw_data.apply(fill_cum_dv, time_col="london_last_open_time",
                                                        pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)
    raw_data["pre_london_open_cum_volume"] = raw_data.apply(fill_cum_volume, time_col="london_last_open_time",
                                                            pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)
    rol = raw_data.pre_london_open_cum_dv.rolling(window=2)
    raw_data["vwap_pre_london_open"] = rol.apply(vwap_around, args=("pre_london_open_cum_dv","pre_london_open_cum_volume"), raw=False)
    raw_data["vwap_pre_london_open"] = raw_data.vwap_pre_london_open.ffill()
    raw_data["ret_from_vwap_pre_london_open"] = raw_data.apply(compute_ret, base_col="vwap_pre_london_open", axis=1)

    raw_data["post_london_open_cum_dv"] = raw_data.apply(fill_cum_dv, time_col="london_last_open_time",
                                                         pre_interval_mins=0, post_interval_mins=interval_mins*3, axis=1)
    raw_data["post_london_open_cum_volume"] = raw_data.apply(fill_cum_volume, time_col="london_last_open_time",
                                                             pre_interval_mins=0, post_interval_mins=interval_mins*3, axis=1)
    rol = raw_data.post_london_open_cum_dv.rolling(window=2)
    raw_data["vwap_post_london_open"] = rol.apply(vwap_around, args=("post_london_open_cum_dv","post_london_open_cum_volume"), raw=False)
    raw_data["vwap_post_london_open"] = raw_data.vwap_post_london_open.ffill()
    raw_data["ret_from_vwap_post_london_open"] = raw_data.apply(compute_ret, base_col="vwap_post_london_open", axis=1)

    raw_data["around_london_open_cum_dv"] = raw_data.apply(fill_cum_dv, time_col="london_last_open_time",
                                                           pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3, axis=1)
    raw_data["around_london_open_cum_volume"] = raw_data.apply(fill_cum_volume, time_col="london_last_open_time",
                                                               pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3, axis=1)
    rol = raw_data.around_london_open_cum_dv.rolling(window=2)
    raw_data["vwap_around_london_open"] = rol.apply(vwap_around, args=("around_london_open_cum_dv","around_london_open_cum_volume"), raw=False)
    raw_data["vwap_around_london_open"] = raw_data.vwap_around_london_open.ffill()
    raw_data["ret_from_vwap_around_london_open"] = raw_data.apply(compute_ret, base_col="vwap_around_london_open", axis=1)
    #raw_data = raw_data.drop(columns=["around_london_open_cum_dv", "around_london_open_cum_volume"])

    raw_data["pre_london_close_cum_dv"] = raw_data.apply(fill_cum_dv, time_col="london_last_close_time",
                                                         pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)
    raw_data["pre_london_close_cum_volume"] = raw_data.apply(fill_cum_volume, time_col="london_last_close_time",
                                                             pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)
    rol = raw_data.pre_london_close_cum_dv.rolling(window=2)
    raw_data["vwap_pre_london_close"] = rol.apply(vwap_around, args=("pre_london_close_cum_dv","pre_london_close_cum_volume"), raw=False)
    raw_data["vwap_pre_london_close"] = raw_data.vwap_pre_london_close.ffill()
    raw_data["ret_from_vwap_pre_london_close"] = raw_data.apply(compute_ret, base_col="vwap_pre_london_close", axis=1)

    raw_data["post_london_close_cum_dv"] = raw_data.apply(fill_cum_dv, time_col="london_last_close_time",
                                                          pre_interval_mins=0, post_interval_mins=interval_mins*3, axis=1)
    raw_data["post_london_close_cum_volume"] = raw_data.apply(fill_cum_volume, time_col="london_last_close_time",
                                                              pre_interval_mins=0, post_interval_mins=interval_mins*3, axis=1)
    rol = raw_data.post_london_close_cum_dv.rolling(window=2)
    raw_data["vwap_post_london_close"] = rol.apply(vwap_around, args=("post_london_close_cum_dv","post_london_close_cum_volume"), raw=False)
    raw_data["vwap_post_london_close"] = raw_data.vwap_post_london_open.ffill()
    raw_data["ret_from_vwap_post_london_close"] = raw_data.apply(compute_ret, base_col="vwap_post_london_close", axis=1)

    raw_data["around_london_close_cum_dv"] = raw_data.apply(fill_cum_dv, time_col="london_last_close_time",
                                                            pre_interval_mins=interval_mins*2, post_interval_mins=interval_mins*2, axis=1)
    raw_data["around_london_close_cum_volume"] = raw_data.apply(fill_cum_volume, time_col="london_last_close_time",
                                                                pre_interval_mins=interval_mins*2, post_interval_mins=interval_mins*2, axis=1)
    rol = raw_data.around_london_close_cum_dv.rolling(window=2)
    raw_data["vwap_around_london_close"] = rol.apply(vwap_around, args=("around_london_close_cum_dv","around_london_close_cum_volume"), raw=False)
    raw_data["vwap_around_london_close"] = raw_data.vwap_around_london_close.ffill()
    raw_data["ret_from_vwap_around_london_close"] = raw_data.apply(compute_ret, base_col="vwap_around_london_close", axis=1)
    raw_data = raw_data.drop(columns=["around_london_close_cum_dv", "around_london_close_cum_volume"])

    raw_data["ret_from_bb_high_5_2"] = raw_data.apply(compute_ret, base_col="bb_high_5_2", axis=1)
    raw_data["ret_from_bb_high_5_3"] = raw_data.apply(compute_ret, base_col="bb_high_5_3", axis=1)
    raw_data["ret_from_bb_high_10_2"] = raw_data.apply(compute_ret, base_col="bb_high_10_2", axis=1)
    raw_data["ret_from_bb_high_10_3"] = raw_data.apply(compute_ret, base_col="bb_high_10_3", axis=1)
    raw_data["ret_from_bb_high_20_2"] = raw_data.apply(compute_ret, base_col="bb_high_20_2", axis=1)
    raw_data["ret_from_bb_high_20_3"] = raw_data.apply(compute_ret, base_col="bb_high_20_3", axis=1)
    raw_data["ret_from_bb_high_50_2"] = raw_data.apply(compute_ret, base_col="bb_high_50_2", axis=1)
    raw_data["ret_from_bb_high_50_3"] = raw_data.apply(compute_ret, base_col="bb_high_50_3", axis=1)
    raw_data["ret_from_bb_high_100_2"] = raw_data.apply(compute_ret, base_col="bb_high_100_2", axis=1)
    raw_data["ret_from_bb_high_100_3"] = raw_data.apply(compute_ret, base_col="bb_high_100_3", axis=1)
    raw_data["ret_from_bb_high_200_2"] = raw_data.apply(compute_ret, base_col="bb_high_200_2", axis=1)
    raw_data["ret_from_bb_high_200_3"] = raw_data.apply(compute_ret, base_col="bb_high_200_3", axis=1)

    raw_data["ret_from_bb_low_5_2"] = raw_data.apply(compute_ret, base_col="bb_low_5_2", axis=1)
    raw_data["ret_from_bb_low_5_3"] = raw_data.apply(compute_ret, base_col="bb_low_5_3", axis=1)
    raw_data["ret_from_bb_low_10_2"] = raw_data.apply(compute_ret, base_col="bb_low_10_2", axis=1)
    raw_data["ret_from_bb_low_10_3"] = raw_data.apply(compute_ret, base_col="bb_low_10_3", axis=1)
    raw_data["ret_from_bb_low_20_2"] = raw_data.apply(compute_ret, base_col="bb_low_20_2", axis=1)
    raw_data["ret_from_bb_low_20_3"] = raw_data.apply(compute_ret, base_col="bb_low_20_3", axis=1)
    raw_data["ret_from_bb_low_50_2"] = raw_data.apply(compute_ret, base_col="bb_low_50_2", axis=1)
    raw_data["ret_from_bb_low_50_3"] = raw_data.apply(compute_ret, base_col="bb_low_50_3", axis=1)
    raw_data["ret_from_bb_low_100_2"] = raw_data.apply(compute_ret, base_col="bb_low_100_2", axis=1)
    raw_data["ret_from_bb_low_100_3"] = raw_data.apply(compute_ret, base_col="bb_low_100_3", axis=1)
    raw_data["ret_from_bb_low_200_2"] = raw_data.apply(compute_ret, base_col="bb_low_200_2", axis=1)
    raw_data["ret_from_bb_low_200_3"] = raw_data.apply(compute_ret, base_col="bb_low_200_3", axis=1)

    raw_data["ret_from_sma_5"] = raw_data.apply(compute_ret, base_col="sma_5", axis=1)
    raw_data["ret_from_sma_10"] = raw_data.apply(compute_ret, base_col="sma_10", axis=1)
    raw_data["ret_from_sma_20"] = raw_data.apply(compute_ret, base_col="sma_20", axis=1)
    raw_data["ret_from_sma_50"] = raw_data.apply(compute_ret, base_col="sma_50", axis=1)
    raw_data["ret_from_sma_100"] = raw_data.apply(compute_ret, base_col="sma_100", axis=1)
    raw_data["ret_from_sma_200"] = raw_data.apply(compute_ret, base_col="sma_200", axis=1)

    if add_daily_rolling_features:
        raw_data["ret_from_bb_high_5d_2"] = raw_data.apply(compute_ret, base_col="bb_high_5d_2", axis=1)
        raw_data["ret_from_bb_high_5d_3"] = raw_data.apply(compute_ret, base_col="bb_high_5d_3", axis=1)
        raw_data["ret_from_bb_high_10d_2"] = raw_data.apply(compute_ret, base_col="bb_high_10d_2", axis=1)
        raw_data["ret_from_bb_high_10d_3"] = raw_data.apply(compute_ret, base_col="bb_high_10d_3", axis=1)
        raw_data["ret_from_bb_high_20d_2"] = raw_data.apply(compute_ret, base_col="bb_high_20d_2", axis=1)
        raw_data["ret_from_bb_high_20d_3"] = raw_data.apply(compute_ret, base_col="bb_high_20d_3", axis=1)
        raw_data["ret_from_bb_high_50d_2"] = raw_data.apply(compute_ret, base_col="bb_high_50d_2", axis=1)
        raw_data["ret_from_bb_high_50d_3"] = raw_data.apply(compute_ret, base_col="bb_high_50d_3", axis=1)
        raw_data["ret_from_bb_high_100d_2"] = raw_data.apply(compute_ret, base_col="bb_high_100d_2", axis=1)
        raw_data["ret_from_bb_high_100d_3"] = raw_data.apply(compute_ret, base_col="bb_high_100d_3", axis=1)
        raw_data["ret_from_bb_high_200d_2"] = raw_data.apply(compute_ret, base_col="bb_high_200d_2", axis=1)
        raw_data["ret_from_bb_high_200d_3"] = raw_data.apply(compute_ret, base_col="bb_high_200d_3", axis=1)

        raw_data["ret_from_bb_low_5d_2"] = raw_data.apply(compute_ret, base_col="bb_low_5d_2", axis=1)
        raw_data["ret_from_bb_low_5d_3"] = raw_data.apply(compute_ret, base_col="bb_low_5d_3", axis=1)
        raw_data["ret_from_bb_low_10d_2"] = raw_data.apply(compute_ret, base_col="bb_low_10d_2", axis=1)
        raw_data["ret_from_bb_low_10d_3"] = raw_data.apply(compute_ret, base_col="bb_low_10d_3", axis=1)
        raw_data["ret_from_bb_low_20d_2"] = raw_data.apply(compute_ret, base_col="bb_low_20d_2", axis=1)
        raw_data["ret_from_bb_low_20d_3"] = raw_data.apply(compute_ret, base_col="bb_low_20d_3", axis=1)
        raw_data["ret_from_bb_low_50d_2"] = raw_data.apply(compute_ret, base_col="bb_low_50d_2", axis=1)
        raw_data["ret_from_bb_low_50d_3"] = raw_data.apply(compute_ret, base_col="bb_low_50d_3", axis=1)
        raw_data["ret_from_bb_low_100d_2"] = raw_data.apply(compute_ret, base_col="bb_low_100d_2", axis=1)
        raw_data["ret_from_bb_low_100d_3"] = raw_data.apply(compute_ret, base_col="bb_low_100d_3", axis=1)
        raw_data["ret_from_bb_low_200d_2"] = raw_data.apply(compute_ret, base_col="bb_low_200d_2", axis=1)
        raw_data["ret_from_bb_low_200d_3"] = raw_data.apply(compute_ret, base_col="bb_low_200d_3", axis=1)

        raw_data["ret_from_sma_5d"] = raw_data.apply(compute_ret, base_col="sma_5d", axis=1)
        raw_data["ret_from_sma_10d"] = raw_data.apply(compute_ret, base_col="sma_10d", axis=1)
        raw_data["ret_from_sma_20d"] = raw_data.apply(compute_ret, base_col="sma_20d", axis=1)
        raw_data["ret_from_sma_50d"] = raw_data.apply(compute_ret, base_col="sma_50d", axis=1)
        raw_data["ret_from_sma_100d"] = raw_data.apply(compute_ret, base_col="sma_100d", axis=1)
        raw_data["ret_from_sma_200d"] = raw_data.apply(compute_ret, base_col="sma_200d", axis=1)

    #logging.error(f"sampled_raw:{raw_data.iloc[-10:]}")
    return raw_data

class Preprocessor:
    def __init__(self, ticker, since, until):
        self.since = since
        self.until = until
        self.ticker = ticker

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index("Time")
        df = df[self.since : self.until]
        df = df.rename(
            columns={
                "Volume": "volume",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
            }
        )
        df["dv"] = df["close"] * df["volume"]
        df = df.resample("30Min").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "dv": "sum",
            }
        )
        df["ticker"] = self.ticker
        df["time"] = df.index
        df["cum_volume"] = df.volume.cumsum()
        df["cum_dv"] = df.dv.cumsum()
        df_pct_back = df[["close", "volume", "dv"]].pct_change(periods=1)
        #df_pct_forward = df[["close", "volume", "dv"]].pct_change(periods=-1)
        #df = df.join(df_pct_back, rsuffix="_back").join(df_pct_forward, rsuffix="_fwd")
        df = df.join(df_pct_back, rsuffix="_back")
        df["month"] = df.time.dt.month  # categories have be strings
        df["year"] = df.time.dt.year  # categories have be strings
        df["hour_of_day"] = df.time.apply(lambda x: x.hour)
        df["day_of_week"] = df.time.apply(lambda x: x.dayofweek)
        df["day_of_month"] = df.time.apply(lambda x: x.day)
        df = df.dropna()
        return df
