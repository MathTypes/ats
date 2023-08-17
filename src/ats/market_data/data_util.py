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



def get_time(x, close_col):
    if x[close_col] == x["close_back_cumsum"]:
        return x["timestamp"]
    else:
        return None
    
@profile_util.profile
#@njit(parallel=True)
def ticker_transform(raw_data, interval_minutes, base_price=500, add_daily_rolling_features=True):
    ewm = raw_data["close"].ewm(halflife=HALFLIFE_WINSORISE)
    means = ewm.mean()
    stds = ewm.std()
    del ewm
    raw_data["close"] = np.minimum(raw_data["close"], means + VOL_THRESHOLD * stds)
    raw_data["close"] = np.maximum(raw_data["close"], means - VOL_THRESHOLD * stds)
    raw_data["cum_volume"] = raw_data.volume.cumsum()
    raw_data["cum_dv"] = raw_data.dv.cumsum()
    squash_factor = 4
    #raw_data['close_back'] = squash_factor * np.tanh((np.log(raw_data.close+base_price) - np.log(raw_data.close.shift(1)+base_price))/squash_factor)
    raw_data['close_back'] = np.log(raw_data.close+base_price) - np.log(raw_data.close.shift(1)+base_price)
    raw_data['high_back'] = squash_factor * np.tanh((np.log(raw_data.high+base_price) - np.log(raw_data.high.shift(1)+base_price))/squash_factor)
    raw_data['open_back'] = squash_factor * np.tanh((np.log(raw_data.open+base_price) - np.log(raw_data.open.shift(1)+base_price))/squash_factor)
    raw_data['low_back'] = squash_factor * np.tanh((np.log(raw_data.low+base_price) - np.log(raw_data.low.shift(1)+base_price))/squash_factor)
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
        raw_data['close_rolling_5d_max'] = raw_data.close_back_cumsum.rolling(5*interval_per_day).max()
        raw_data["close_high_5d_ff"] = raw_data["close_rolling_5d_max"].ffill()
        raw_data["time_high_5d_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_high_5d_ff"), axis=1)
        raw_data["time_high_5d_ff"]  = raw_data["time_high_5d_ff"].ffill()

        raw_data['close_rolling_5d_min'] = raw_data.close_back_cumsum.rolling(5*interval_per_day).min()
        raw_data["close_low_5d_ff"] = raw_data["close_rolling_5d_min"].ffill()
        raw_data["time_low_5d_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_low_5d_ff"), axis=1)
        raw_data["time_low_5d_ff"]  = raw_data["time_low_5d_ff"].ffill()

        raw_data['close_rolling_11d_max'] = raw_data.close_back_cumsum.rolling(11*interval_per_day).max()
        raw_data["close_high_11d_ff"] = raw_data["close_rolling_11d_max"].ffill()
        raw_data["time_high_11d_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_high_11d_ff"), axis=1)
        raw_data["time_high_11d_ff"]  = raw_data["time_high_11d_ff"].ffill()

        raw_data['close_rolling_11d_min'] = raw_data.close_back_cumsum.rolling(11*interval_per_day).min()
        raw_data["close_low_11d_ff"] = raw_data["close_rolling_11d_min"].ffill()
        raw_data["time_low_11d_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_low_11d_ff"), axis=1)
        raw_data["time_low_11d_ff"]  = raw_data["time_low_11d_ff"].ffill()
 
        raw_data['close_rolling_21d_max'] = raw_data.close_back_cumsum.rolling(21*interval_per_day).max()
        raw_data["close_high_21d_ff"] = raw_data["close_rolling_21d_max"].ffill()
        raw_data["time_high_21d_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_high_21d_ff"), axis=1)
        raw_data["time_high_21d_ff"]  = raw_data["time_high_21d_ff"].ffill()

        raw_data['close_rolling_21d_min'] = raw_data.close_back_cumsum.rolling(21*interval_per_day).min()
        raw_data["close_low_21d_ff"] = raw_data["close_rolling_21d_min"].ffill()
        raw_data["time_low_21d_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_low_21d_ff"), axis=1)
        raw_data["time_low_21d_ff"]  = raw_data["time_low_21d_ff"].ffill()

        raw_data['close_rolling_51d_max'] = raw_data.close_back_cumsum.rolling(51*interval_per_day).max()
        raw_data["close_high_51d_ff"] = raw_data["close_rolling_51d_max"].ffill()
        raw_data["time_high_51d_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_high_51d_ff"), axis=1)
        raw_data["time_high_51d_ff"]  = raw_data["time_high_51d_ff"].ffill()

        raw_data['close_rolling_51d_min'] = raw_data.close_back_cumsum.rolling(51*interval_per_day).min()
        raw_data["close_low_51d_ff"] = raw_data["close_rolling_51d_min"].ffill()
        raw_data["time_low_51d_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_low_51d_ff"), axis=1)
        raw_data["time_low_51d_ff"]  = raw_data["time_low_51d_ff"].ffill()

        raw_data['close_rolling_201d_max'] = raw_data.close_back_cumsum.rolling(201*interval_per_day).max()
        raw_data["close_high_201d_ff"] = raw_data["close_rolling_201d_max"].ffill()
        raw_data["time_high_201d_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_high_201d_ff"), axis=1)
        raw_data["time_high_201d_ff"]  = raw_data["time_high_201d_ff"].ffill()

        raw_data['close_rolling_201d_min'] = raw_data.close_back_cumsum.rolling(51*interval_per_day).min()
        raw_data["close_low_201d_ff"] = raw_data["close_rolling_201d_min"].ffill()
        raw_data["time_low_201d_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_low_201d_ff"), axis=1)
        raw_data["time_low_201d_ff"]  = raw_data["time_low_201d_ff"].ffill()

    raw_data['close_rolling_5_max'] = raw_data.close_back_cumsum.rolling(5).max()
    raw_data["close_high_5_ff"] = raw_data["close_rolling_5_max"].ffill()
    raw_data["time_high_5_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_high_5_ff"), axis=1)
    raw_data["time_high_5_ff"]  = raw_data["time_high_5_ff"].ffill()

    raw_data['close_rolling_5_min'] = raw_data.close_back_cumsum.rolling(5).min()
    raw_data["close_low_5_ff"] = raw_data["close_rolling_5_min"].ffill()
    raw_data["time_low_5_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_low_5_ff"), axis=1)
    raw_data["time_low_5_ff"]  = raw_data["time_low_5_ff"].ffill()

    raw_data['close_rolling_11_max'] = raw_data.close_back_cumsum.rolling(11).max()
    raw_data["close_high_11_ff"] = raw_data["close_rolling_11_max"].ffill()
    raw_data["time_high_11_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_high_11_ff"), axis=1)
    raw_data["time_high_11_ff"]  = raw_data["time_high_11_ff"].ffill()

    raw_data['close_rolling_11_min'] = raw_data.close_back_cumsum.rolling(11).min()
    raw_data["close_low_11_ff"] = raw_data["close_rolling_11_min"].ffill()
    raw_data["time_low_11_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_low_11_ff"), axis=1)
    raw_data["time_low_11_ff"]  = raw_data["time_low_11_ff"].ffill()
 
    raw_data['close_rolling_21_max'] = raw_data.close_back_cumsum.rolling(21).max()
    raw_data["close_high_21_ff"] = raw_data["close_rolling_21_max"].ffill()
    raw_data["time_high_21_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_high_21_ff"), axis=1)
    raw_data["time_high_21_ff"]  = raw_data["time_high_21_ff"].ffill()

    raw_data['close_rolling_21_min'] = raw_data.close_back_cumsum.rolling(21).min()
    raw_data["close_low_21_ff"] = raw_data["close_rolling_21_min"].ffill()
    raw_data["time_low_21_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_low_21_ff"), axis=1)
    raw_data["time_low_21_ff"]  = raw_data["time_low_21_ff"].ffill()

    raw_data['close_rolling_51_max'] = raw_data.close_back_cumsum.rolling(51).max()
    raw_data["close_high_51_ff"] = raw_data["close_rolling_51_max"].ffill()
    raw_data["time_high_51_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_high_51_ff"), axis=1)
    raw_data["time_high_51_ff"]  = raw_data["time_high_51_ff"].ffill()

    raw_data['close_rolling_51_min'] = raw_data.close_back_cumsum.rolling(51).min()
    raw_data["close_low_51_ff"] = raw_data["close_rolling_51_min"].ffill()
    raw_data["time_low_51_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_low_51_ff"), axis=1)
    raw_data["time_low_51_ff"]  = raw_data["time_low_51_ff"].ffill()

    raw_data['close_rolling_201_max'] = raw_data.close_back_cumsum.rolling(201).max()
    raw_data["close_high_201_ff"] = raw_data["close_rolling_201_max"].ffill()
    raw_data["time_high_201_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_high_201_ff"), axis=1)
    raw_data["time_high_201_ff"]  = raw_data["time_high_201_ff"].ffill()

    raw_data['close_rolling_201_min'] = raw_data.close_back_cumsum.rolling(51).min()
    raw_data["close_low_201_ff"] = raw_data["close_rolling_201_min"].ffill()
    raw_data["time_low_201_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_low_201_ff"), axis=1)
    raw_data["time_low_201_ff"]  = raw_data["time_low_201_ff"].ffill()
    del close_back_cumsum

    
    # Compute RSI
    raw_data["rsi"] = ta.momentum.RSIIndicator(close=raw_data["close"]).rsi()

    # Compute MACD
    macd = ta.trend.MACD(close=raw_data["close"])
    raw_data["macd"] = macd.macd()
    raw_data["macd_signal"] = macd.macd_signal()

    # Compute Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=raw_data["close"])
    raw_data["bb_high"] = bollinger.bollinger_hband()
    raw_data["bb_low"] = bollinger.bollinger_lband()

    # Compute Moving Averages
    if add_daily_rolling_features:
        interval_per_day = int(23 * 60 / interval_minutes)
        raw_data["sma_50d"] = ta.trend.SMAIndicator(
            close=raw_data["close"], window=50*interval_per_day
        ).sma_indicator()
        raw_data["sma_200d"] = ta.trend.SMAIndicator(
            close=raw_data["close"], window=200*interval_per_day
        ).sma_indicator()

    raw_data["sma_50"] = ta.trend.SMAIndicator(
        close=raw_data["close"], window=50
    ).sma_indicator()
    raw_data["sma_200"] = ta.trend.SMAIndicator(
        close=raw_data["close"], window=200
    ).sma_indicator()
    return raw_data

def time_diff(row, base_col, diff_col):
    if pd.isna(row[diff_col]):
        return np.nan
    else:
        return (row[diff_col] - row[base_col])

def ret_diff(row, base_col, diff_col):
    if pd.isna(row[base_col]):
        return np.nan
    return (row[diff_col] - row[base_col])

def compute_ret(row, base_col, base_price=500):
    if pd.isna(row[base_col]):
        return np.nan
    return np.log(row["close"]+base_price) - np.log(row[base_col]+base_price)
    
def compute_vwap(row, dv_col, volume_col, base_price=500):
    #logging.error(f"volume_col:{volume_col}")
    if pd.isna(row[dv_col]) or pd.isna(row[volume_col]):
        return np.nan
    else:
        #logging.error(f"row['cum_volume']:{row['cum_volume']}")
        #logging.error(f"row['cum_dv']:{row['cum_dv']}")
        #logging.error(f"row[{volume_col}]:{row[volume_col]}")
        #logging.error(f"row[{dv_col}]:{row[dv_col]}")
        #logging.error(f"row[close]:{row['close']}")
        if row["cum_volume"]>row[volume_col]:
            if row["cum_dv"] == row[dv_col]:
                #logging.error(f"no cum_dv change")
                return 0
            else:
                vwap_price = (row["cum_dv"] - row[dv_col])/(row["cum_volume"]-row[volume_col])
                #logging.error(f"vwap_price:{vwap_price}, close:{row['close']}")
                return np.log(row["close"]+base_price) - np.log(vwap_price+base_price)
        else:
            #logging.error(f"cum volume does not change")
            return 0

def fill_cum_dv(row, time_col, interval_mins=30):
    before_mins=interval_mins*60*0.9+1
    after_mins=interval_mins*60*0.9+1
    if pd.isna(row[time_col]):
        return np.nan
    else:
        if ((row[time_col] > row["timestamp"]-before_mins) and
            (row[time_col] < row["timestamp"]+after_mins)):
            return row["cum_dv"]
        else:
            return np.nan

def fill_cum_volume(row, time_col, interval_mins=30):
    before_mins=interval_mins*60*.9+1
    after_mins=interval_mins*60*.9+1
    if pd.isna(row[time_col]):
        return np.nan
    else:
        if ((row[time_col] > row["timestamp"]-before_mins) and
            (row[time_col] < row["timestamp"]+after_mins)):
            return row["cum_volume"]
        else:
            return np.nan

#@profile_util.profile
#@njit(parallel=True)
def add_group_features(interval_minutes, raw_data, add_daily_rolling_features=True):
    for column in [
        "close_back",
            "high_back", "low_back", "open_back",
        "volume_back",
        "dv_back",
        #"close_fwd",
        #"volume_fwd",
        #"dv_fwd",
        "cum_volume",
        "cum_dv",
        "close_back_cumsum",
        "volume_back_cumsum",
        "close_high_5_ff",
        "time_high_5_ff",
        "close_low_5_ff",
        "time_low_5_ff",
        "close_high_11_ff",
        "time_high_11_ff",
        "close_low_11_ff",
        "time_low_11_ff",
        "close_high_21_ff",
        "time_high_21_ff",
        "close_low_21_ff",
        "time_low_21_ff",
        "close_high_51_ff",
        "time_high_51_ff",
        "close_low_51_ff",
        "time_low_51_ff",
        "close_high_201_ff",
        "time_high_201_ff",
        "close_low_201_ff",
        "time_low_201_ff",
        "close_high_5d_ff",
        "time_high_5d_ff",
        "close_low_5d_ff",
        "time_low_5d_ff",
        "close_high_11d_ff",
        "time_high_11d_ff",
        "close_low_11d_ff",
        "time_low_11d_ff",
        "close_high_21d_ff",
        "time_high_21d_ff",
        "close_low_21d_ff",
        "time_low_21d_ff",
        "close_high_51d_ff",
        "time_high_51d_ff",
        "close_low_51d_ff",
        "time_low_51d_ff",
        "close_high_201d_ff",
        "time_high_201d_ff",
        "close_low_201d_ff",
        "time_low_201d_ff",
        'close_rolling_5d_max', 'close_rolling_201_min', 'sma_50d', 'sma_200d',
        "rsi",
        "macd",
        "macd_signal",
        "bb_high",
        "bb_low",
        "sma_50",
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
    new_features = raw_data.groupby(["ticker"], group_keys=False)[[
        "volume", "dv", "close", "high", "low", "open", "timestamp"]].apply(ticker_transform,
                                                                            interval_minutes=interval_minutes,
                                                                            add_daily_rolling_features=add_daily_rolling_features)
    new_features = new_features.drop(columns=["volume", "dv", "close", "high", "low", "open", "timestamp"])
    raw_data = raw_data.join(new_features)
    raw_data.reset_index(drop=True, inplace=True)
    #del new_features

    raw_data["daily_returns"] = calc_returns(raw_data["close"])
    raw_data["daily_vol"] = calc_daily_vol(raw_data["daily_returns"])
    raw_data["daily_skew"] = calc_skew(raw_data["close"])
    raw_data["daily_kurt"] = calc_kurt(raw_data["close"])
    trend_combinations = [(8, 24), (16, 48), (32, 96)]
    for short_window, long_window in trend_combinations:
        raw_data[f"macd_{short_window}_{long_window}"] = MACDStrategy.calc_signal(
            raw_data["close"], short_window, long_window
        )
    if add_daily_rolling_features:
        for short_window, long_window in trend_combinations:
            raw_data[f"macd_{short_window}_{long_window}_day"] = MACDStrategy.calc_signal(
                raw_data["close"], short_window, long_window, 46
            )

    return raw_data


@profile_util.profile
def add_example_level_features(cal, macro_data_builder, raw_data, add_daily_rolling_features):
    raw_data = pd.DataFrame(raw_data)
    return add_example_level_features_df(cal, macro_data_builder, raw_data, add_daily_rolling_features)

@profile_util.profile
#@njit(parallel=True)
def add_example_level_features_df(cal, macro_data_builder, raw_data,
                                  add_daily_rolling_features):
    new_york_cal = mcal.get_calendar("NYSE")
    lse_cal = mcal.get_calendar("LSE")
    raw_data["week_of_year"] = raw_data["time"].apply(lambda x: x.isocalendar()[1])
    raw_data["month_of_year"] = raw_data["time"].apply(lambda x: x.month)

    lse_cal = mcal.get_calendar("LSE")
    raw_data["weekly_close_time"] = raw_data.timestamp.apply(
        market_time.compute_weekly_close_time, cal=cal
    )
    raw_data["monthly_close_time"] = raw_data.timestamp.apply(
        market_time.compute_monthly_close_time, cal=cal
    )
    raw_data["option_expiration_time"] = raw_data.timestamp.apply(
        market_time.compute_option_expiration_time, cal=cal
    )

    if macro_data_builder.add_macro_event:
        raw_data["last_macro_event_time_imp1"] = raw_data.timestamp.apply(
            market_time.compute_last_macro_event_time, cal=cal, mdb=macro_data_builder, imp=1
        )
        raw_data["next_macro_event_time_imp1"] = raw_data.timestamp.apply(
            market_time.compute_next_macro_event_time, cal=cal, mdb=macro_data_builder, imp=1
        )
        raw_data["last_macro_event_time_imp2"] = raw_data.timestamp.apply(
            market_time.compute_last_macro_event_time, cal=cal, mdb=macro_data_builder, imp=2
        )
        raw_data["next_macro_event_time_imp2"] = raw_data.timestamp.apply(
            market_time.compute_next_macro_event_time, cal=cal, mdb=macro_data_builder, imp=1
        )
        raw_data["last_macro_event_time_imp3"] = raw_data.timestamp.apply(
            market_time.compute_last_macro_event_time, cal=cal, mdb=macro_data_builder, imp=3
        )
        raw_data["next_macro_event_time_imp3"] = raw_data.timestamp.apply(
            market_time.compute_next_macro_event_time, cal=cal, mdb=macro_data_builder, imp=3
        )

    raw_data["new_york_open_time"] = raw_data.timestamp.apply(
        market_time.compute_open_time, cal=new_york_cal
    )
    raw_data["new_york_last_open_time"] = raw_data.timestamp.apply(
        market_time.compute_last_open_time, cal=new_york_cal
    )

    raw_data["new_york_close_time"] = raw_data.timestamp.apply(
        market_time.compute_close_time, cal=new_york_cal
    )
    raw_data["london_open_time"] = raw_data.timestamp.apply(
        market_time.compute_open_time, cal=lse_cal
    )
    raw_data["london_last_open_time"] = raw_data.timestamp.apply(
        market_time.compute_last_open_time, cal=lse_cal
    )
    raw_data["london_close_time"] = raw_data.timestamp.apply(
        market_time.compute_close_time, cal=lse_cal
    )
    raw_data["time_to_new_york_open"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="new_york_open_time"
    )
    raw_data["time_to_new_york_last_open"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="new_york_last_open_time"
    )
    raw_data["time_to_new_york_close"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="new_york_close_time"
    )
    raw_data["time_to_london_open"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="london_open_time"
    )
    raw_data["time_to_london_last_open"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="london_last_open_time"
    )
    raw_data["time_to_london_close"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="london_close_time"
    )
    raw_data["month"] = raw_data.time.dt.month  # categories have be strings
    raw_data["year"] = raw_data.time.dt.year  # categories have be strings
    raw_data["hour_of_day"] = raw_data.time.apply(lambda x: x.hour)
    raw_data["day_of_week"] = raw_data.time.apply(lambda x: x.dayofweek)
    raw_data["day_of_month"] = raw_data.time.apply(lambda x: x.day)
    # TODO: use business time instead of calendar time. this changes a lot
    # during new year.
    raw_data["time_to_weekly_close"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="weekly_close_time"
    )
    raw_data["time_to_monthly_close"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="monthly_close_time"
    )
    raw_data["time_to_option_expiration"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="option_expiration_time"
    )
    if macro_data_builder.add_macro_event:
        raw_data["time_to_last_macro_event_imp1"] = raw_data.apply(
            time_diff, axis=1, base_col="timestamp", diff_col="last_macro_event_time_imp1"
        )
        raw_data["time_to_next_macro_event_imp1"] = raw_data.apply(
            time_diff, axis=1, base_col="timestamp", diff_col="next_macro_event_time_imp1"
        )
        raw_data["time_to_last_macro_event_imp2"] = raw_data.apply(
            time_diff, axis=1, base_col="timestamp", diff_col="last_macro_event_time_imp2"
        )
        raw_data["time_to_next_macro_event_imp2"] = raw_data.apply(
            time_diff, axis=1, base_col="timestamp", diff_col="next_macro_event_time_imp2"
        )
        raw_data["time_to_last_macro_event_imp3"] = raw_data.apply(
            time_diff, axis=1, base_col="timestamp", diff_col="last_macro_event_time_imp3"
        )
        raw_data["time_to_next_macro_event_imp3"] = raw_data.apply(
            time_diff, axis=1, base_col="timestamp", diff_col="next_macro_event_time_imp3"
        )
    raw_data["time_to_high_5_ff"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="time_high_5_ff"
    )
    raw_data["time_to_low_5_ff"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="time_low_5_ff"
    )
    raw_data["time_to_high_11_ff"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="time_high_11_ff"
    )
    raw_data["time_to_low_11_ff"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="time_low_11_ff"
    )
    raw_data["time_to_high_21_ff"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="time_high_21_ff"
    )
    raw_data["time_to_low_21_ff"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="time_low_21_ff"
    )
    raw_data["time_to_high_51_ff"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="time_high_51_ff"
    )
    raw_data["time_to_low_51_ff"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="time_low_51_ff"
    )
    raw_data["time_to_high_201_ff"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="time_high_201_ff"
    )
    raw_data["time_to_low_201_ff"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="time_low_201_ff"
    )

    if add_daily_rolling_features:
        raw_data["time_to_high_5d_ff"] = raw_data.apply(
            time_diff, axis=1, base_col="timestamp", diff_col="time_high_5d_ff"
        )
        raw_data["time_to_low_5d_ff"] = raw_data.apply(
            time_diff, axis=1, base_col="timestamp", diff_col="time_low_5d_ff"
        )
        raw_data["time_to_high_11d_ff"] = raw_data.apply(
            time_diff, axis=1, base_col="timestamp", diff_col="time_high_11d_ff"
        )
        raw_data["time_to_low_11d_ff"] = raw_data.apply(
            time_diff, axis=1, base_col="timestamp", diff_col="time_low_11d_ff"
        )
        raw_data["time_to_high_21d_ff"] = raw_data.apply(
            time_diff, axis=1, base_col="timestamp", diff_col="time_high_21d_ff"
        )
        raw_data["time_to_low_21d_ff"] = raw_data.apply(
            time_diff, axis=1, base_col="timestamp", diff_col="time_low_21d_ff"
        )
        raw_data["time_to_high_51d_ff"] = raw_data.apply(
            time_diff, axis=1, base_col="timestamp", diff_col="time_high_51d_ff"
        )
        raw_data["time_to_low_51d_ff"] = raw_data.apply(
            time_diff, axis=1, base_col="timestamp", diff_col="time_low_51d_ff"
        )
        raw_data["time_to_high_201d_ff"] = raw_data.apply(
            time_diff, axis=1, base_col="timestamp", diff_col="time_high_201d_ff"
        )
        raw_data["time_to_low_201d_ff"] = raw_data.apply(
            time_diff, axis=1, base_col="timestamp", diff_col="time_low_201d_ff"
        )
    return raw_data


@profile_util.profile
def add_example_group_features(cal, macro_data_builder, raw_data,
                               add_daily_rolling_features=True,
                               interval_mins=30):
    new_york_cal = mcal.get_calendar("NYSE")
    lse_cal = mcal.get_calendar("LSE")
    lse_cal = mcal.get_calendar("LSE")

    def vwap_around(ser, cum_dv_col, cum_volume_col):
        data = raw_data.loc[ser.index]
        cum_volume = data[cum_volume_col].iloc[-1]-data[cum_volume_col].iloc[0]
        cum_dv = data[cum_dv_col].iloc[-1]-data[cum_dv_col].iloc[0]
        if cum_volume>0:
            return cum_dv/cum_volume
        else:
            return data["close"].iloc[-1]
        return 0

    if add_daily_rolling_features:
        raw_data["ret_from_high_5d"] = raw_data.apply(
            time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_high_5d_ff"
        )
        raw_data["ret_from_high_11d"] = raw_data.apply(
            time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_high_11d_ff"
        )
        raw_data["ret_from_high_21d"] = raw_data.apply(
            time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_high_21d_ff"
        )
        raw_data["ret_from_high_51d"] = raw_data.apply(
            time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_high_51d_ff"
        )
        raw_data["ret_from_high_201d"] = raw_data.apply(
            time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_high_201d_ff"
        )
        raw_data["ret_from_low_5d"] = raw_data.apply(
            time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_low_5d_ff"
        )
        raw_data["ret_from_low_11d"] = raw_data.apply(
            time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_low_11d_ff"
        )
        raw_data["ret_from_low_21d"] = raw_data.apply(
            time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_low_21d_ff"
        )
        raw_data["ret_from_low_51d"] = raw_data.apply(
            time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_low_51d_ff"
        )
        raw_data["ret_from_low_201d"] = raw_data.apply(
            time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_low_201d_ff"
        )

    raw_data["ret_from_high_5"] = raw_data.apply(
        time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_high_5_ff"
    )
    raw_data["ret_from_high_11"] = raw_data.apply(
        time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_high_11_ff"
    )
    raw_data["ret_from_high_21"] = raw_data.apply(
        time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_high_21_ff"
    )
    raw_data["ret_from_high_51"] = raw_data.apply(
        time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_high_51_ff"
    )
    raw_data["ret_from_high_201"] = raw_data.apply(
        time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_high_201_ff"
    )
    raw_data["ret_from_low_5"] = raw_data.apply(
        time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_low_5_ff"
    )
    raw_data["ret_from_low_11"] = raw_data.apply(
        time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_low_11_ff"
    )
    raw_data["ret_from_low_21"] = raw_data.apply(
        time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_low_21_ff"
    )
    raw_data["ret_from_low_51"] = raw_data.apply(
        time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_low_51_ff"
    )
    raw_data["ret_from_low_201"] = raw_data.apply(
        time_diff, axis=1, base_col="close_back_cumsum", diff_col="close_low_201_ff"
    )
    
    if macro_data_builder.add_macro_event:
        raw_data["last_macro_event_cum_dv_imp1"] = raw_data.apply(
            fill_cum_dv, time_col="last_macro_event_time_imp1", interval_mins=interval_mins, axis=1)
        raw_data["last_macro_event_cum_volume_imp1"] = raw_data.apply(
            fill_cum_volume, time_col="last_macro_event_time_imp1", interval_mins=interval_mins, axis=1)
        raw_data["last_macro_event_cum_dv_imp1"] = raw_data.last_macro_event_cum_dv_imp1.ffill()
        raw_data["last_macro_event_cum_volume_imp1"] = raw_data.last_macro_event_cum_volume_imp1.ffill()
        raw_data["vwap_since_last_macro_event_imp1"] = raw_data.apply(
            compute_vwap, dv_col="last_macro_event_cum_dv_imp1",
            volume_col="last_macro_event_cum_volume_imp1", axis=1)
        logging.error(f"raw_data:{raw_data.iloc[-3:]}")

        raw_data["around_macro_event_cum_dv_imp1"] = raw_data.apply(
            fill_cum_dv, time_col="last_macro_event_time_imp1", interval_mins=interval_mins,
            axis=1)
        raw_data["around_macro_event_cum_volume_imp1"] = raw_data.apply(
            fill_cum_volume, time_col="last_macro_event_time_imp1",
            interval_mins=interval_mins,
            axis=1)
        rol = raw_data.around_macro_event_cum_dv_imp1.rolling(window=2)
        raw_data["vwap_around_macro_event_imp1"] = rol.apply(
            vwap_around,
            args=("around_macro_event_cum_dv_imp1","around_macro_event_cum_volume_imp1"), raw=False)
        raw_data["vwap_around_macro_event_imp1"] = raw_data.vwap_around_macro_event_imp1.ffill()
        raw_data["ret_from_vwap_around_macro_event_imp1"] = raw_data.apply(
            compute_ret, base_col="vwap_around_macro_event_imp1", axis=1)

        raw_data = raw_data.drop(columns=["around_macro_event_cum_dv_imp1", "around_macro_event_cum_volume_imp1",
                                          "last_macro_event_cum_dv_imp1", "last_macro_event_cum_volume_imp1",
                                          "around_macro_event_cum_dv_imp1", "around_macro_event_cum_volume_imp1"])

        raw_data["last_macro_event_cum_dv_imp2"] = raw_data.apply(
            fill_cum_dv, time_col="last_macro_event_time_imp2", axis=1)
        raw_data["last_macro_event_cum_volume_imp2"] = raw_data.apply(
            fill_cum_volume, time_col="last_macro_event_time_imp2", axis=1)
        raw_data["last_macro_event_cum_dv_imp2"] = raw_data.last_macro_event_cum_dv_imp2.ffill()
        raw_data["last_macro_event_cum_volume_imp2"] = raw_data.last_macro_event_cum_volume_imp2.ffill()
        raw_data["vwap_since_last_macro_event_imp2"] = raw_data.apply(
            compute_vwap, dv_col="last_macro_event_cum_dv_imp2",
            volume_col="last_macro_event_cum_volume_imp2", axis=1)

        raw_data["around_macro_event_cum_dv_imp2"] = raw_data.apply(
            fill_cum_dv, time_col="last_macro_event_time_imp2", interval_mins=interval_mins,
            axis=1)
        raw_data["around_macro_event_cum_volume_imp2"] = raw_data.apply(
            fill_cum_volume, time_col="last_macro_event_time_imp2",
            interval_mins=interval_mins,
            axis=1)
        rol = raw_data.around_macro_event_cum_dv_imp2.rolling(window=2)
        raw_data["vwap_around_macro_event_imp2"] = rol.apply(
            vwap_around,
            args=("around_macro_event_cum_dv_imp2","around_macro_event_cum_volume_imp2"), raw=False)
        raw_data["vwap_around_macro_event_imp2"] = raw_data.vwap_around_macro_event_imp2.ffill()
        raw_data["ret_from_vwap_around_macro_event_imp2"] = raw_data.apply(
            compute_ret, base_col="vwap_around_macro_event_imp2", axis=1)

        raw_data = raw_data.drop(columns=["around_macro_event_cum_dv_imp2", "around_macro_event_cum_volume_imp2",
                                          "last_macro_event_cum_dv_imp2", "last_macro_event_cum_volume_imp2",
                                          "around_macro_event_cum_dv_imp2", "around_macro_event_cum_volume_imp2"])

        raw_data["last_macro_event_cum_dv_imp3"] = raw_data.apply(
            fill_cum_dv, time_col="last_macro_event_time_imp3", axis=1)
        raw_data["last_macro_event_cum_volume_imp3"] = raw_data.apply(
            fill_cum_volume, time_col="last_macro_event_time_imp3", axis=1)
        raw_data["last_macro_event_cum_dv_imp3"] = raw_data.last_macro_event_cum_dv_imp3.ffill()
        raw_data["last_macro_event_cum_volume_imp3"] = raw_data.last_macro_event_cum_volume_imp3.ffill()
        raw_data["vwap_since_last_macro_event_imp3"] = raw_data.apply(
            compute_vwap, dv_col="last_macro_event_cum_dv_imp3",
            volume_col="last_macro_event_cum_volume_imp3", axis=1)

        raw_data["around_macro_event_cum_dv_imp3"] = raw_data.apply(
            fill_cum_dv, time_col="last_macro_event_time_imp3", interval_mins=interval_mins,
            axis=1)
        raw_data["around_macro_event_cum_volume_imp3"] = raw_data.apply(
            fill_cum_volume, time_col="last_macro_event_time_imp3",
            interval_mins=interval_mins,
            axis=1)
        rol = raw_data.around_macro_event_cum_dv_imp3.rolling(window=2)
        raw_data["vwap_around_macro_event_imp3"] = rol.apply(
            vwap_around,
            args=("around_macro_event_cum_dv_imp3","around_macro_event_cum_volume_imp3"), raw=False)
        raw_data["vwap_around_macro_event_imp3"] = raw_data.vwap_around_macro_event_imp3.ffill()
        raw_data["ret_from_vwap_around_macro_event_imp3"] = raw_data.apply(
            compute_ret, base_col="vwap_around_macro_event_imp3", axis=1)

        raw_data = raw_data.drop(columns=["around_macro_event_cum_dv_imp3", "around_macro_event_cum_volume_imp3",
                                          "last_macro_event_cum_dv_imp3", "last_macro_event_cum_volume_imp3",
                                          "around_macro_event_cum_dv_imp3", "around_macro_event_cum_volume_imp3"])

    raw_data["new_york_open_cum_dv"] = raw_data.apply(fill_cum_dv, time_col="new_york_last_open_time",
                                                      interval_mins=interval_mins, axis=1)
    raw_data["new_york_open_cum_volume"] = raw_data.apply(fill_cum_volume, time_col="new_york_last_open_time",
                                                          interval_mins=interval_mins, axis=1)
    raw_data["new_york_open_cum_dv"] = raw_data.new_york_open_cum_dv.ffill()
    raw_data["new_york_open_cum_volume"] = raw_data.new_york_open_cum_volume.ffill()
    raw_data["vwap_since_new_york_open"] = raw_data.apply(compute_vwap, dv_col="new_york_open_cum_dv",
                                                          volume_col="new_york_open_cum_volume", axis=1)

    raw_data["around_new_york_open_cum_dv"] = raw_data.apply(
        fill_cum_dv, time_col="new_york_last_open_time",
        interval_mins=interval_mins,
        axis=1)
    raw_data["around_new_york_open_cum_volume"] = raw_data.apply(
        fill_cum_volume, time_col="new_york_last_open_time",
        interval_mins=interval_mins, 
        axis=1)
    rol = raw_data.around_new_york_open_cum_dv.rolling(window=2)

    raw_data["vwap_around_new_york_open"] = rol.apply(
        vwap_around,
        args=("around_new_york_open_cum_dv","around_new_york_open_cum_volume"), raw=False)
    raw_data["vwap_around_new_york_open"] = raw_data.vwap_around_new_york_open.ffill()
    raw_data["ret_from_vwap_around_new_york_open"] = raw_data.apply(
        compute_ret, base_col="vwap_around_new_york_open", axis=1)
    raw_data = raw_data.drop(columns=[
        "around_new_york_open_cum_dv", "around_new_york_open_cum_volume"])

    raw_data["london_open_cum_dv"] = raw_data.apply(fill_cum_dv, time_col="london_last_open_time", interval_mins=interval_mins, axis=1)
    raw_data["london_open_cum_volume"] = raw_data.apply(fill_cum_volume, time_col="london_last_open_time", interval_mins=interval_mins, axis=1)
    raw_data["london_open_cum_dv"] = raw_data.london_open_cum_dv.ffill()
    raw_data["london_open_cum_volume"] = raw_data.london_open_cum_volume.ffill()
    raw_data["vwap_since_london_open"] = raw_data.apply(compute_vwap, dv_col="london_open_cum_dv",
                                                        volume_col="london_open_cum_volume", axis=1)    
    raw_data["around_london_open_cum_dv"] = raw_data.apply(
        fill_cum_dv, time_col="london_last_open_time", interval_mins=interval_mins,
        axis=1)
    raw_data["around_london_open_cum_volume"] = raw_data.apply(
        fill_cum_volume, time_col="london_last_open_time",
        interval_mins=interval_mins, 
        axis=1)
    rol = raw_data.around_london_open_cum_dv.rolling(window=2)
    raw_data["vwap_around_london_open"] = rol.apply(
        vwap_around,
        args=("around_london_open_cum_dv","around_london_open_cum_volume"), raw=False)
    raw_data["vwap_around_london_open"] = raw_data.vwap_around_london_open.ffill()
    raw_data["ret_from_vwap_around_london_open"] = raw_data.apply(
        compute_ret, base_col="vwap_around_london_open", axis=1)
    raw_data = raw_data.drop(columns=[
        "around_london_open_cum_dv", "around_london_open_cum_volume"])

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
