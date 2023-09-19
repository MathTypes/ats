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

from hamilton.function_modifiers import subdag, value
from hamilton.function_modifiers import does, extract_columns, parameterize, source, value

from omegaconf.dictconfig import DictConfig
import pandas_market_calendars as mcal
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler

from pandas_market_calendars.exchange_calendar_cme import CMEEquityExchangeCalendar
from ats.event.macro_indicator import MacroDataBuilder
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
from ats.features.preprocess.feature_utils import *
from ats.features.preprocess import time_features
from pandas.core.groupby import generic

VOL_THRESHOLD = 5  # multiple to winsorise by
HALFLIFE_WINSORISE = 252

#@profile_util.profile
#@njit(parallel=True)

def clean_sorted_data(sorted_data : pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    interval_minutes = config.dataset.interval_mins
    add_daily_rolling_features = config.model.features.add_daily_rolling_features
    base_price = config.dataset.base_price
    raw_data = sorted_data
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
            
    return raw_data.sort_index()

#def timestamp(clean_sorted_data: pd.DataFrame) -> generic.SeriesGroupBy:
#    return clean_sorted_data["timestamp"]


#[[
#        "volume", "dv", "close", "high", "low", "open", "timestamp"]]

#    return new_features


#    .apply(ticker_transform, config)
#    new_features = new_features.drop(columns=["volume", "dv", "close", "high", "low", "open", "timestamp"])
#    raw_data = raw_data.join(new_features)
#    raw_data.reset_index(drop=True, inplace=True)
#    #del new_features

#    return raw_data

def winsorize_col(col: pd.Series) -> pd.Series:
    ewm = col.ewm(halflife=HALFLIFE_WINSORISE)
    means = ewm.mean()
    stds = ewm.std()
    del ewm
    col = np.minimum(col, means + VOL_THRESHOLD * stds)
    col = np.maximum(col, means - VOL_THRESHOLD * stds)
    return col

def close(clean_sorted_data: pd.DataFrame) -> pd.Series:
    series = clean_sorted_data["close"].groupby(["ticker"], group_keys=False)
    return series.transform(winsorize_col)

def ticker(clean_sorted_data: pd.DataFrame) -> pd.Series:
    series = clean_sorted_data.index.get_level_values("ticker").to_series()
    return series

def high(clean_sorted_data: pd.DataFrame) -> pd.Series:
    series = clean_sorted_data["high"].groupby(["ticker"], group_keys=False)
    return series.transform(winsorize_col)

def low(clean_sorted_data: pd.DataFrame) -> pd.Series:
    series = clean_sorted_data["low"].groupby(["ticker"], group_keys=False)
    return series.transform(winsorize_col)

def open(clean_sorted_data: pd.DataFrame) -> pd.Series:
    series = clean_sorted_data["open"].groupby(["ticker"], group_keys=False)
    return series.transform(winsorize_col)

def cum_volume(clean_sorted_data: pd.DataFrame) -> pd.Series:
    series = clean_sorted_data["volume"].groupby(["ticker"], group_keys=False)
    return series.transform("cumsum")

def cum_dv(clean_sorted_data: pd.DataFrame) -> pd.Series:
    series = clean_sorted_data["dv"].groupby(["ticker"], group_keys=False)
    return series.transform("cumsum")

def back_ret(x:pd.Series, base_price:float) -> pd.Series:
    return np.log(x+base_price) - np.log(x.shift(1)+base_price)

def back_volume(x:pd.Series) -> pd.Series:
    return np.log(x+2) - np.log(x.shift(1)+2)

def close_back(close: pd.Series, base_price: float) -> pd.Series:
    return close.groupby(["ticker"]).transform(lambda x : back_ret(x, base_price))

def close_velocity_back(close_back: pd.Series) -> pd.Series:
    series = close_back.groupby(["ticker"], group_keys=False)
    return series.transform(lambda x: x-x.shift(1))

def high_back(high: pd.Series, base_price: float) -> pd.Series:
    return high.groupby(["ticker"]).transform(lambda x: back_ret(x, base_price))

def low_back(low: pd.Series, base_price:float) -> pd.Series:
    return low.groupby(["ticker"]).transform(lambda x:back_ret(x, base_price))

def open_back(open: pd.Series,base_price: float) -> pd.Series:
    return open.groupby(["ticker"]).transform(lambda x:back_ret(x, base_price))

def volume_back(cum_volume: pd.Series) -> pd.Series:
    return cum_volume.groupby(["ticker"]).transform(lambda x:back_volume(x))

def dv_back(dv: pd.Series) -> pd.Series:
    return dv.groupby(["ticker"]).transform(lambda x:back_volume(x))

def close_back_cumsum(close_back: pd.Series) -> pd.Series:
    return close_back.groupby(['ticker']).transform("cumsum")

def volume_back_cumsum(volume_back: pd.Series) -> pd.Series:
    return volume_back.groupby(['ticker']).transform("cumsum")

@parameterize(
    close_back_cumsum_high_1d_ff={"steps": value(1)},
    close_back_cumsum_high_5d_ff={"steps": value(5)},
    close_back_cumsum_high_11d_ff={"steps": value(11)},
    close_back_cumsum_high_21d_ff={"steps": value(21)},
    close_back_cumsum_high_51d_ff={"steps": value(51)},
    close_back_cumsum_high_101d_ff={"steps": value(101)},
    close_back_cumsum_high_201d_ff={"steps": value(201)},
)
def close_back_cumsum_day_high_tmpl(steps:int, close_back_cumsum:pd.Series, interval_per_day:int) -> pd.Series:
    return close_back_cumsum.groupby(['ticker']).transform(lambda x: x.rolling(steps*interval_per_day).max().ffill())

@parameterize(
    close_back_cumsum_high_5_ff={"steps": value(5)},
    close_back_cumsum_high_11_ff={"steps": value(11)},
    close_back_cumsum_high_21_ff={"steps": value(21)},
    close_back_cumsum_high_51_ff={"steps": value(51)},
    close_back_cumsum_high_101_ff={"steps": value(101)},
    close_back_cumsum_high_201_ff={"steps": value(201)},
)
def close_back_cumsum_high_tmpl(steps:int, close_back_cumsum:pd.Series) -> pd.Series:
    return close_back_cumsum.groupby(['ticker']).transform(lambda x: x.rolling(steps).max().ffill())

@parameterize(
    close_back_cumsum_low_1d_ff={"steps": value(1)},
    close_back_cumsum_low_5d_ff={"steps": value(5)},
    close_back_cumsum_low_11d_ff={"steps": value(11)},
    close_back_cumsum_low_21d_ff={"steps": value(21)},
    close_back_cumsum_low_51d_ff={"steps": value(51)},
    close_back_cumsum_low_101d_ff={"steps": value(101)},
    close_back_cumsum_low_201d_ff={"steps": value(201)},
)
def close_back_cumsum_day_low_tmpl(steps:int, close_back_cumsum:pd.Series, interval_per_day:int) -> pd.Series:
    return close_back_cumsum.groupby(['ticker']).transform(lambda x: x.rolling(steps*interval_per_day).min().ffill())

@parameterize(
    close_back_cumsum_low_5_ff={"steps": value(5)},
    close_back_cumsum_low_11_ff={"steps": value(11)},
    close_back_cumsum_low_21_ff={"steps": value(21)},
    close_back_cumsum_low_51_ff={"steps": value(51)},
    close_back_cumsum_low_101_ff={"steps": value(101)},
    close_back_cumsum_low_201_ff={"steps": value(201)},
)
def close_back_cumsum_low_tmpl(steps:int, close_back_cumsum:pd.Series) -> pd.Series:
    return close_back_cumsum.groupby(['ticker']).transform(lambda x: x.rolling(steps).min().ffill())

@parameterize(
    time_high_1d_ff={"close_col": source("close_high_1d_ff"), "col_name":value("close_high_1d_ff")},
    time_high_5d_ff={"close_col": source("close_high_5d_ff"), "col_name":value("close_high_5d_ff")},
    time_high_11d_ff={"close_col": source("close_high_11d_ff"), "col_name":value("close_high_11d_ff")},
    time_high_21d_ff={"close_col": source("close_high_21d_ff"), "col_name":value("close_high_21d_ff")},
    time_high_51d_ff={"close_col": source("close_high_51d_ff"), "col_name":value("close_high_51d_ff")},
    time_high_101d_ff={"close_col": source("close_high_101d_ff"), "col_name":value("close_high_101d_ff")},
    time_high_201d_ff={"close_col": source("close_high_201d_ff"), "col_name":value("close_high_201d_ff")},
    time_low_1d_ff={"close_col":source("close_low_1d_ff"), "col_name":value("close_low_1d_ff")},
    time_low_5d_ff={"close_col": source("close_low_5d_ff"), "col_name":value("close_low_5d_ff")},
    time_low_11d_ff={"close_col": source("close_low_11d_ff"), "col_name":value("close_low_11d_ff")},
    time_low_21d_ff={"close_col": source("close_low_21d_ff"), "col_name":value("close_low_21d_ff")},
    time_low_51d_ff={"close_col": source("close_low_51d_ff"), "col_name":value("close_low_51d_ff")},
    time_low_101d_ff={"close_col": source("close_low_101d_ff"), "col_name":value("close_low_101d_ff")},
    time_low_201d_ff={"close_col": source("close_low_201d_ff"), "col_name":value("close_low_201d_ff")},
    time_high_5d_bf={"close_col": source("close_high_5d_bf"), "col_name":value("close_high_5d_bf")},
    time_high_21d_bf={"close_col": source("close_high_21d_bf"), "col_name":value("close_high_21d_bf")},
    time_high_51d_bf={"close_col": source("close_high_51d_bf"), "col_name":value("close_high_51d_bf")},
    time_high_101d_bf={"close_col": source("close_high_101d_bf"), "col_name":value("close_high_101d_bf")},
    time_high_201d_bf={"close_col": source("close_high_201d_bf"), "col_name":value("close_high_201d_bf")},
    time_low_5d_bf={"close_col": source("close_low_5d_bf"), "col_name":value("close_low_5d_bf")},
    time_low_21d_bf={"close_col": source("close_low_21d_bf"), "col_name":value("close_low_21d_bf")},
    time_low_51d_bf={"close_col": source("close_low_51d_bf"), "col_name":value("close_low_51d_bf")},
    time_low_101d_bf={"close_col": source("close_low_101d_bf"), "col_name":value("close_low_101d_bf")},
    time_low_201d_bf={"close_col": source("close_low_201d_bf"), "col_name":value("close_low_201d_bf")},
    time_high_5_ff={"close_col": source("close_high_5_ff"), "col_name":value("close_high_5_ff")},
    time_high_11_ff={"close_col": source("close_high_11_ff"), "col_name":value("close_high_11_ff")},
    time_high_21_ff={"close_col": source("close_high_21_ff"), "col_name":value("close_high_21_ff")},
    time_high_51_ff={"close_col": source("close_high_51_ff"), "col_name":value("close_high_51_ff")},
    time_high_101_ff={"close_col": source("close_high_101_ff"), "col_name":value("close_high_101_ff")},
    time_high_201_ff={"close_col": source("close_high_201_ff"), "col_name":value("close_high_201_ff")},
    time_low_5_ff={"close_col": source("close_low_5_ff"), "col_name":value("close_low_5_ff")},
    time_low_11_ff={"close_col": source("close_low_11_ff"), "col_name":value("close_low_11_ff")},
    time_low_21_ff={"close_col": source("close_low_21_ff"), "col_name":value("close_low_21_ff")},
    time_low_51_ff={"close_col": source("close_low_51_ff"), "col_name":value("close_low_51_ff")},
    time_low_101_ff={"close_col": source("close_low_101_ff"), "col_name":value("close_low_101_ff")},
    time_low_201_ff={"close_col": source("close_low_201_ff"), "col_name":value("close_low_201_ff")},
    time_high_5_bf={"close_col": source("close_high_5_bf"), "col_name":value("close_high_5_bf")},
    time_high_21_bf={"close_col": source("close_high_21_bf"), "col_name":value("close_high_21_bf")},
    time_high_51_bf={"close_col": source("close_high_51_bf"), "col_name":value("close_high_51_bf")},
    time_high_101_bf={"close_col": source("close_high_101_bf"), "col_name":value("close_high_101_bf")},
    time_high_201_bf={"close_col": source("close_high_201_bf"), "col_name":value("close_high_201_bf")},
    time_low_5_bf={"close_col": source("close_low_5_bf"), "col_name":value("close_low_5_bf")},
    time_low_21_bf={"close_col": source("close_low_21_bf"), "col_name":value("close_low_21_bf")},
    time_low_51_bf={"close_col": source("close_low_51_bf"), "col_name":value("close_low_51_bf")},
    time_low_101_bf={"close_col": source("close_low_101_bf"), "col_name":value("close_low_101_bf")},
    time_low_201_bf={"close_col": source("close_low_201_bf"), "col_name":value("close_low_201_bf")},
)
def time_with_close_tmpl(close:pd.Series, timestamp:pd.Series, close_col:pd.Series, col_name:str) -> pd.Series:
    #logging.error(f"time_with_close_tmpl_close:{close}, col_name:{col_name}")
    #logging.error(f"time_with_close_tmpl_timestamp:{timestamp}, col_name:{col_name}")
    #logging.error(f"time_with_close_tmpl_close_col:{close_col}, col_name:{col_name}")
    df = pd.concat([close, close_col], axis=1)
    df.columns = ["close", "close_high"]
    #logging.error(f"df:{df}")
    #logging.error(f"df.index:{df.index}")
    df = df.reset_index()
    #logging.error(f"df after reset:{df}")
    series = df.groupby(['ticker']).apply(lambda x: find_close_time(x, "close_high"))
    #series = series.unstack(level=2).drop(columns=['ticker'])
    series = series.set_index(["timestamp","ticker"])
    series = series.ffill()
    #logging.error(f"time_with_close_tmpl after unstack col_name:{col_name}, time_with_close_tmpl:{series.shape}, series:{series}")
    #res = series[['timestamp']]
    #logging.error(f"time_with_close_tmpl_res, col_name:{col_name}, time_with_close_tmpl:{series}")
    return series
    #return series

def find_close_time(df:pd.DataFrame, close_col_name:str) -> pd.Series:
    #logging.error(f"close_col:{close_col_name}")
    #logging.error(f"df:{df.shape}")
    #logging.error(f"df:{df}")
    res = df.apply(lambda x: get_close_time(x, close_col_name), axis=1, result_type="expand")
    logging.error(f"find_close_time:{res}, res.shape:{res.shape}")
    return res


@parameterize(
    ret_from_vwap_since_last_macro_event_imp1={"time_col": value("last_macro_event_time_imp1")},
    ret_from_vwap_since_last_macro_event_imp2={"time_col": value("last_macro_event_time_imp2")},
    ret_from_vwap_since_last_macro_event_imp3={"time_col": value("last_macro_event_time_imp3")},
    ret_from_vwap_since_new_york_open={"time_col": value("new_york_last_open_time")},
    ret_from_vwap_since_london_open={"time_col": value("london_last_open_time")},
)
def ret_from_vwap(time_features:pd.DataFrame,
                  cum_dv:pd.Series, cum_volume:pd.Series,
                  time_col:str, interval_mins: int) ->pd.Series:
    raw_data = time_features
    cum_dv_col = f"{time_col}_cum_dv"
    cum_volume_col = f"{time_col}_cum_volume"
    raw_data["cum_dv"] = cum_dv
    raw_data["cum_volume"] = cum_volume
    #logging.error(f"time_features:{time_features.iloc[20:40]}")
    raw_data[cum_dv_col] = raw_data.apply(
        fill_cum_dv, time_col=time_col,
        pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3, axis=1)
    raw_data[cum_volume_col] = raw_data.apply(
        fill_cum_volume, time_col=time_col,
        pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3, axis=1)
    raw_data[cum_dv_col] = raw_data[cum_dv_col].ffill()
    raw_data[cum_volume_col] = raw_data[cum_volume_col].ffill()
    return raw_data.apply(compute_ret_from_vwap, dv_col=cum_dv_col, volume_col=cum_volume_col, axis=1)


@parameterize(
    vwap_around_macro_event_imp1={"time_col": value("last_macro_event_time_imp1")},
    vwap_around_macro_event_imp2={"time_col": value("last_macro_event_time_imp2")},
    vwap_around_macro_event_imp3={"time_col": value("last_macro_event_time_imp3")},
    vwap_around_new_york_open={"time_col": value("new_york_last_open_time")},
    vwap_around_london_open={"time_col": value("london_last_open_time")},
    vwap_around_new_york_close={"time_col": value("new_york_last_close_time")},
    vwap_around_london_close={"time_col": value("london_last_close_time")},
)
def around_vwap(time_features:pd.DataFrame,
                cum_dv:pd.Series, cum_volume:pd.Series,
                time_col:str, interval_mins:int) ->pd.Series:
    raw_data = time_features
    cum_dv_col = f"around_{time_col}_cum_dv"
    cum_volume_col = f"around_{time_col}_cum_volume"
    raw_data["cum_dv"] = cum_dv
    raw_data["cum_volume"] = cum_volume
    raw_data[cum_dv_col] = raw_data.apply(fill_cum_dv, time_col=time_col,
                                          pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3, axis=1)
    raw_data[cum_volume_col] = raw_data.apply(fill_cum_volume, time_col=time_col,
                                              pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3, axis=1)
    rol = raw_data[cum_dv_col].rolling(window=2)

    def vwap_around(ser, cum_dv_col, cum_volume_col):
        data = raw_data.loc[ser.index]
        cum_volume_diff = data[cum_volume_col].iloc[-1]-data[cum_volume_col].iloc[0]
        cum_dv_diff = data[cum_dv_col].iloc[-1]-data[cum_dv_col].iloc[0]
        if cum_volume_diff>0:
            return cum_dv_diff/cum_volume_diff
        else:
            return data["close"].iloc[-1]
        return 0
    return rol.apply(vwap_around, args=(cum_dv_col,cum_volume_col), raw=False).ffill()


@parameterize(
    vwap_post_new_york_open={"time_col": value("new_york_last_open_time")},
    vwap_post_london_open={"time_col": value("london_last_open_time")},
    vwap_post_new_york_close={"time_col": value("new_york_last_close_time")},
    vwap_post_london_close={"time_col": value("london_last_close_time")},
    vwap_post_macro_event_imp1={"time_col": value("last_macro_event_time_imp1")},
    vwap_post_macro_event_imp2={"time_col": value("last_macro_event_time_imp2")},
    vwap_post_macro_event_imp3={"time_col": value("last_macro_event_time_imp3")},
)
def post_vwap(time_features:pd.DataFrame,
              cum_dv:pd.Series, cum_volume:pd.Series,
              time_col:str, interval_mins:int) ->pd.Series:
    raw_data = time_features
    raw_data["cum_dv"] = cum_dv
    raw_data["cum_volume"] = cum_volume
    cum_dv_col = f"post_{time_col}_cum_dv"
    cum_volume_col = f"post_{time_col}_cum_volume"
    raw_data[cum_dv_col] = raw_data.apply(fill_cum_dv, time_col=time_col,
                                          pre_interval_mins=0, post_interval_mins=interval_mins*3, axis=1)
    raw_data[cum_volume_col] = raw_data.apply(fill_cum_volume, time_col=time_col,
                                              pre_interval_mins=0, post_interval_mins=interval_mins*3, axis=1)
    rol = raw_data[cum_dv_col].rolling(window=2)

    def vwap_around(ser, cum_dv_col, cum_volume_col):
        data = raw_data.loc[ser.index]
        cum_volume_diff = data[cum_volume_col].iloc[-1]-data[cum_volume_col].iloc[0]
        cum_dv_diff = data[cum_dv_col].iloc[-1]-data[cum_dv_col].iloc[0]
        if cum_volume_diff>0:
            return cum_dv_diff/cum_volume_diff
        else:
            return data["close"].iloc[-1]
        return 0
    series = rol.apply(vwap_around, args=(cum_dv_col,cum_volume_col), raw=False).ffill()
    #logging.error(f"pre_vwap_series:{series}")
    return series


@parameterize(
    vwap_pre_new_york_open={"time_col": value("new_york_last_open_time")},
    vwap_pre_london_open={"time_col": value("london_last_open_time")},
    vwap_pre_new_york_close={"time_col": value("new_york_last_close_time")},
    vwap_pre_london_close={"time_col": value("london_last_close_time")},
    vwap_pre_macro_event_imp1={"time_col": value("last_macro_event_time_imp1")},
    vwap_pre_macro_event_imp2={"time_col": value("last_macro_event_time_imp2")},
    vwap_pre_macro_event_imp3={"time_col": value("last_macro_event_time_imp3")},
)
def pre_vwap(time_features:pd.DataFrame,
             cum_dv:pd.Series, cum_volume:pd.Series,
             time_col:str, interval_mins:int) ->pd.Series:
    raw_data = time_features
    raw_data["cum_dv"] = cum_dv
    raw_data["cum_volume"] = cum_volume
    cum_dv_col = f"pre_{time_col}_cum_dv"
    cum_volume_col = f"pre_{time_col}_cum_volume"
    #logging.error(f"pre_vwap_time_features:{time_features.iloc[10:40][[time_col,'timestamp','cum_dv','cum_volume']]}")
    #logging.error(f"pre_vwap_time_col:{time_col}, interval_mins:{interval_mins}")
    raw_data[cum_dv_col] = raw_data.apply(fill_cum_dv, time_col=time_col,
                                          pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)
    raw_data[cum_volume_col] = raw_data.apply(fill_cum_volume, time_col=time_col,
                                              pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)
    rol = raw_data[cum_dv_col].rolling(window=2)
    #logging.error(f"raw_data[cum_dv_col]:{raw_data[cum_dv_col].iloc[10:60]}")
    #logging.error(f"raw_data[cum_volume_col]:{raw_data[cum_volume_col].iloc[10:60]}")

    def vwap_around(ser, cum_dv_col, cum_volume_col):
        data = raw_data.loc[ser.index]
        cum_volume_diff = data[cum_volume_col].iloc[-1]-data[cum_volume_col].iloc[0]
        cum_dv_diff = data[cum_dv_col].iloc[-1]-data[cum_dv_col].iloc[0]
        if cum_volume_diff>0:
            return cum_dv_diff/cum_volume_diff
        else:
            return data["close"].iloc[-1]
        return 0
    series = rol.apply(vwap_around, args=(cum_dv_col,cum_volume_col), raw=False).ffill()
    #logging.error(f"vwap_around_series:{series}")
    return series

        
#def close_back_cumsum(example_level_features:pd.DataFrame) -> pd.Series:
#    return example_level_features["close_back_cumsum"]

@parameterize(
    ret_from_close_cumsum_high_5d={"close_back_cumsum_ff": source("close_back_cumsum_high_5d_ff")},
    ret_from_close_cumsum_high_11d={"close_back_cumsum_ff": source("close_back_cumsum_high_11d_ff")},
    ret_from_close_cumsum_high_21d={"close_back_cumsum_ff": source("close_back_cumsum_high_21d_ff")},
    ret_from_close_cumsum_high_51d={"close_back_cumsum_ff": source("close_back_cumsum_high_51d_ff")},
    ret_from_close_cumsum_high_101d={"close_back_cumsum_ff": source("close_back_cumsum_high_101d_ff")},
    ret_from_close_cumsum_high_201d={"close_back_cumsum_ff": source("close_back_cumsum_high_201d_ff")},
    ret_from_close_cumsum_high_5={"close_back_cumsum_ff": source("close_back_cumsum_high_5_ff")},
    ret_from_close_cumsum_high_11={"close_back_cumsum_ff": source("close_back_cumsum_high_11_ff")},
    ret_from_close_cumsum_high_21={"close_back_cumsum_ff": source("close_back_cumsum_high_21_ff")},
    ret_from_close_cumsum_high_51={"close_back_cumsum_ff": source("close_back_cumsum_high_51_ff")},
    ret_from_close_cumsum_high_101={"close_back_cumsum_ff": source("close_back_cumsum_high_101_ff")},
    ret_from_close_cumsum_high_201={"close_back_cumsum_ff": source("close_back_cumsum_high_201_ff")},
    ret_from_close_cumsum_low_5d={"close_back_cumsum_ff": source("close_back_cumsum_low_5d_ff")},
    ret_from_close_cumsum_low_11d={"close_back_cumsum_ff": source("close_back_cumsum_low_11d_ff")},
    ret_from_close_cumsum_low_21d={"close_back_cumsum_ff": source("close_back_cumsum_low_21d_ff")},
    ret_from_close_cumsum_low_51d={"close_back_cumsum_ff": source("close_back_cumsum_low_51d_ff")},
    ret_from_close_cumsum_low_101d={"close_back_cumsum_ff": source("close_back_cumsum_low_101d_ff")},
    ret_from_close_cumsum_low_201d={"close_back_cumsum_ff": source("close_back_cumsum_low_201d_ff")},
    ret_from_close_cumsum_low_5={"close_back_cumsum_ff": source("close_back_cumsum_low_5_ff")},
    ret_from_close_cumsum_low_11={"close_back_cumsum_ff": source("close_back_cumsum_low_11_ff")},
    ret_from_close_cumsum_low_21={"close_back_cumsum_ff": source("close_back_cumsum_low_21_ff")},
    ret_from_close_cumsum_low_51={"close_back_cumsum_ff": source("close_back_cumsum_low_51_ff")},
    ret_from_close_cumsum_low_101={"close_back_cumsum_ff": source("close_back_cumsum_low_101_ff")},
    ret_from_close_cumsum_low_201={"close_back_cumsum_ff": source("close_back_cumsum_low_201_ff")},
)
def ret_from_close_cumsum_tmpl(close_back_cumsum: pd.Series, close_back_cumsum_ff: pd.Series) -> pd.Series:
    return close_back_cumsum_ff - close_back_cumsum

@parameterize(
    ret_from_high_1d={"base_col": source("close_high_1d_ff"),"col_name":value("close_high_1d_ff")},
    ret_from_high_1d_shift_1d={"base_col": source("close_high_1d_ff_shift_1d"),"col_name":value("close_high_1d_ff_shift_1d")},
    ret_from_high_5d={"base_col": source("close_high_5d_ff"),"col_name":value("close_high_5d_ff")},
    ret_from_high_5d_shift_5d={"base_col": source("close_high_5d_ff_shift_5d"),"col_name":value("close_high_5d_ff_shift_5d")},
    ret_from_high_11d={"base_col": source("close_high_11d_ff"),"col_name":value("close_high_11d_ff")},
    ret_from_high_11d_shift_11d={"base_col": source("close_high_11d_ff_shift_11d"),"col_name":value("close_high_11d_ff_shift_11d")},
    ret_from_high_21d={"base_col": source("close_high_21d_ff"),"col_name":value("close_high_21d_ff")},
    ret_from_high_21d_shift_21d={"base_col": source("close_high_21d_ff_shift_21d"),"col_name":value("close_high_21d_ff_shift_21d")},
    ret_from_high_51d={"base_col": source("close_high_51d_ff"),"col_name":value("close_high_51d_ff")},
    ret_from_high_101d={"base_col": source("close_high_101d_ff"),"col_name":value("close_high_101d_ff")},
    ret_from_high_201d={"base_col": source("close_high_201d_ff"),"col_name":value("close_high_201d_ff")},
    ret_from_low_1d={"base_col": source("close_low_1d_ff"),"col_name":value("close_low_1d_ff")},
    ret_from_low_1d_shift_1d={"base_col": source("close_low_1d_ff_shift_1d"),"col_name":value("close_low_1d_ff_shift_1d")},
    ret_from_low_5d={"base_col": source("close_low_5d_ff"),"col_name":value("close_low_5d_ff")},
    ret_from_low_5d_shift_5d={"base_col": source("close_low_5d_ff_shift_5d"),"col_name":value("close_low_5d_ff_shift_5d")},
    ret_from_low_11d={"base_col": source("close_low_11d_ff"),"col_name":value("close_low_11d_ff")},
    ret_from_low_11d_shift_11d={"base_col": source("close_low_11d_ff_shift_11d"),"col_name":value("close_low_11d_ff_shift_11d")},
    ret_from_low_21d={"base_col": source("close_low_21d_ff"),"col_name":value("close_low_21d_ff")},
    ret_from_low_21d_shift_21d={"base_col": source("close_low_21d_ff_shift_21d"),"col_name":value("close_low_21d_ff_shift_21d")},
    ret_from_low_51d={"base_col": source("close_low_51d_ff"),"col_name":value("close_low_51d_ff")},
    ret_from_low_101d={"base_col": source("close_low_101d_ff"),"col_name":value("close_low_101d_ff")},
    ret_from_low_201d={"base_col": source("close_low_201d_ff"),"col_name":value("close_low_201d_ff")},
    ret_from_high_5d_bf={"base_col": source("close_high_5d_bf"),"col_name":value("close_high_5d_bf")},
    ret_from_high_11d_bf={"base_col": source("close_high_11d_bf"),"col_name":value("close_high_11d_bf")},
    ret_from_high_21d_bf={"base_col": source("close_high_21d_bf"),"col_name":value("close_high_21d_bf")},
    ret_from_high_51d_bf={"base_col": source("close_high_51d_bf"),"col_name":value("close_high_51d_bf")},
    ret_from_low_5d_bf={"base_col": source("close_low_5d_bf"),"col_name":value("close_low_5d_bf")},
    ret_from_low_11d_bf={"base_col": source("close_low_11d_bf"),"col_name":value("close_low_11d_bf")},
    ret_from_low_21d_bf={"base_col": source("close_low_21d_bf"),"col_name":value("close_low_21d_bf")},
    ret_from_low_51d_bf={"base_col": source("close_low_51d_bf"),"col_name":value("close_low_51d_bf")},
    ret_from_high_5={"base_col": source("close_high_5_ff"),"col_name":value("close_high_5_ff")},
    ret_from_high_11={"base_col": source("close_high_11_ff"),"col_name":value("close_high_11_ff")},
    ret_from_high_21={"base_col": source("close_high_21_ff"),"col_name":value("close_high_21_ff")},
    ret_from_high_51={"base_col": source("close_high_51_ff"),"col_name":value("close_high_51_ff")},
    ret_from_high_101={"base_col": source("close_high_101_ff"),"col_name":value("close_high_101_ff")},
    ret_from_high_201={"base_col": source("close_high_201_ff"),"col_name":value("close_high_201_ff")},
    ret_from_low_5={"base_col": source("close_low_5_ff"),"col_name":value("close_low_5_ff")},
    ret_from_low_11={"base_col": source("close_low_11_ff"),"col_name":value("close_low_11_ff")},
    ret_from_low_21={"base_col": source("close_low_21_ff"),"col_name":value("close_low_21_ff")},
    ret_from_low_51={"base_col": source("close_low_51_ff"),"col_name":value("close_low_51_ff")},
    ret_from_low_101={"base_col": source("close_low_101_ff"),"col_name":value("close_low_101_ff")},
    ret_from_low_201={"base_col": source("close_low_201_ff"),"col_name":value("close_low_201_ff")},
    ret_from_vwap_pre_macro_event_imp1={"base_col": source("vwap_pre_macro_event_imp1"),"col_name":value("vwap_pre_macro_event_imp1")},
    ret_from_vwap_post_macro_event_imp1={"base_col": source("vwap_post_macro_event_imp1"),"col_name":value("vwap_post_macro_event_imp1")},
    ret_from_vwap_around_macro_event_imp1={"base_col": source("vwap_around_macro_event_imp1"),"col_name":value("vwap_around_macro_event_imp1")},
    ret_from_vwap_pre_macro_event_imp2={"base_col": source("vwap_pre_macro_event_imp2"),"col_name":value("vwap_pre_macro_event_imp2")},
    ret_from_vwap_post_macro_event_imp2={"base_col": source("vwap_post_macro_event_imp2"),"col_name":value("vwap_post_macro_event_imp2")},
    ret_from_vwap_around_macro_event_imp2={"base_col": source("vwap_around_macro_event_imp2"),"col_name":value("vwap_around_macro_event_imp2")},
    ret_from_vwap_pre_macro_event_imp3={"base_col": source("vwap_pre_macro_event_imp3"),"col_name":value("vwap_pre_macro_event_imp3")},
    ret_from_vwap_post_macro_event_imp3={"base_col": source("vwap_post_macro_event_imp3"),"col_name":value("vwap_post_macro_event_imp3")},
    ret_from_vwap_around_macro_event_imp3={"base_col": source("vwap_around_macro_event_imp3"),"col_name":value("vwap_around_macro_event_imp3")},
    ret_from_vwap_pre_new_york_open={"base_col": source("vwap_pre_new_york_open"),"col_name":value("vwap_pre_new_york_open")},
    ret_from_vwap_post_new_york_open={"base_col": source("vwap_post_new_york_open"),"col_name":value("vwap_post_new_york_close")},
    ret_from_vwap_pre_new_york_close={"base_col":source("vwap_pre_new_york_close"),"col_name":value("vwap_pre_new_york_close")},
    ret_from_vwap_post_new_york_close={"base_col":source("vwap_post_new_york_close"),"col_name":value("vwap_post_new_york_close")},
    ret_from_vwap_around_new_york_open={"base_col": source("vwap_around_new_york_open"),"col_name":value("vwap_around_new_york_open")},
    ret_from_vwap_around_new_york_close={"base_col": source("vwap_around_new_york_close"),"col_name":value("vwap_around_new_york_close")},

    ret_from_vwap_pre_london_open={"base_col": source("vwap_pre_london_open"),"col_name":value("vwap_pre_london_open")},
    ret_from_vwap_around_london_open={"base_col": source("vwap_around_london_open"),"col_name":value("vwap_around_london_open")},
    ret_from_vwap_post_london_open={"base_col": source("vwap_post_london_open"),"col_name":value("vwap_post_london_open")},
    ret_from_vwap_pre_london_close={"base_col": source("vwap_pre_london_close"),"col_name":value("vwap_pre_london_close")},
    ret_from_vwap_around_london_close={"base_col": source("vwap_around_london_close"),"col_name":value("vwap_around_london_close")},
    ret_from_vwap_post_london_close={"base_col": source("vwap_post_london_close"),"col_name":value("vwap_post_london_close")},

    ret_from_last_daily_close_0={"base_col": source("last_daily_close"),"col_name":value("last_daily_close")},
    ret_from_last_daily_close_1={"base_col": source("last_daily_close_1"),"col_name":value("last_daily_close_1")},
    ret_from_last_daily_close_2={"base_col": source("last_daily_close_2"),"col_name":value("last_daily_close_2")},
    ret_from_last_daily_close_3={"base_col": source("last_daily_close_3"),"col_name":value("last_daily_close_3")},
    ret_from_last_daily_close_4={"base_col": source("last_daily_close_4"),"col_name":value("last_daily_close_4")},
    ret_from_last_daily_close_5={"base_col": source("last_daily_close_5"),"col_name":value("last_daily_close_5")},
    ret_from_last_daily_close_6={"base_col": source("last_daily_close_6"),"col_name":value("last_daily_close_6")},
    ret_from_last_daily_close_7={"base_col": source("last_daily_close_7"),"col_name":value("last_daily_close_7")},
    ret_from_last_daily_close_8={"base_col": source("last_daily_close_8"),"col_name":value("last_daily_close_8")},
    ret_from_last_daily_close_9={"base_col": source("last_daily_close_9"),"col_name":value("last_daily_close_9")},
    ret_from_last_daily_close_10={"base_col": source("last_daily_close_10"),"col_name":value("last_daily_close_10")},
    ret_from_last_daily_close_11={"base_col": source("last_daily_close_11"),"col_name":value("last_daily_close_11")},
    ret_from_last_daily_close_12={"base_col": source("last_daily_close_12"),"col_name":value("last_daily_close_12")},
    ret_from_last_daily_close_13={"base_col": source("last_daily_close_13"),"col_name":value("last_daily_close_13")},
    ret_from_last_daily_close_14={"base_col": source("last_daily_close_14"),"col_name":value("last_daily_close_14")},
    ret_from_last_daily_close_15={"base_col": source("last_daily_close_15"),"col_name":value("last_daily_close_15")},
    ret_from_last_daily_close_16={"base_col": source("last_daily_close_16"),"col_name":value("last_daily_close_16")},
    ret_from_last_daily_close_17={"base_col": source("last_daily_close_17"),"col_name":value("last_daily_close_17")},
    ret_from_last_daily_close_18={"base_col": source("last_daily_close_18"),"col_name":value("last_daily_close_18")},
    ret_from_last_daily_close_19={"base_col": source("last_daily_close_19"),"col_name":value("last_daily_close_19")},

    ret_from_last_weekly_close_0={"base_col": source("last_weekly_close"),"col_name":value("last_weekly_close")},
    ret_from_last_weekly_close_1={"base_col": source("last_weekly_close_1"),"col_name":value("last_weekly_close_1")},
    ret_from_last_weekly_close_2={"base_col": source("last_weekly_close_2"),"col_name":value("last_weekly_close_2")},
    ret_from_last_weekly_close_3={"base_col": source("last_weekly_close_3"),"col_name":value("last_weekly_close_3")},
    ret_from_last_weekly_close_4={"base_col": source("last_weekly_close_4"),"col_name":value("last_weekly_close_4")},
    ret_from_last_weekly_close_5={"base_col": source("last_weekly_close_5"),"col_name":value("last_weekly_close_5")},
    ret_from_last_weekly_close_6={"base_col": source("last_weekly_close_6"),"col_name":value("last_weekly_close_6")},
    ret_from_last_weekly_close_7={"base_col": source("last_weekly_close_7"),"col_name":value("last_weekly_close_7")},
    ret_from_last_weekly_close_8={"base_col": source("last_weekly_close_8"),"col_name":value("last_weekly_close_8")},
    ret_from_last_weekly_close_9={"base_col": source("last_weekly_close_9"),"col_name":value("last_weekly_close_9")},
    ret_from_last_weekly_close_10={"base_col": source("last_weekly_close_10"),"col_name":value("last_weekly_close_10")},
    ret_from_last_weekly_close_11={"base_col": source("last_weekly_close_11"),"col_name":value("last_weekly_close_11")},
    ret_from_last_weekly_close_12={"base_col": source("last_weekly_close_12"),"col_name":value("last_weekly_close_12")},
    ret_from_last_weekly_close_13={"base_col": source("last_weekly_close_13"),"col_name":value("last_weekly_close_13")},
    ret_from_last_weekly_close_14={"base_col": source("last_weekly_close_14"),"col_name":value("last_weekly_close_14")},
    ret_from_last_weekly_close_15={"base_col": source("last_weekly_close_15"),"col_name":value("last_weekly_close_15")},
    ret_from_last_weekly_close_16={"base_col": source("last_weekly_close_16"),"col_name":value("last_weekly_close_16")},
    ret_from_last_weekly_close_17={"base_col": source("last_weekly_close_17"),"col_name":value("last_weekly_close_17")},
    ret_from_last_weekly_close_18={"base_col": source("last_weekly_close_18"),"col_name":value("last_weekly_close_18")},
    ret_from_last_weekly_close_19={"base_col": source("last_weekly_close_19"),"col_name":value("last_weekly_close_19")},

    ret_from_last_monthly_close_0={"base_col": source("last_monthly_close"),"col_name":value("last_monthly_close")},
    ret_from_last_monthly_close_1={"base_col": source("last_monthly_close_1"),"col_name":value("last_monthly_close_1")},
    ret_from_last_monthly_close_2={"base_col": source("last_monthly_close_2"),"col_name":value("last_monthly_close_2")},
    ret_from_last_monthly_close_3={"base_col": source("last_monthly_close_3"),"col_name":value("last_monthly_close_3")},
    ret_from_last_monthly_close_4={"base_col": source("last_monthly_close_4"),"col_name":value("last_monthly_close_4")},
    ret_from_last_monthly_close_5={"base_col": source("last_monthly_close_5"),"col_name":value("last_monthly_close_5")},
    ret_from_last_monthly_close_6={"base_col": source("last_monthly_close_6"),"col_name":value("last_monthly_close_6")},
    ret_from_last_monthly_close_7={"base_col": source("last_monthly_close_7"),"col_name":value("last_monthly_close_7")},
    ret_from_last_monthly_close_8={"base_col": source("last_monthly_close_8"),"col_name":value("last_monthly_close_8")},
    ret_from_last_monthly_close_9={"base_col": source("last_monthly_close_9"),"col_name":value("last_monthly_close_9")},
    ret_from_last_monthly_close_10={"base_col": source("last_monthly_close_10"),"col_name":value("last_monthly_close_10")},
    ret_from_last_monthly_close_11={"base_col": source("last_monthly_close_11"),"col_name":value("last_monthly_close_11")},
    ret_from_last_monthly_close_12={"base_col": source("last_monthly_close_12"),"col_name":value("last_monthly_close_12")},
    ret_from_last_monthly_close_13={"base_col": source("last_monthly_close_13"),"col_name":value("last_monthly_close_13")},
    ret_from_last_monthly_close_14={"base_col": source("last_monthly_close_14"),"col_name":value("last_monthly_close_14")},
    ret_from_last_monthly_close_15={"base_col": source("last_monthly_close_15"),"col_name":value("last_monthly_close_15")},
    ret_from_last_monthly_close_16={"base_col": source("last_monthly_close_16"),"col_name":value("last_monthly_close_16")},
    ret_from_last_monthly_close_17={"base_col": source("last_monthly_close_17"),"col_name":value("last_monthly_close_17")},
    ret_from_last_monthly_close_18={"base_col": source("last_monthly_close_18"),"col_name":value("last_monthly_close_18")},
    ret_from_last_monthly_close_19={"base_col": source("last_monthly_close_19"),"col_name":value("last_monthly_close_19")},
    ret_to_next_new_york_close={"base_col": source("next_new_york_close"),"col_name":value("next_new_york_close")},
    ret_to_next_weekly_close={"base_col": source("next_weekly_close"),"col_name":value("next_weekly_close")},
    ret_to_next_monthly_close={"base_col": source("next_monthly_close"),"col_name":value("next_monthly_close")},
    ret_from_bb_high_5_2={"base_col": source("bb_high_5_2"), "col_name":value("bb_high_5_2")},
    ret_from_bb_high_5_3={"base_col": source("bb_high_5_3"), "col_name":value("bb_high_5_3")},
    ret_from_bb_high_10_2={"base_col": source("bb_high_10_2"), "col_name":value("bb_high_10_2")},
    ret_from_bb_high_10_3={"base_col": source("bb_high_10_3"), "col_name":value("bb_high_10_3")},
    ret_from_bb_high_20_2={"base_col": source("bb_high_20_2"), "col_name":value("bb_high_20_2")},
    ret_from_bb_high_20_3={"base_col": source("bb_high_20_3"), "col_name":value("bb_high_20_3")},
    ret_from_bb_high_50_2={"base_col": source("bb_high_50_2"), "col_name":value("bb_high_50_2")},
    ret_from_bb_high_50_3={"base_col": source("bb_high_50_3"), "col_name":value("bb_high_50_3")},
    ret_from_bb_high_100_2={"base_col": source("bb_high_100_2"), "col_name":value("bb_high_100_2")},
    ret_from_bb_high_100_3={"base_col": source("bb_high_100_3"), "col_name":value("bb_high_100_3")},
    ret_from_bb_high_200_2={"base_col": source("bb_high_200_2"), "col_name":value("bb_high_200_2")},
    ret_from_bb_high_200_3={"base_col": source("bb_high_200_3"), "col_name":value("bb_high_200_3")},

    ret_from_bb_low_5_2={"base_col": source("bb_low_5_2"), "col_name":value("bb_low_5_2")},
    ret_from_bb_low_5_3={"base_col": source("bb_low_5_3"), "col_name":value("bb_low_5_3")},
    ret_from_bb_low_10_2={"base_col": source("bb_low_10_2"), "col_name":value("bb_low_10_2")},
    ret_from_bb_low_10_3={"base_col": source("bb_low_10_3"), "col_name":value("bb_low_10_3")},
    ret_from_bb_low_20_2={"base_col": source("bb_low_20_2"), "col_name":value("bb_low_20_2")},
    ret_from_bb_low_20_3={"base_col": source("bb_low_20_3"), "col_name":value("bb_low_20_3")},
    ret_from_bb_low_50_2={"base_col": source("bb_low_50_2"), "col_name":value("bb_low_50_2")},
    ret_from_bb_low_50_3={"base_col": source("bb_low_50_3"), "col_name":value("bb_low_50_3")},
    ret_from_bb_low_100_2={"base_col": source("bb_low_100_2"), "col_name":value("bb_low_100_2")},
    ret_from_bb_low_100_3={"base_col": source("bb_low_100_3"), "col_name":value("bb_low_100_3")},
    ret_from_bb_low_200_2={"base_col": source("bb_low_200_2"), "col_name":value("bb_low_200_2")},
    ret_from_bb_low_200_3={"base_col": source("bb_low_200_3"), "col_name":value("bb_low_200_3")},

    ret_from_bb_high_5d_2={"base_col": source("bb_high_5d_2"), "col_name":value("bb_high_5d_2")},
    ret_from_bb_high_5d_3={"base_col": source("bb_high_5d_3"), "col_name":value("bb_high_5d_3")},
    ret_from_bb_high_10d_2={"base_col": source("bb_high_10d_2"), "col_name":value("bb_high_10d_2")},
    ret_from_bb_high_10d_3={"base_col": source("bb_high_10d_3"), "col_name":value("bb_high_10d_3")},
    ret_from_bb_high_20d_2={"base_col": source("bb_high_20d_2"), "col_name":value("bb_high_20d_2")},
    ret_from_bb_high_20d_3={"base_col": source("bb_high_20d_3"), "col_name":value("bb_high_20d_3")},
    ret_from_bb_high_50d_2={"base_col": source("bb_high_50d_2"), "col_name":value("bb_high_50d_2")},
    ret_from_bb_high_50d_3={"base_col": source("bb_high_50d_3"), "col_name":value("bb_high_50d_3")},
    ret_from_bb_high_100d_2={"base_col": source("bb_high_100d_2"), "col_name":value("bb_high_100d_2")},
    ret_from_bb_high_100d_3={"base_col": source("bb_high_100d_3"), "col_name":value("bb_high_100d_3")},
    ret_from_bb_high_200d_2={"base_col": source("bb_high_200d_2"), "col_name":value("bb_high_200d_2")},
    ret_from_bb_high_200d_3={"base_col": source("bb_high_200d_3"), "col_name":value("bb_high_200d_3")},

    ret_from_bb_low_5d_2={"base_col": source("bb_low_5d_2"), "col_name":value("bb_low_5d_3")},
    ret_from_bb_low_5d_3={"base_col": source("bb_low_5d_3"), "col_name":value("bb_low_5d_3")},
    ret_from_bb_low_10d_2={"base_col": source("bb_low_10d_2"), "col_name":value("bb_low_10d_2")},
    ret_from_bb_low_10d_3={"base_col": source("bb_low_10d_3"), "col_name":value("bb_low_10d_3")},
    ret_from_bb_low_20d_2={"base_col": source("bb_low_20d_2"), "col_name":value("bb_low_20d_2")},
    ret_from_bb_low_20d_3={"base_col": source("bb_low_20d_3"), "col_name":value("bb_low_20d_3")},
    ret_from_bb_low_50d_2={"base_col": source("bb_low_50d_2"), "col_name":value("bb_low_50d_2")},
    ret_from_bb_low_50d_3={"base_col": source("bb_low_50d_3"), "col_name":value("bb_low_50d_3")},
    ret_from_bb_low_100d_2={"base_col": source("bb_low_100d_2"), "col_name":value("bb_low_100d_2")},
    ret_from_bb_low_100d_3={"base_col": source("bb_low_100d_3"), "col_name":value("bb_low_100d_3")},
    ret_from_bb_low_200d_2={"base_col": source("bb_low_200d_2"), "col_name":value("bb_low_200d_2")},
    ret_from_bb_low_200d_3={"base_col": source("bb_low_200d_3"), "col_name":value("bb_low_200d_3")},
    ret_from_sma_5={"base_col": source("sma_5"), "col_name":value("sma_5")},
    ret_from_sma_10={"base_col": source("sma_10"), "col_name":value("sma_10")},
    ret_from_sma_20={"base_col": source("sma_20"), "col_name":value("sma_20")},
    ret_from_sma_50={"base_col": source("sma_50"), "col_name":value("sma_50")},
    ret_from_sma_100={"base_col": source("sma_100"), "col_name":value("sma_100")},
    ret_from_sma_200={"base_col": source("sma_200"), "col_name":value("sma_200")},
    ret_from_sma_5d={"base_col": source("sma_5d"), "col_name":value("sma_5d")},
    ret_from_sma_10d={"base_col": source("sma_10d"), "col_name":value("sma_10d")},
    ret_from_sma_20d={"base_col": source("sma_20d"), "col_name":value("sma_20d")},
    ret_from_sma_50d={"base_col": source("sma_50d"), "col_name":value("sma_50d")},
    ret_from_sma_100d={"base_col": source("sma_100d"), "col_name":value("sma_100d")},
    ret_from_sma_200d={"base_col": source("sma_200d"), "col_name":value("sma_200d")},
)
def ret_from_price(close: pd.Series, base_col: pd.Series, base_price:float, col_name:str) -> pd.Series:
    #logging.error(f"ret_from_price_close, col:col_name:{col_name}, close:{close.iloc[50:100]}")
    #logging.error(f"ret_from_price_close, col:col_name:{col_name}, close_index:{close.index[50:100]}")
    #logging.error(f"ret_from_price_base_col:col_name:{col_name}, before reindex base_col:{base_col.iloc[50:100]}")
    #logging.error(f"ret_from_price_base_col:col_name:{col_name}, before reindex base_col_index:{base_col.index[50:100]}")
    #base_col = base_col.reset_index()
    #logging.error(f"ret_from_price_base_col:col_name:{col_name}, after reset base_col:{base_col.iloc[50:100]}")
    #logging.error(f"ret_from_price_base_col:col_name:{col_name}, base_col_type:{type(base_col)}")
    #base_col.set_index(["timestamp","ticker"], inplace = True)
    #logging.error(f"ret_from_price_base_col:col_name:{col_name}, after reindex base_col:{base_col.iloc[50:100]}")
    #logging.error(f"ret_from_price_base_col:col_name:{col_name}, after reindex base_col_index:{base_col.index[50:100]}")
    #logging.error(f"ret_from_price_base_col: base_col_type:{type(base_col)}")
    base_series = None
    if isinstance(base_col, (pd.Series)) or not "close" in base_col.columns:
        base_series = base_col
    else:
        base_series = base_col["close"]
    df = pd.concat([close, base_series], axis=1)
    #logging.error(f"ret_from_price_base_col:col_name:{col_name}, df:{df.iloc[50:100]}")
    df.columns = ["close", "diff_close"]
    #logging.error(f"ret_from_price_base_col:col_name:{col_name}, df:{df.iloc[50:100]}")
    series = df.apply(lambda x: math.log(x["close"]+base_price)-math.log(x["diff_close"]+base_price), axis=1)
    #logging.error(f"ret_from_price_base_col:col_name:{col_name}, series:{series}")
    return series

@parameterize(
    ret_velocity_from_high_5={"ret_col": source("ret_from_high_5"), "time_col": source("time_to_high_5_ff"), "col_name":value("time_to_high_5_ff")},
    ret_velocity_from_high_11={"ret_col": source("ret_from_high_11"), "time_col": source("time_to_high_11_ff"), "col_name":value("time_to_high_11_ff")},
    ret_velocity_from_high_21={"ret_col": source("ret_from_high_21"), "time_col": source("time_to_high_21_ff"), "col_name":value("time_to_high_21_ff")},
    ret_velocity_from_high_51={"ret_col": source("ret_from_high_51"), "time_col": source("time_to_high_51_ff"), "col_name":value("time_to_high_51_ff")},
    ret_velocity_from_high_101={"ret_col": source("ret_from_high_101"), "time_col": source("time_to_high_101_ff"), "col_name":value("time_to_high_101_ff")},
    ret_velocity_from_high_201={"ret_col": source("ret_from_high_201"), "time_col": source("time_to_high_201_ff"), "col_name":value("time_to_high_201_ff")},
    ret_velocity_from_low_5={"ret_col": source("ret_from_low_5"), "time_col": source("time_to_low_5_ff"), "col_name":value("time_to_low_5_ff")},
    ret_velocity_from_low_11={"ret_col": source("ret_from_low_11"), "time_col": source("time_to_low_11_ff"), "col_name":value("time_to_low_11_ff")},
    ret_velocity_from_low_21={"ret_col": source("ret_from_low_21"), "time_col": source("time_to_low_21_ff"), "col_name":value("time_to_low_21_ff")},
    ret_velocity_from_low_51={"ret_col": source("ret_from_low_51"), "time_col": source("time_to_low_51_ff"), "col_name":value("time_to_low_51_ff")},
    ret_velocity_from_low_101={"ret_col": source("ret_from_low_101"), "time_col": source("time_to_low_101_ff"), "col_name":value("time_to_low_101_ff")},
    ret_velocity_from_low_201={"ret_col": source("ret_from_low_201"), "time_col": source("time_to_low_201_ff"), "col_name":value("time_to_low_201_ff")},
)
def ret_velocity_tmpl(ret_col: pd.Series, time_col: pd.Series, col_name:str) -> pd.Series:
    logging.error(f"ret_velocity_tmpl_col_name:{col_name}, ret_col:{ret_col}")
    logging.error(f"ret_velocity_tmpl_col_name:{col_name}, time_col:{time_col}")
    return ret_col/time_col
    
@parameterize(
    daily_returns_1d={"day_offset":value(1)},
    daily_returns_5d={"day_offset":value(5)},
    daily_returns_10d={"day_offset":value(10)},
    daily_returns_20d={"day_offset":value(20)}
)
def daily_returns_day_tmpl(close:pd.Series, base_price:float, day_offset:int, interval_per_day:int) -> pd.Series:
    return calc_returns(close, day_offset=day_offset*interval_per_day, base_price=base_price)

@parameterize(
    daily_returns={"day_offset":value(1)},
    daily_returns_5={"day_offset":value(5)},
    daily_returns_10={"day_offset":value(10)},
    daily_returns_20={"day_offset":value(20)}
)
def daily_returns_tmpl(close:pd.Series, base_price:float, day_offset:int) -> pd.Series:
    return calc_returns(close, day_offset=day_offset, base_price=base_price)

@parameterize(
    daily_vol={"return_col":source("daily_returns")},
    daily_vol_5={"return_col":source("daily_returns_5")},
    daily_vol_10={"return_col":source("daily_returns_10")},
    daily_vol_20={"return_col":source("daily_returns_20")},
    daily_vol_1d={"return_col":source("daily_returns_1d")},
    daily_vol_5d={"return_col":source("daily_returns_5d")},
    daily_vol_10d={"return_col":source("daily_returns_10d")},
    daily_vol_20d={"return_col":source("daily_returns_20d")},
)
def daily_vol_tmpl(return_col:pd.Series) -> pd.Series:
    return calc_daily_vol(return_col)

@parameterize(
    daily_skew={"return_col":source("daily_returns")},
    daily_skew_5={"return_col":source("daily_returns_5")},
    daily_skew_10={"return_col":source("daily_returns_10")},
    daily_skew_20={"return_col":source("daily_returns_20")},
    daily_skew_1d={"return_col":source("daily_returns_1d")},
    daily_skew_5d={"return_col":source("daily_returns_5d")},
    daily_skew_10d={"return_col":source("daily_returns_10d")},
    daily_skew_20d={"return_col":source("daily_returns_20d")},
)
def daily_skew_tmpl(return_col:pd.Series) -> pd.Series:
    return calc_skew(return_col)

@parameterize(
    daily_kurt={"return_col":source("daily_returns")},
    daily_kurt_5={"return_col":source("daily_returns_5")},
    daily_kurt_10={"return_col":source("daily_returns_10")},
    daily_kurt_20={"return_col":source("daily_returns_20")},
    daily_kurt_1d={"return_col":source("daily_returns_1d")},
    daily_kurt_5d={"return_col":source("daily_returns_5d")},
    daily_kurt_10d={"return_col":source("daily_returns_10d")},
    daily_kurt_20d={"return_col":source("daily_returns_20d")},
)
def daily_kurt_tmpl(return_col:pd.Series) -> pd.Series:
    return calc_kurt(return_col)

#@profile_util.profile
def example_group_features(cal:CMEEquityExchangeCalendar, macro_data_builder:MacroDataBuilder,
                           base_price:float,
                           clean_sorted_data:pd.DataFrame,
                           config:DictConfig,
                           daily_returns:pd.Series, daily_returns_5:pd.Series,daily_returns_10:pd.Series,daily_returns_20:pd.Series,
                           daily_vol:pd.Series,daily_vol_5:pd.Series,daily_vol_10:pd.Series,daily_vol_20:pd.Series,
                           daily_skew:pd.Series,daily_skew_5:pd.Series,daily_skew_10:pd.Series,daily_skew_20:pd.Series,
                           daily_kurt:pd.Series,daily_kurt_5:pd.Series,daily_kurt_10:pd.Series,daily_kurt_20:pd.Series,
                           daily_returns_1d:pd.Series, daily_returns_5d:pd.Series,daily_returns_10d:pd.Series,daily_returns_20d:pd.Series,
                           daily_vol_1d:pd.Series,daily_vol_5d:pd.Series,daily_vol_10d:pd.Series,daily_vol_20d:pd.Series,
                           daily_skew_1d:pd.Series,daily_skew_5d:pd.Series,daily_skew_10d:pd.Series,daily_skew_20d:pd.Series,
                           daily_kurt_1d:pd.Series,daily_kurt_5d:pd.Series,daily_kurt_10d:pd.Series,daily_kurt_20d:pd.Series,
                           ticker: pd.Series,
                           close: pd.Series,
                           close_velocity_back: pd.Series,
                           trimmed_close_back: pd.Series,
                           volume_back: pd.Series,
                           trimmed_high_back: pd.Series,
                           trimmed_low_back: pd.Series,
                           trimmed_open_back: pd.Series,
                           timestamp: pd.Series,
                           time: pd.Series,
                           month: pd.Series, year:pd.Series, hour_of_day:pd.Series,
                           time_to_low_1d_ff_shift_1d: pd.Series, time_to_low_5d_ff_shift_5d: pd.Series,
                           time_to_low_11d_ff_shift_11d: pd.Series, time_to_low_21d_ff_shift_21d: pd.Series,
                           time_to_high_1d_ff_shift_1d: pd.Series, time_to_high_5d_ff_shift_5d: pd.Series,
                           time_to_high_11d_ff_shift_11d: pd.Series, time_to_high_21d_ff_shift_21d: pd.Series,
                           time_to_low_5_ff: pd.Series,
                           time_to_low_11_ff: pd.Series, time_to_low_21_ff: pd.Series,
                           time_to_low_51_ff: pd.Series, time_to_low_101_ff: pd.Series,
                           time_to_low_201_ff: pd.Series,
                           time_to_high_5_ff: pd.Series,
                           time_to_high_11_ff: pd.Series, time_to_high_21_ff: pd.Series,
                           time_to_high_51_ff: pd.Series, time_to_high_101_ff: pd.Series,
                           time_to_high_201_ff: pd.Series,
                           day_of_week:pd.Series, day_of_month:pd.Series,
                           time_to_last_macro_event_imp1: pd.Series, time_to_next_macro_event_imp1: pd.Series,
                           time_to_last_macro_event_imp2: pd.Series, time_to_next_macro_event_imp2: pd.Series,
                           time_to_last_macro_event_imp3: pd.Series, time_to_next_macro_event_imp3: pd.Series,
                           time_to_low_1d_ff: pd.Series, time_to_low_5d_ff: pd.Series,
                           time_to_low_11d_ff: pd.Series, time_to_low_21d_ff: pd.Series,
                           time_to_low_51d_ff: pd.Series, time_to_low_101d_ff: pd.Series,
                           time_to_low_201d_ff: pd.Series,
                           time_to_high_1d_ff: pd.Series, time_to_high_5d_ff: pd.Series,
                           time_to_high_11d_ff: pd.Series, time_to_high_21d_ff: pd.Series,
                           time_to_high_51d_ff: pd.Series, time_to_high_101d_ff: pd.Series,
                           time_to_high_201d_ff: pd.Series,
                           time_to_weekly_close:pd.Series, time_to_monthly_close:pd.Series,
                           time_to_option_expiration:pd.Series,
                           time_to_new_york_open:pd.Series,
                           time_to_new_york_last_open:pd.Series,
                           time_to_new_york_last_close:pd.Series,
                           time_to_new_york_close:pd.Series,
                           time_to_london_open:pd.Series,
                           time_to_london_last_open:pd.Series,
                           time_to_london_close:pd.Series,
                           time_to_london_last_close:pd.Series,
                           rsi_14: pd.Series,
                           rsi_28: pd.Series,
                           rsi_42: pd.Series,
                           rsi_14d: pd.Series,
                           rsi_28d: pd.Series,
                           rsi_42d: pd.Series,
                           ret_from_high_1d: pd.Series,
                           ret_from_high_5d: pd.Series,
                           ret_from_high_11d: pd.Series,
                           ret_from_high_21d: pd.Series,
                           ret_from_high_51d: pd.Series,
                           ret_from_high_101d: pd.Series,
                           ret_from_high_201d: pd.Series,
                           ret_from_low_1d: pd.Series,
                           ret_from_low_5d: pd.Series,
                           ret_from_low_11d: pd.Series,
                           ret_from_low_21d: pd.Series,
                           ret_from_low_51d: pd.Series,
                           ret_from_low_101d: pd.Series,
                           ret_from_low_201d: pd.Series,
                           ret_from_high_1d_shift_1d: pd.Series,
                           ret_from_high_5d_shift_5d: pd.Series,
                           ret_from_high_11d_shift_11d: pd.Series,
                           ret_from_high_21d_shift_21d: pd.Series,
                           ret_from_low_1d_shift_1d: pd.Series,
                           ret_from_low_5d_shift_5d: pd.Series,
                           ret_from_low_11d_shift_11d: pd.Series,
                           ret_from_low_21d_shift_21d: pd.Series,
                           ret_from_high_5d_bf: pd.Series,
                           ret_from_high_11d_bf: pd.Series,
                           ret_from_high_21d_bf: pd.Series,
                           ret_from_high_51d_bf: pd.Series,
                           ret_from_low_5d_bf: pd.Series,
                           ret_from_low_11d_bf: pd.Series,
                           ret_from_low_21d_bf: pd.Series,
                           ret_from_low_51d_bf: pd.Series,
                           weekly_close_time: pd.Series,
                           monthly_close_time: pd.Series,
                           last_weekly_close_time: pd.Series,
                           last_weekly_close_time_0: pd.Series,
                           last_weekly_close_time_1: pd.Series,
                           last_weekly_close_time_2: pd.Series,
                           last_weekly_close_time_3: pd.Series,
                           last_weekly_close_time_4: pd.Series,
                           last_weekly_close_time_5: pd.Series,
                           last_weekly_close_time_6: pd.Series,
                           last_weekly_close_time_7: pd.Series,
                           last_weekly_close_time_8: pd.Series,
                           last_weekly_close_time_9: pd.Series,
                           last_weekly_close_time_10: pd.Series,
                           last_weekly_close_time_11: pd.Series,
                           last_weekly_close_time_12: pd.Series,
                           last_weekly_close_time_13: pd.Series,
                           last_weekly_close_time_14: pd.Series,
                           last_weekly_close_time_15: pd.Series,
                           last_weekly_close_time_16: pd.Series,
                           last_weekly_close_time_17: pd.Series,
                           last_weekly_close_time_18: pd.Series,
                           last_weekly_close_time_19: pd.Series,
                           last_monthly_close_time: pd.Series,
                           last_monthly_close_time_0: pd.Series,
                           last_monthly_close_time_1: pd.Series,
                           last_monthly_close_time_2: pd.Series,
                           last_monthly_close_time_3: pd.Series,
                           last_monthly_close_time_4: pd.Series,
                           last_monthly_close_time_5: pd.Series,
                           last_monthly_close_time_6: pd.Series,
                           last_monthly_close_time_7: pd.Series,
                           last_monthly_close_time_8: pd.Series,
                           last_monthly_close_time_9: pd.Series,
                           last_monthly_close_time_10: pd.Series,
                           last_monthly_close_time_11: pd.Series,
                           last_monthly_close_time_12: pd.Series,
                           last_monthly_close_time_13: pd.Series,
                           last_monthly_close_time_14: pd.Series,
                           last_monthly_close_time_15: pd.Series,
                           last_monthly_close_time_16: pd.Series,
                           last_monthly_close_time_17: pd.Series,
                           last_monthly_close_time_18: pd.Series,
                           last_monthly_close_time_19: pd.Series,
                           ret_from_close_cumsum_high_5d: pd.Series,
                           ret_from_close_cumsum_high_11d: pd.Series,
                           ret_from_close_cumsum_high_21d: pd.Series,
                           ret_from_close_cumsum_high_51d: pd.Series,
                           ret_from_close_cumsum_high_101d: pd.Series,
                           ret_from_close_cumsum_high_201d: pd.Series,
                           ret_from_close_cumsum_low_5d: pd.Series,
                           ret_from_close_cumsum_low_11d: pd.Series,
                           ret_from_close_cumsum_low_21d: pd.Series,
                           ret_from_close_cumsum_low_51d: pd.Series,
                           ret_from_close_cumsum_low_101d: pd.Series,
                           ret_from_close_cumsum_low_201d: pd.Series,
                           ret_from_close_cumsum_high_5: pd.Series,
                           ret_from_close_cumsum_high_11: pd.Series,
                           ret_from_close_cumsum_high_21: pd.Series,
                           ret_from_close_cumsum_high_51: pd.Series,
                           ret_from_close_cumsum_high_101: pd.Series,
                           ret_from_close_cumsum_high_201: pd.Series,
                           ret_from_close_cumsum_low_5: pd.Series,
                           ret_from_close_cumsum_low_11: pd.Series,
                           ret_from_close_cumsum_low_21: pd.Series,
                           ret_from_close_cumsum_low_51: pd.Series,
                           ret_from_close_cumsum_low_101: pd.Series,
                           ret_from_close_cumsum_low_201: pd.Series,
                           ret_from_last_weekly_close_0: pd.Series,
                           ret_from_last_weekly_close_1: pd.Series,
                           ret_from_last_weekly_close_2: pd.Series,
                           ret_from_last_weekly_close_3: pd.Series,
                           ret_from_last_weekly_close_4: pd.Series,
                           ret_from_last_weekly_close_5: pd.Series,
                           ret_from_last_weekly_close_6: pd.Series,
                           ret_from_last_weekly_close_7: pd.Series,
                           ret_from_last_weekly_close_8: pd.Series,
                           ret_from_last_weekly_close_9: pd.Series,
                           ret_from_last_weekly_close_10: pd.Series,
                           ret_from_last_weekly_close_11: pd.Series,
                           ret_from_last_weekly_close_12: pd.Series,
                           ret_from_last_weekly_close_13: pd.Series,
                           ret_from_last_weekly_close_14: pd.Series,
                           ret_from_last_weekly_close_15: pd.Series,
                           ret_from_last_weekly_close_16: pd.Series,
                           ret_from_last_weekly_close_17: pd.Series,
                           ret_from_last_weekly_close_18: pd.Series,
                           ret_from_last_weekly_close_19: pd.Series,
                           ret_from_last_monthly_close_0: pd.Series,
                           ret_from_last_monthly_close_1: pd.Series,
                           ret_from_last_monthly_close_2: pd.Series,
                           ret_from_last_monthly_close_3: pd.Series,
                           ret_from_last_monthly_close_4: pd.Series,
                           ret_from_last_monthly_close_5: pd.Series,
                           ret_from_last_monthly_close_6: pd.Series,
                           ret_from_last_monthly_close_7: pd.Series,
                           ret_from_last_monthly_close_8: pd.Series,
                           ret_from_last_monthly_close_9: pd.Series,
                           ret_from_last_monthly_close_10: pd.Series,
                           ret_from_last_monthly_close_11: pd.Series,
                           ret_from_last_monthly_close_12: pd.Series,
                           ret_from_last_monthly_close_13: pd.Series,
                           ret_from_last_monthly_close_14: pd.Series,
                           ret_from_last_monthly_close_15: pd.Series,
                           ret_from_last_monthly_close_16: pd.Series,
                           ret_from_last_monthly_close_17: pd.Series,
                           ret_from_last_monthly_close_18: pd.Series,
                           ret_from_last_monthly_close_19: pd.Series,
                           ret_from_vwap_since_last_macro_event_imp1: pd.Series,
                           ret_from_vwap_since_last_macro_event_imp2: pd.Series,
                           ret_from_vwap_since_last_macro_event_imp3: pd.Series,
                           vwap_pre_macro_event_imp1: pd.Series,
                           vwap_post_macro_event_imp1: pd.Series,
                           vwap_around_macro_event_imp1: pd.Series,
                           vwap_pre_macro_event_imp2: pd.Series,
                           vwap_post_macro_event_imp2: pd.Series,
                           vwap_around_macro_event_imp2: pd.Series,
                           vwap_pre_macro_event_imp3: pd.Series,
                           vwap_post_macro_event_imp3: pd.Series,
                           vwap_around_macro_event_imp3: pd.Series,
                           week_of_year: pd.Series,
                           month_of_year: pd.Series,
                           vwap_around_new_york_open: pd.Series,
                           vwap_pre_new_york_open: pd.Series,
                           vwap_post_new_york_open: pd.Series,
                           vwap_around_new_york_close: pd.Series,
                           vwap_pre_new_york_close: pd.Series,
                           vwap_post_new_york_close: pd.Series,
                           vwap_around_london_open: pd.Series,
                           vwap_pre_london_open: pd.Series,
                           vwap_post_london_open: pd.Series,
                           vwap_around_london_close: pd.Series,
                           vwap_pre_london_close: pd.Series,
                           last_macro_event_time_imp1: pd.Series,
                           last_macro_event_time_imp2: pd.Series,
                           last_macro_event_time_imp3: pd.Series,
                           next_macro_event_time_imp1: pd.Series,
                           next_macro_event_time_imp2: pd.Series,
                           next_macro_event_time_imp3: pd.Series,
                           ret_from_high_5: pd.Series,
                           ret_from_high_11: pd.Series,
                           ret_from_high_21: pd.Series,
                           ret_from_high_51: pd.Series,
                           ret_from_high_101: pd.Series,
                           ret_from_high_201: pd.Series,
                           ret_from_low_5: pd.Series,
                           ret_from_low_11: pd.Series,
                           ret_from_low_21: pd.Series,
                           ret_from_low_51: pd.Series,
                           ret_from_low_101: pd.Series,
                           ret_from_low_201: pd.Series,
                           ret_velocity_from_high_5: pd.Series,
                           ret_velocity_from_high_11: pd.Series,
                           ret_velocity_from_high_21: pd.Series,
                           ret_velocity_from_high_51: pd.Series,
                           ret_velocity_from_high_101: pd.Series,
                           ret_velocity_from_high_201: pd.Series,
                           ret_velocity_from_low_5: pd.Series,
                           ret_velocity_from_low_11: pd.Series,
                           ret_velocity_from_low_21: pd.Series,
                           ret_velocity_from_low_51: pd.Series,
                           ret_velocity_from_low_101: pd.Series,
                           ret_velocity_from_low_201: pd.Series,
                           ret_from_vwap_pre_macro_event_imp1: pd.Series,
                           ret_from_vwap_post_macro_event_imp1: pd.Series,
                           ret_from_vwap_around_macro_event_imp1: pd.Series,
                           ret_from_vwap_pre_macro_event_imp2: pd.Series,
                           ret_from_vwap_post_macro_event_imp2: pd.Series,
                           ret_from_vwap_around_macro_event_imp2: pd.Series,
                           ret_from_vwap_pre_macro_event_imp3: pd.Series,
                           ret_from_vwap_post_macro_event_imp3: pd.Series,
                           ret_from_vwap_around_macro_event_imp3: pd.Series,
                           ret_from_vwap_since_new_york_open: pd.Series,
                           ret_from_vwap_pre_new_york_open: pd.Series,
                           ret_from_vwap_post_new_york_open: pd.Series,
                           ret_from_vwap_around_new_york_open: pd.Series,
                           ret_from_vwap_pre_new_york_close: pd.Series,
                           ret_from_vwap_around_new_york_close: pd.Series,
                           ret_from_vwap_post_new_york_close: pd.Series,
                           ret_from_vwap_since_london_open: pd.Series,
                           ret_from_vwap_around_london_open: pd.Series,
                           ret_from_vwap_post_london_open: pd.Series,
                           ret_from_vwap_pre_london_open: pd.Series,
                           ret_from_vwap_around_london_close: pd.Series,
                           ret_from_vwap_post_london_close: pd.Series,
                           ret_from_vwap_pre_london_close: pd.Series,
                           ret_from_last_daily_close_0: pd.Series,
                           ret_from_last_daily_close_1: pd.Series,
                           ret_from_last_daily_close_2: pd.Series,
                           ret_from_last_daily_close_3: pd.Series,
                           ret_from_last_daily_close_4: pd.Series,
                           ret_from_last_daily_close_5: pd.Series,
                           ret_from_last_daily_close_6: pd.Series,
                           ret_from_last_daily_close_7: pd.Series,
                           ret_from_last_daily_close_8: pd.Series,
                           ret_from_last_daily_close_9: pd.Series,
                           ret_from_last_daily_close_10: pd.Series,
                           ret_from_last_daily_close_11: pd.Series,
                           ret_from_last_daily_close_12: pd.Series,
                           ret_from_last_daily_close_13: pd.Series,
                           ret_from_last_daily_close_14: pd.Series,
                           ret_from_last_daily_close_15: pd.Series,
                           ret_from_last_daily_close_16: pd.Series,
                           ret_from_last_daily_close_17: pd.Series,
                           ret_from_last_daily_close_18: pd.Series,
                           ret_from_last_daily_close_19: pd.Series,
                           ret_from_sma_5: pd.Series,
                           ret_from_sma_10: pd.Series,
                           ret_from_sma_20: pd.Series,
                           ret_from_sma_50: pd.Series,
                           ret_from_sma_100: pd.Series,
                           ret_from_sma_200: pd.Series,
                           ret_from_sma_5d: pd.Series,
                           ret_from_sma_10d: pd.Series,
                           ret_from_sma_20d: pd.Series,
                           ret_from_sma_50d: pd.Series,
                           ret_from_sma_100d: pd.Series,
                           ret_from_sma_200d: pd.Series,
                           ret_from_bb_high_5_2: pd.Series,
                           ret_from_bb_high_5_3: pd.Series,
                           ret_from_bb_high_10_2: pd.Series,
                           ret_from_bb_high_10_3: pd.Series,
                           ret_from_bb_high_20_2: pd.Series,
                           ret_from_bb_high_20_3: pd.Series,
                           ret_from_bb_high_50_2: pd.Series,
                           ret_from_bb_high_50_3: pd.Series,
                           ret_from_bb_high_100_2: pd.Series,
                           ret_from_bb_high_100_3: pd.Series,
                           ret_from_bb_high_200_2: pd.Series,
                           ret_from_bb_high_200_3: pd.Series,
                           ret_from_bb_low_5_2: pd.Series,
                           ret_from_bb_low_5_3: pd.Series,
                           ret_from_bb_low_10_2: pd.Series,
                           ret_from_bb_low_10_3: pd.Series,
                           ret_from_bb_low_20_2: pd.Series,
                           ret_from_bb_low_20_3: pd.Series,
                           ret_from_bb_low_50_2: pd.Series,
                           ret_from_bb_low_50_3: pd.Series,
                           ret_from_bb_low_100_2: pd.Series,
                           ret_from_bb_low_100_3: pd.Series,
                           ret_from_bb_low_200_2: pd.Series,
                           ret_from_bb_low_200_3: pd.Series,
                           ret_from_bb_high_5d_2: pd.Series,
                           ret_from_bb_high_5d_3: pd.Series,
                           ret_from_bb_high_10d_2: pd.Series,
                           ret_from_bb_high_10d_3: pd.Series,
                           ret_from_bb_high_20d_2: pd.Series,
                           ret_from_bb_high_20d_3: pd.Series,
                           ret_from_bb_high_50d_2: pd.Series,
                           ret_from_bb_high_50d_3: pd.Series,
                           ret_from_bb_high_100d_2: pd.Series,
                           ret_from_bb_high_100d_3: pd.Series,
                           ret_from_bb_high_200d_2: pd.Series,
                           ret_from_bb_high_200d_3: pd.Series,
                           ret_from_bb_low_5d_2: pd.Series,
                           ret_from_bb_low_5d_3: pd.Series,
                           ret_from_bb_low_10d_2: pd.Series,
                           ret_from_bb_low_10d_3: pd.Series,
                           ret_from_bb_low_20d_2: pd.Series,
                           ret_from_bb_low_20d_3: pd.Series,
                           ret_from_bb_low_50d_2: pd.Series,
                           ret_from_bb_low_50d_3: pd.Series,
                           ret_from_bb_low_100d_2: pd.Series,
                           ret_from_bb_low_100d_3: pd.Series,
                           ret_from_bb_low_200d_2: pd.Series,
                           ret_from_bb_low_200d_3: pd.Series,
                           ret_to_next_new_york_close: pd.Series,
                           ret_to_next_weekly_close: pd.Series,
                           ret_to_next_monthly_close: pd.Series,
                           option_expiration_time: pd.Series,
                           last_option_expiration_close: pd.Series,
                           next_new_york_close: pd.Series,
                           next_weekly_close: pd.Series,
                           next_monthly_close: pd.Series,
                           next_london_close: pd.Series,
                           last_option_expiration_time: pd.Series) -> pd.Series:
    raw_data = clean_sorted_data.copy()
    add_daily_rolling_features=config.model.features.add_daily_rolling_features
    interval_mins = config.dataset.interval_mins
    new_york_cal = mcal.get_calendar("NYSE")
    lse_cal = mcal.get_calendar("LSE")
    #logging.error(f"raw_data:{raw_data.iloc[-10:]}")
    #logging.error(f"time_to_low_21d_ff_shift_21d:{time_to_low_21d_ff_shift_21d[-10:]}")
    #logging.error(f"time_to_low_21d_ff_shift_21d:{time_to_low_21d_ff_shift_21d.shape}")
    raw_data["ticker"] = ticker
    raw_data["timestamp"] = timestamp
    raw_data["week_of_year"] = week_of_year
    raw_data["month_of_year"] = month_of_year
    raw_data["weekly_close_time"] = weekly_close_time
    raw_data["last_weekly_close_time"] = last_weekly_close_time
    raw_data["monthly_close_time"] = monthly_close_time
    raw_data["last_monthly_close_time"] = last_monthly_close_time
    raw_data["option_expiration_time"] = option_expiration_time
    raw_data["last_option_expiration_time"] = last_option_expiration_time
    raw_data["time_to_low_1d_ff_shift_1d"] = time_to_low_1d_ff_shift_1d
    raw_data["time_to_low_5d_ff_shift_5d"] = time_to_low_5d_ff_shift_5d
    raw_data["time_to_low_11d_ff_shift_11d"] = time_to_low_11d_ff_shift_11d
    raw_data["time_to_low_21d_ff_shift_21d"] = time_to_low_21d_ff_shift_21d
    raw_data["time_to_high_1d_ff_shift_1d"] = time_to_high_1d_ff_shift_1d
    raw_data["time_to_high_5d_ff_shift_5d"] = time_to_high_5d_ff_shift_5d
    raw_data["time_to_high_11d_ff_shift_11d"] = time_to_high_11d_ff_shift_11d
    raw_data["time_to_high_21d_ff_shift_21d"] = time_to_high_21d_ff_shift_21d
    raw_data["time_to_last_macro_event_imp1"] = time_to_last_macro_event_imp1
    raw_data["time_to_last_macro_event_imp2"] = time_to_last_macro_event_imp2
    raw_data["time_to_last_macro_event_imp3"] = time_to_last_macro_event_imp3
    raw_data["time_to_next_macro_event_imp1"] = time_to_next_macro_event_imp1
    raw_data["time_to_next_macro_event_imp2"] = time_to_next_macro_event_imp2
    raw_data["time_to_next_macro_event_imp3"] = time_to_next_macro_event_imp3
    raw_data["time_to_low_1d_ff"] = time_to_low_1d_ff
    raw_data["time_to_low_5d_ff"] = time_to_low_5d_ff
    raw_data["time_to_low_11d_ff"] = time_to_low_11d_ff
    raw_data["time_to_low_21d_ff"] = time_to_low_21d_ff
    raw_data["time_to_low_51d_ff"] = time_to_low_51d_ff
    raw_data["time_to_low_101d_ff"] = time_to_low_101d_ff
    raw_data["time_to_low_201d_ff"] = time_to_low_201d_ff
    raw_data["time_to_high_1d_ff"] = time_to_high_1d_ff
    raw_data["time_to_high_5d_ff" ] = time_to_high_5d_ff
    raw_data["time_to_high_11d_ff"] = time_to_high_11d_ff
    raw_data["time_to_high_21d_ff"] = time_to_high_21d_ff
    raw_data["time_to_high_51d_ff"] = time_to_high_51d_ff
    raw_data["time_to_high_101d_ff"] = time_to_high_101d_ff
    raw_data["time_to_high_201d_ff"] = time_to_high_201d_ff
    raw_data["time_to_low_5_ff"] = time_to_low_5_ff
    raw_data["time_to_low_11_ff"] = time_to_low_11_ff
    raw_data["time_to_low_21_ff"] = time_to_low_21_ff
    raw_data["time_to_low_51_ff"] = time_to_low_51_ff
    raw_data["time_to_low_101_ff"] = time_to_low_101_ff
    raw_data["time_to_low_201_ff"] = time_to_low_201_ff
    raw_data["time_to_high_5_ff" ] = time_to_high_5_ff
    raw_data["time_to_high_11_ff"] = time_to_high_11_ff
    raw_data["time_to_high_21_ff"] = time_to_high_21_ff
    raw_data["time_to_high_51_ff"] = time_to_high_51_ff
    raw_data["time_to_high_101_ff"] = time_to_high_101_ff
    raw_data["time_to_high_201_ff"] = time_to_high_201_ff
    raw_data["time_to_weekly_close"] = time_to_weekly_close
    raw_data["time_to_monthly_close"] = time_to_monthly_close
    raw_data["time_to_option_expiration"] = time_to_option_expiration
    raw_data["time_to_new_york_open" ] = time_to_new_york_open
    raw_data["time_to_new_york_last_open"] = time_to_new_york_last_open
    raw_data["time_to_new_york_last_close"] = time_to_new_york_last_close
    raw_data["time_to_new_york_close"] = time_to_new_york_close
    raw_data["time_to_london_open"] = time_to_london_open
    raw_data["time_to_london_last_open"] = time_to_london_last_open
    raw_data["time_to_london_close"] = time_to_london_close
    raw_data["time_to_london_last_close"] = time_to_london_last_close

    
    if add_daily_rolling_features:
        raw_data["ret_from_close_cumsum_high_5d"] = ret_from_close_cumsum_high_5d
        raw_data["ret_from_close_cumsum_high_11d"] = ret_from_close_cumsum_high_11d
        raw_data["ret_from_close_cumsum_high_21d"] = ret_from_close_cumsum_high_21d
        raw_data["ret_from_close_cumsum_high_51d"] = ret_from_close_cumsum_high_51d
        raw_data["ret_from_close_cumsum_high_101d"] = ret_from_close_cumsum_high_101d
        raw_data["ret_from_close_cumsum_high_201d"] = ret_from_close_cumsum_high_201d
        raw_data["ret_from_close_cumsum_low_5d"] = ret_from_close_cumsum_low_5d
        raw_data["ret_from_close_cumsum_low_11d"] = ret_from_close_cumsum_low_11d
        raw_data["ret_from_close_cumsum_low_21d"] = ret_from_close_cumsum_low_21d
        raw_data["ret_from_close_cumsum_low_51d"] = ret_from_close_cumsum_low_51d
        raw_data["ret_from_close_cumsum_low_101d"] = ret_from_close_cumsum_low_101d
        raw_data["ret_from_close_cumsum_low_201d"] = ret_from_close_cumsum_low_201d

        raw_data["ret_from_high_1d"] = ret_from_high_1d
        raw_data["ret_from_high_1d_shift_1d"] = ret_from_high_1d_shift_1d
        raw_data["ret_from_high_5d"] = ret_from_high_5d
        raw_data["ret_from_high_5d_shift_5d"] = ret_from_high_5d_shift_5d
        raw_data["ret_from_high_11d"] = ret_from_high_11d
        raw_data["ret_from_high_11d_shift_11d"] = ret_from_high_11d_shift_11d
        raw_data["ret_from_high_21d"] = ret_from_high_21d
        raw_data["ret_from_high_21d_shift_21d"] = ret_from_high_21d_shift_21d
        raw_data["ret_from_high_51d"] = ret_from_high_51d
        raw_data["ret_from_high_101d"] = ret_from_high_101d
        raw_data["ret_from_high_201d"] = ret_from_high_201d
        raw_data["ret_from_low_1d"] = ret_from_low_1d
        raw_data["ret_from_low_1d_shift_1d"] = ret_from_low_1d_shift_1d
        raw_data["ret_from_low_5d"] = ret_from_low_5d
        raw_data["ret_from_low_5d_shift_5d"] = ret_from_low_5d_shift_5d
        raw_data["ret_from_low_11d"] = ret_from_low_11d
        raw_data["ret_from_low_11d_shift_11d"] = ret_from_low_11d_shift_11d
        raw_data["ret_from_low_21d"] = ret_from_low_21d
        raw_data["ret_from_low_21d_shift_21d"] = ret_from_low_21d_shift_21d
        raw_data["ret_from_low_51d"] = ret_from_low_51d
        raw_data["ret_from_low_101d"] = ret_from_low_101d
        raw_data["ret_from_low_201d"] = ret_from_low_201d

        raw_data["ret_from_high_5d_bf"] = ret_from_high_5d_bf
        raw_data["ret_from_high_11d_bf"] = ret_from_high_11d_bf
        raw_data["ret_from_high_21d_bf"] = ret_from_high_21d_bf
        raw_data["ret_from_high_51d_bf"] = ret_from_high_51d_bf
        raw_data["ret_from_low_5d_bf"] = ret_from_low_5d_bf
        raw_data["ret_from_low_11d_bf"] = ret_from_low_11d_bf
        raw_data["ret_from_low_21d_bf"] = ret_from_low_21d_bf
        raw_data["ret_from_low_51d_bf"] = ret_from_low_51d_bf

    raw_data["ret_from_close_cumsum_high_5"] = ret_from_close_cumsum_high_5
    raw_data["ret_from_close_cumsum_high_11"] = ret_from_close_cumsum_high_11
    raw_data["ret_from_close_cumsum_high_21"] = ret_from_close_cumsum_high_21
    raw_data["ret_from_close_cumsum_high_51"] = ret_from_close_cumsum_high_51
    raw_data["ret_from_close_cumsum_high_101"] = ret_from_close_cumsum_high_101
    raw_data["ret_from_close_cumsum_high_201"] = ret_from_close_cumsum_high_201
    raw_data["ret_from_close_cumsum_low_5"] = ret_from_close_cumsum_low_5
    raw_data["ret_from_close_cumsum_low_11"] = ret_from_close_cumsum_low_11
    raw_data["ret_from_close_cumsum_low_21"] = ret_from_close_cumsum_low_21
    raw_data["ret_from_close_cumsum_low_51"] = ret_from_close_cumsum_low_51
    raw_data["ret_from_close_cumsum_low_101"] = ret_from_close_cumsum_low_101
    raw_data["ret_from_close_cumsum_low_201"] = ret_from_close_cumsum_low_201

    raw_data["ret_from_high_5"] = ret_from_high_5
    raw_data["ret_velocity_from_high_5"] = ret_velocity_from_high_5
    raw_data["ret_from_high_11"] = ret_from_high_11
    raw_data["ret_velocity_from_high_11"] = ret_velocity_from_high_11
    raw_data["ret_from_high_21"] = ret_from_high_21
    raw_data["ret_velocity_from_high_21"] = ret_velocity_from_high_21
    raw_data["ret_from_high_51"] = ret_from_high_51
    raw_data["ret_velocity_from_high_51"] = ret_velocity_from_high_51
    raw_data["ret_from_high_101"] = ret_from_high_101
    raw_data["ret_velocity_from_high_101"] = ret_velocity_from_high_101
    raw_data["ret_from_high_201"] = ret_from_high_201
    raw_data["ret_velocity_from_high_201"] = ret_velocity_from_high_201
    raw_data["ret_from_low_5"] = ret_from_low_5
    raw_data["ret_velocity_from_low_5"] = ret_velocity_from_low_5
    raw_data["ret_from_low_11"] = ret_from_low_11
    raw_data["ret_velocity_from_low_11"] = ret_velocity_from_low_11
    raw_data["ret_from_low_21"] = ret_from_low_21
    raw_data["ret_velocity_from_low_21"] = ret_velocity_from_low_21
    raw_data["ret_from_low_51"] = ret_from_low_51
    raw_data["ret_velocity_from_low_51"] = ret_velocity_from_low_51
    raw_data["ret_from_low_101"] = ret_from_low_101
    raw_data["ret_velocity_from_low_101"] = ret_velocity_from_low_101
    raw_data["ret_from_low_201"] = ret_from_low_201
    raw_data["ret_velocity_from_low_201"] = ret_velocity_from_low_201

    if macro_data_builder.add_macro_event:
        raw_data["ret_from_vwap_since_last_macro_event_imp1"] = ret_from_vwap_since_last_macro_event_imp1
        raw_data["ret_from_vwap_pre_macro_event_imp1"] = ret_from_vwap_pre_macro_event_imp1
        raw_data["ret_from_vwap_post_macro_event_imp1"] = ret_from_vwap_post_macro_event_imp1
        raw_data["ret_from_vwap_around_macro_event_imp1"] = ret_from_vwap_around_macro_event_imp1

        raw_data["ret_from_vwap_since_last_macro_event_imp2"] = ret_from_vwap_since_last_macro_event_imp2
        raw_data["ret_from_vwap_pre_macro_event_imp2"] = ret_from_vwap_pre_macro_event_imp2
        raw_data["ret_from_vwap_post_macro_event_imp2"] = ret_from_vwap_post_macro_event_imp2
        raw_data["ret_from_vwap_around_macro_event_imp2"] = ret_from_vwap_around_macro_event_imp2

        raw_data["ret_from_vwap_since_last_macro_event_imp3"] = ret_from_vwap_since_last_macro_event_imp3
        raw_data["ret_from_vwap_pre_macro_event_imp3"] = ret_from_vwap_pre_macro_event_imp3
        raw_data["ret_from_vwap_post_macro_event_imp3"] = ret_from_vwap_post_macro_event_imp3
        raw_data["ret_from_vwap_around_macro_event_imp3"] = ret_from_vwap_around_macro_event_imp3
        
    raw_data["ret_from_vwap_since_new_york_open"] = ret_from_vwap_since_new_york_open
    raw_data["ret_from_vwap_since_london_open"] = ret_from_vwap_since_london_open
    raw_data["ret_from_vwap_pre_new_york_open"] = ret_from_vwap_pre_new_york_open
    raw_data["ret_from_vwap_post_new_york_open"] = ret_from_vwap_post_new_york_open

    raw_data["daily_returns"] = daily_returns
    raw_data["daily_returns_5"] = daily_returns_5
    raw_data["daily_returns_10"] = daily_returns_10
    raw_data["daily_returns_20"] = daily_returns_20
    raw_data["daily_vol"] = daily_vol
    raw_data["daily_vol_5"] = daily_vol_5
    raw_data["daily_vol_10"] = daily_vol_10
    raw_data["daily_vol_20"] = daily_vol_20
    raw_data["daily_skew"] = daily_skew
    raw_data["daily_skew_5"] = daily_skew_5
    raw_data["daily_skew_10"] = daily_skew_10
    raw_data["daily_skew_20"] = daily_skew_20
    raw_data["daily_kurt"] = daily_kurt
    raw_data["daily_kurt_5"] = daily_kurt_5
    raw_data["daily_kurt_10"] = daily_kurt_10
    raw_data["daily_kurt_20"] = daily_kurt_20

    trend_combinations = [(8, 24), (16, 48), (32, 96)]
    for short_window, long_window in trend_combinations:
        raw_data[f"macd_{short_window}_{long_window}"] = MACDStrategy.calc_signal(
            raw_data["close"], short_window, long_window, interval_per_day=1
        )
    if add_daily_rolling_features:
        interval_per_day = int(23 * 60 / interval_mins)        
        for short_window, long_window in trend_combinations:
            raw_data[f"macd_{short_window}_{long_window}_day"] = MACDStrategy.calc_signal(
                raw_data["close"], short_window*interval_per_day, long_window*interval_per_day, interval_per_day=46
            )
        raw_data["daily_returns_1d"] = daily_returns_1d
        raw_data["daily_returns_5d"] = daily_returns_5d
        raw_data["daily_returns_10d"] = daily_returns_10d
        raw_data["daily_returns_20d"] = daily_returns_20d
        raw_data["daily_vol_1d"] = daily_vol_1d
        raw_data["daily_vol_5d"] = daily_vol_5d
        raw_data["daily_vol_10d"] = daily_vol_10d
        raw_data["daily_vol_20d"] = daily_vol_20d
        raw_data["daily_vol_diff_1_20d"] = raw_data["daily_vol_1d"] - raw_data["daily_vol_20d"]
        raw_data["daily_skew_1d"] = daily_skew_1d
        raw_data["daily_skew_5d"] = daily_skew_5d
        raw_data["daily_skew_10d"] = daily_skew_10d
        raw_data["daily_skew_20d"] = daily_skew_20d
        raw_data["daily_kurt_1d"] = daily_kurt_1d
        raw_data["daily_kurt_5d"] = daily_kurt_5d
        raw_data["daily_kurt_10d"] = daily_kurt_10d
        raw_data["daily_kurt_20d"] = daily_kurt_20d

    raw_data[f"ret_from_last_daily_close_0"] = ret_from_last_daily_close_0
    raw_data[f"ret_from_last_daily_close_1"] = ret_from_last_daily_close_1
    raw_data[f"ret_from_last_daily_close_2"] = ret_from_last_daily_close_2
    raw_data[f"ret_from_last_daily_close_3"] = ret_from_last_daily_close_3
    raw_data[f"ret_from_last_daily_close_4"] = ret_from_last_daily_close_4
    raw_data[f"ret_from_last_daily_close_5"] = ret_from_last_daily_close_5
    raw_data[f"ret_from_last_daily_close_6"] = ret_from_last_daily_close_6
    raw_data[f"ret_from_last_daily_close_7"] = ret_from_last_daily_close_7
    raw_data[f"ret_from_last_daily_close_8"] = ret_from_last_daily_close_8
    raw_data[f"ret_from_last_daily_close_9"] = ret_from_last_daily_close_9
    raw_data[f"ret_from_last_daily_close_10"] = ret_from_last_daily_close_10
    raw_data[f"ret_from_last_daily_close_11"] = ret_from_last_daily_close_11
    raw_data[f"ret_from_last_daily_close_12"] = ret_from_last_daily_close_12
    raw_data[f"ret_from_last_daily_close_13"] = ret_from_last_daily_close_13
    raw_data[f"ret_from_last_daily_close_14"] = ret_from_last_daily_close_14
    raw_data[f"ret_from_last_daily_close_15"] = ret_from_last_daily_close_15
    raw_data[f"ret_from_last_daily_close_16"] = ret_from_last_daily_close_16
    raw_data[f"ret_from_last_daily_close_17"] = ret_from_last_daily_close_17
    raw_data[f"ret_from_last_daily_close_18"] = ret_from_last_daily_close_18
    raw_data[f"ret_from_last_daily_close_19"] = ret_from_last_daily_close_19

    raw_data[f"ret_from_last_weekly_close_0"] = ret_from_last_weekly_close_0
    raw_data[f"ret_from_last_weekly_close_1"] = ret_from_last_weekly_close_1
    raw_data[f"ret_from_last_weekly_close_2"] = ret_from_last_weekly_close_2
    raw_data[f"ret_from_last_weekly_close_3"] = ret_from_last_weekly_close_3
    raw_data[f"ret_from_last_weekly_close_4"] = ret_from_last_weekly_close_4
    raw_data[f"ret_from_last_weekly_close_5"] = ret_from_last_weekly_close_5
    raw_data[f"ret_from_last_weekly_close_6"] = ret_from_last_weekly_close_6
    raw_data[f"ret_from_last_weekly_close_7"] = ret_from_last_weekly_close_7
    raw_data[f"ret_from_last_weekly_close_8"] = ret_from_last_weekly_close_8
    raw_data[f"ret_from_last_weekly_close_9"] = ret_from_last_weekly_close_9
    raw_data[f"ret_from_last_weekly_close_10"] = ret_from_last_weekly_close_10
    raw_data[f"ret_from_last_weekly_close_11"] = ret_from_last_weekly_close_11
    raw_data[f"ret_from_last_weekly_close_12"] = ret_from_last_weekly_close_12
    raw_data[f"ret_from_last_weekly_close_13"] = ret_from_last_weekly_close_13
    raw_data[f"ret_from_last_weekly_close_14"] = ret_from_last_weekly_close_14
    raw_data[f"ret_from_last_weekly_close_15"] = ret_from_last_weekly_close_15
    raw_data[f"ret_from_last_weekly_close_16"] = ret_from_last_weekly_close_16
    raw_data[f"ret_from_last_weekly_close_17"] = ret_from_last_weekly_close_17
    raw_data[f"ret_from_last_weekly_close_18"] = ret_from_last_weekly_close_18
    raw_data[f"ret_from_last_weekly_close_19"] = ret_from_last_weekly_close_19

    raw_data[f"ret_from_last_monthly_close_0"] = ret_from_last_monthly_close_0
    raw_data[f"ret_from_last_monthly_close_1"] = ret_from_last_monthly_close_1
    raw_data[f"ret_from_last_monthly_close_2"] = ret_from_last_monthly_close_2
    raw_data[f"ret_from_last_monthly_close_3"] = ret_from_last_monthly_close_3
    raw_data[f"ret_from_last_monthly_close_4"] = ret_from_last_monthly_close_4
    raw_data[f"ret_from_last_monthly_close_5"] = ret_from_last_monthly_close_5
    raw_data[f"ret_from_last_monthly_close_6"] = ret_from_last_monthly_close_6
    raw_data[f"ret_from_last_monthly_close_7"] = ret_from_last_monthly_close_7
    raw_data[f"ret_from_last_monthly_close_8"] = ret_from_last_monthly_close_8
    raw_data[f"ret_from_last_monthly_close_9"] = ret_from_last_monthly_close_9
    raw_data[f"ret_from_last_monthly_close_10"] = ret_from_last_monthly_close_10
    raw_data[f"ret_from_last_monthly_close_11"] = ret_from_last_monthly_close_11
    raw_data[f"ret_from_last_monthly_close_12"] = ret_from_last_monthly_close_12
    raw_data[f"ret_from_last_monthly_close_13"] = ret_from_last_monthly_close_13
    raw_data[f"ret_from_last_monthly_close_14"] = ret_from_last_monthly_close_14
    raw_data[f"ret_from_last_monthly_close_15"] = ret_from_last_monthly_close_15
    raw_data[f"ret_from_last_monthly_close_16"] = ret_from_last_monthly_close_16
    raw_data[f"ret_from_last_monthly_close_17"] = ret_from_last_monthly_close_17
    raw_data[f"ret_from_last_monthly_close_18"] = ret_from_last_monthly_close_18
    raw_data[f"ret_from_last_monthly_close_19"] = ret_from_last_monthly_close_19

    raw_data["ret_to_next_new_york_close"] = ret_to_next_new_york_close
    raw_data["ret_to_next_weekly_close"] = ret_to_next_weekly_close
    raw_data["ret_to_next_monthly_close"] = ret_to_next_monthly_close
                                                        
    raw_data["ret_from_vwap_around_new_york_open"] = ret_from_vwap_around_new_york_open
    raw_data["ret_from_vwap_pre_new_york_close"] = ret_from_vwap_pre_new_york_close
    raw_data["ret_from_vwap_post_new_york_close"] = ret_from_vwap_post_new_york_close
    raw_data["ret_from_vwap_around_new_york_close"] = ret_from_vwap_around_new_york_close
    raw_data["ret_from_vwap_pre_london_open"] = ret_from_vwap_pre_london_open
    raw_data["ret_from_vwap_post_london_open"] = ret_from_vwap_post_london_open
    raw_data["ret_from_vwap_around_london_open"] = ret_from_vwap_around_london_open
    raw_data["ret_from_vwap_pre_london_close"] = ret_from_vwap_pre_london_close
    raw_data["ret_from_vwap_post_london_close"] = ret_from_vwap_post_london_close
    raw_data["ret_from_vwap_around_london_close"] = ret_from_vwap_around_london_close

    raw_data["ret_from_bb_high_5_2"] = ret_from_bb_high_5_2
    raw_data["ret_from_bb_high_5_3"] = ret_from_bb_high_5_3
    raw_data["ret_from_bb_high_10_2"] = ret_from_bb_high_10_2
    raw_data["ret_from_bb_high_10_3"] = ret_from_bb_high_10_3
    raw_data["ret_from_bb_high_20_2"] = ret_from_bb_high_20_2
    raw_data["ret_from_bb_high_20_3"] = ret_from_bb_high_20_3
    raw_data["ret_from_bb_high_50_2"] = ret_from_bb_high_50_2
    raw_data["ret_from_bb_high_50_3"] = ret_from_bb_high_50_3
    raw_data["ret_from_bb_high_100_2"] = ret_from_bb_high_100_2
    raw_data["ret_from_bb_high_100_3"] = ret_from_bb_high_100_3
    raw_data["ret_from_bb_high_200_2"] = ret_from_bb_high_200_2
    raw_data["ret_from_bb_high_200_3"] = ret_from_bb_high_200_3

    raw_data["ret_from_bb_low_5_2"] = ret_from_bb_low_5_2
    raw_data["ret_from_bb_low_5_3"] = ret_from_bb_low_5_3
    raw_data["ret_from_bb_low_10_2"] = ret_from_bb_low_10_2
    raw_data["ret_from_bb_low_10_3"] = ret_from_bb_low_10_3
    raw_data["ret_from_bb_low_20_2"] = ret_from_bb_low_20_2
    raw_data["ret_from_bb_low_20_3"] = ret_from_bb_low_20_3
    raw_data["ret_from_bb_low_50_2"] = ret_from_bb_low_50_2
    raw_data["ret_from_bb_low_50_3"] = ret_from_bb_low_50_3
    raw_data["ret_from_bb_low_100_2"] = ret_from_bb_low_100_2
    raw_data["ret_from_bb_low_100_3"] = ret_from_bb_low_100_3
    raw_data["ret_from_bb_low_200_2"] = ret_from_bb_low_200_2
    raw_data["ret_from_bb_low_200_3"] = ret_from_bb_low_200_3

    raw_data["ret_from_sma_5"] = ret_from_sma_5
    raw_data["ret_from_sma_10"] = ret_from_sma_10
    raw_data["ret_from_sma_20"] = ret_from_sma_20
    raw_data["ret_from_sma_50"] = ret_from_sma_50
    raw_data["ret_from_sma_100"] = ret_from_sma_100
    raw_data["ret_from_sma_200"] = ret_from_sma_200

    if add_daily_rolling_features:
        raw_data["ret_from_bb_high_5d_2"] = ret_from_bb_high_5d_2
        raw_data["ret_from_bb_high_5d_3"] = ret_from_bb_high_5d_3
        raw_data["ret_from_bb_high_10d_2"] = ret_from_bb_high_10d_2
        raw_data["ret_from_bb_high_10d_3"] = ret_from_bb_high_10d_3
        raw_data["ret_from_bb_high_20d_2"] = ret_from_bb_high_20d_2
        raw_data["ret_from_bb_high_20d_3"] = ret_from_bb_high_20d_3
        raw_data["ret_from_bb_high_50d_2"] = ret_from_bb_high_50d_2
        raw_data["ret_from_bb_high_50d_3"] = ret_from_bb_high_50d_3
        raw_data["ret_from_bb_high_100d_2"] = ret_from_bb_high_100d_2
        raw_data["ret_from_bb_high_100d_3"] = ret_from_bb_high_100d_3
        raw_data["ret_from_bb_high_200d_2"] = ret_from_bb_high_200d_2
        raw_data["ret_from_bb_high_200d_3"] = ret_from_bb_high_200d_3

        raw_data["ret_from_bb_low_5d_2"] = ret_from_bb_low_5d_2
        raw_data["ret_from_bb_low_5d_3"] = ret_from_bb_low_5d_3
        raw_data["ret_from_bb_low_10d_2"] = ret_from_bb_low_10d_2
        raw_data["ret_from_bb_low_10d_3"] = ret_from_bb_low_10d_3
        raw_data["ret_from_bb_low_20d_2"] = ret_from_bb_low_20d_2
        raw_data["ret_from_bb_low_20d_3"] = ret_from_bb_low_20d_3
        raw_data["ret_from_bb_low_50d_2"] = ret_from_bb_low_50d_2
        raw_data["ret_from_bb_low_50d_3"] = ret_from_bb_low_50d_3
        raw_data["ret_from_bb_low_100d_2"] = ret_from_bb_low_100d_2
        raw_data["ret_from_bb_low_100d_3"] = ret_from_bb_low_100d_3
        raw_data["ret_from_bb_low_200d_2"] = ret_from_bb_low_200d_2
        raw_data["ret_from_bb_low_200d_3"] = ret_from_bb_low_200d_3

        raw_data["ret_from_sma_5d"] = ret_from_sma_5d
        raw_data["ret_from_sma_10d"] = ret_from_sma_10d
        raw_data["ret_from_sma_20d"] = ret_from_sma_20d
        raw_data["ret_from_sma_50d"] = ret_from_sma_50d
        raw_data["ret_from_sma_100d"] = ret_from_sma_100d
        raw_data["ret_from_sma_200d"] = ret_from_sma_200d

    
    #logging.error(f"sampled_raw:{raw_data.iloc[-10:]}")
    raw_data["idx_ticker"] = raw_data["ticker"]
    raw_data["close_back"] = trimmed_close_back
    raw_data["close_velocity_back"] = close_velocity_back
    raw_data["open_back"] = trimmed_open_back
    raw_data["high_back"] = trimmed_high_back
    raw_data["low_back"] = trimmed_low_back
    raw_data["volume_back"] = volume_back
    raw_data["rsi_14"] = rsi_14
    raw_data["rsi_28"] = rsi_28
    raw_data["rsi_42"] = rsi_42
    raw_data["rsi_14d"] = rsi_14d
    raw_data["rsi_28d"] = rsi_28d
    raw_data["rsi_42d"] = rsi_42d
    raw_data["idx_timestamp"] = raw_data["timestamp"]
    raw_data["time"] = time
    raw_data = raw_data.set_index(["idx_ticker","idx_timestamp"])
    raw_data = raw_data.sort_values(["ticker", "timestamp"])

    return raw_data

@parameterize(
    trimmed_close_back={"price_col": source("close_back")},
    trimmed_open_back={"price_col": source("open_back")},
    trimmed_high_back={"price_col": source("high_back")},
    trimmed_low_back={"price_col": source("low_back")},
)
def trimmed_price_back_tmpl(price_col:pd.Series, ret_std:float, vol_threshold:float) -> pd.Series:
    trimmed = np.minimum(price_col, vol_threshold * ret_std)
    trimmed = np.minimum(price_col, vol_threshold * ret_std)
    return trimmed
