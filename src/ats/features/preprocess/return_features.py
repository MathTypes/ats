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
def group_features(sorted_data : pd.DataFrame, config: DictConfig) -> pd.DataFrame:
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
    new_features = raw_data.sort_values(['timestamp']).groupby(["ticker"], group_keys=False)[[
        "volume", "dv", "close", "high", "low", "open", "timestamp"]].apply(ticker_transform, config)
    new_features = new_features.drop(columns=["volume", "dv", "close", "high", "low", "open", "timestamp"])
    raw_data = raw_data.join(new_features)
    raw_data.reset_index(drop=True, inplace=True)
    #del new_features

    return raw_data

@extract_columns(
    "close",
    "cum_volume",
    "cum_dv",
    "close_back",
    "high_back",
    "open_back",
    "low_back",
    "volume_back",
    "dv_back",
    "close_back_cumsum",
    "volume_back_cumsum",
    "close_high_1d_ff",
    "time_high_1d_ff",
    "close_high_1d_ff_shift_1d",
    "time_high_1d_ff_shift_1d",
    "close_low_1d_ff",
    "time_low_1d_ff",
    "close_low_1d_ff_shift_1d",
    "time_low_1d_ff_shift_1d",
    "close_high_5d_ff",
    "time_high_5d_ff",
    "close_high_5d_ff_shift_5d",
    "time_high_5d_ff_shift_5d",
    "close_low_5d_ff",
    "time_low_5d_ff",
    "close_low_5d_ff_shift_5d",
    "time_low_5d_ff_shift_5d",
    "close_high_11d_ff",
    "time_high_11d_ff",
    "close_high_11d_ff_shift_11d",
    "time_high_11d_ff_shift_11d",
    "close_low_11d_ff",
    "time_low_11d_ff",
    "close_low_11d_ff_shift_11d",
    "time_low_11d_ff_shift_11d",
    "close_high_21d_ff",
    "time_high_21d_ff",
    "close_high_21d_ff_shift_21d",
    "time_high_21d_ff_shift_21d",
    "close_low_21d_ff",
    "time_low_21d_ff",
    "close_low_21d_ff_shift_21d",
    "time_low_21d_ff_shift_21d",    
    "close_high_51d_ff",
    "time_high_51d_ff",
    "close_low_51d_ff",
    "time_low_51d_ff",
    "close_high_101d_ff",
    "time_high_101d_ff",
    "close_low_101d_ff",
    "time_low_101d_ff",
    "close_high_201d_ff",
    "time_high_201d_ff",
    "close_low_201d_ff",
    "time_low_201d_ff",
    "close_high_5_ff",
    "time_high_5_ff",
    "close_high_5_ff_shift_5",
    "time_high_5_ff_shift_5",
    "close_low_5_ff",
    "time_low_5_ff",
    "close_low_5_ff_shift_5",
    "time_low_5_ff_shift_5",
    "close_high_11_ff",
    "time_high_11_ff",
    "close_high_11_ff_shift_11",
    "time_high_11_ff_shift_11",
    "close_low_11_ff",
    "time_low_11_ff",
    "close_low_11_ff_shift_11",
    "time_low_11_ff_shift_11",
    "close_high_21_ff",
    "time_high_21_ff",
    "close_high_21_ff_shift_21",
    "time_high_21_ff_shift_21",
    "close_low_21_ff",
    "time_low_21_ff",
    "close_low_21_ff_shift_21",
    "time_low_21_ff_shift_21",    
    "close_high_51_ff",
    "time_high_51_ff",
    "close_low_51_ff",
    "time_low_51_ff",
    "close_high_101_ff",
    "time_high_101_ff",
    "close_low_101_ff",
    "time_low_101_ff",
    "close_high_201_ff",
    "time_high_201_ff",
    "close_low_201_ff",
    "time_low_201_ff",
)
def joined_data(group_features: pd.DataFrame) -> pd.DataFrame:
    return group_features

@parameterize(
    ret_from_vwap_since_last_macro_event_imp1={"time_col": value("last_macro_event_time_imp1")},
    ret_from_vwap_since_last_macro_event_imp2={"time_col": value("last_macro_event_time_imp2")},
    ret_from_vwap_since_last_macro_event_imp3={"time_col": value("last_macro_event_time_imp3")},
    ret_from_vwap_since_new_york_open={"time_col": value("new_york_last_open_time")},
    ret_from_vwap_since_london_open={"time_col": value("london_last_open_time")},
)
def ret_from_vwap(time_features:pd.DataFrame, time_col:str, interval_mins: int) ->pd.Series:
    raw_data = time_features
    cum_dv_col = f"{time_col}_cum_dv"
    cum_volume_col = f"{time_col}_cum_volume"
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
def around_vwap(time_features:pd.DataFrame, time_col:str, interval_mins:int) ->pd.Series:
    raw_data = time_features
    cum_dv_col = f"around_{time_col}_cum_dv"
    cum_volume_col = f"around_{time_col}_cum_volume"
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
def post_vwap(time_features:pd.DataFrame, time_col:str, interval_mins:int) ->pd.Series:
    raw_data = time_features
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
    return rol.apply(vwap_around, args=(cum_dv_col,cum_volume_col), raw=False).ffill()


@parameterize(
    vwap_pre_new_york_open={"time_col": value("new_york_last_open_time")},
    vwap_pre_london_open={"time_col": value("london_last_open_time")},
    vwap_pre_new_york_close={"time_col": value("new_york_last_close_time")},
    vwap_pre_london_close={"time_col": value("london_last_close_time")},
    vwap_pre_macro_event_imp1={"time_col": value("last_macro_event_time_imp1")},
    vwap_pre_macro_event_imp2={"time_col": value("last_macro_event_time_imp2")},
    vwap_pre_macro_event_imp3={"time_col": value("last_macro_event_time_imp3")},
)
def pre_vwap(time_features:pd.DataFrame, time_col:str, interval_mins:int) ->pd.Series:
    raw_data = time_features
    cum_dv_col = f"pre_{time_col}_cum_dv"
    cum_volume_col = f"pre_{time_col}_cum_volume"
    raw_data[cum_dv_col] = raw_data.apply(fill_cum_dv, time_col=time_col,
                                          pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)
    raw_data[cum_volume_col] = raw_data.apply(fill_cum_volume, time_col=time_col,
                                              pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)
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
    ret_from_high_1d={"base_col": source("close_high_1d_ff")},
    ret_from_high_1d_shift_1d={"base_col": source("close_high_1d_ff_shift_1d")},
    ret_from_high_5d={"base_col": source("close_high_5d_ff")},
    ret_from_high_5d_shift_5d={"base_col": source("close_high_5d_ff_shift_5d")},
    ret_from_high_11d={"base_col": source("close_high_11d_ff")},
    ret_from_high_11d_shift_11d={"base_col": source("close_high_11d_ff_shift_11d")},
    ret_from_high_21d={"base_col": source("close_high_21d_ff")},
    ret_from_high_21d_shift_21d={"base_col": source("close_high_21d_ff_shift_21d")},
    ret_from_high_51d={"base_col": source("close_high_51d_ff")},
    ret_from_high_101d={"base_col": source("close_high_101d_ff")},
    ret_from_high_201d={"base_col": source("close_high_201d_ff")},
    ret_from_low_1d={"base_col": source("close_low_1d_ff")},
    ret_from_low_1d_shift_1d={"base_col": source("close_low_1d_ff_shift_1d")},
    ret_from_low_5d={"base_col": source("close_low_5d_ff")},
    ret_from_low_5d_shift_5d={"base_col": source("close_low_5d_ff_shift_5d")},
    ret_from_low_11d={"base_col": source("close_low_11d_ff")},
    ret_from_low_11d_shift_11d={"base_col": source("close_low_11d_ff_shift_11d")},
    ret_from_low_21d={"base_col": source("close_low_21d_ff")},
    ret_from_low_21d_shift_21d={"base_col": source("close_low_21d_ff_shift_21d")},
    ret_from_low_51d={"base_col": source("close_low_51d_ff")},
    ret_from_low_101d={"base_col": source("close_low_101d_ff")},
    ret_from_low_201d={"base_col": source("close_low_201d_ff")},
    ret_from_high_5d_bf={"base_col": source("close_high_5d_bf")},
    ret_from_high_11d_bf={"base_col": source("close_high_11d_bf")},
    ret_from_high_21d_bf={"base_col": source("close_high_21d_bf")},
    ret_from_high_51d_bf={"base_col": source("close_high_51d_bf")},
    ret_from_low_5d_bf={"base_col": source("close_low_5d_bf")},
    ret_from_low_11d_bf={"base_col": source("close_low_11d_bf")},
    ret_from_low_21d_bf={"base_col": source("close_low_21d_bf")},
    ret_from_low_51d_bf={"base_col": source("close_low_51d_bf")},
    ret_from_high_5={"base_col": source("close_high_5_ff")},
    ret_from_high_11={"base_col": source("close_high_11_ff")},
    ret_from_high_21={"base_col": source("close_high_21_ff")},
    ret_from_high_51={"base_col": source("close_high_51_ff")},
    ret_from_high_101={"base_col": source("close_high_101_ff")},
    ret_from_high_201={"base_col": source("close_high_201_ff")},
    ret_from_low_5={"base_col": source("close_low_5_ff")},
    ret_from_low_11={"base_col": source("close_low_11_ff")},
    ret_from_low_21={"base_col": source("close_low_21_ff")},
    ret_from_low_51={"base_col": source("close_low_51_ff")},
    ret_from_low_101={"base_col": source("close_low_101_ff")},
    ret_from_low_201={"base_col": source("close_low_201_ff")},
    ret_from_vwap_pre_macro_event_imp1={"base_col": source("vwap_pre_macro_event_imp1")},
    ret_from_vwap_post_macro_event_imp1={"base_col": source("vwap_post_macro_event_imp1")},
    ret_from_vwap_around_macro_event_imp1={"base_col": source("vwap_around_macro_event_imp1")},
    ret_from_vwap_pre_macro_event_imp2={"base_col": source("vwap_pre_macro_event_imp2")},
    ret_from_vwap_post_macro_event_imp2={"base_col": source("vwap_post_macro_event_imp2")},
    ret_from_vwap_around_macro_event_imp2={"base_col": source("vwap_around_macro_event_imp2")},
    ret_from_vwap_pre_macro_event_imp3={"base_col": source("vwap_pre_macro_event_imp3")},
    ret_from_vwap_post_macro_event_imp3={"base_col": source("vwap_post_macro_event_imp3")},
    ret_from_vwap_around_macro_event_imp3={"base_col": source("vwap_around_macro_event_imp3")},
    ret_from_vwap_pre_new_york_open={"base_col": source("vwap_pre_new_york_open")},
    ret_from_vwap_post_new_york_open={"base_col": source("vwap_post_new_york_open")},
    ret_from_vwap_pre_new_york_close={"base_col":source("vwap_pre_new_york_close")},
    ret_from_vwap_post_new_york_close={"base_col":source("vwap_post_new_york_close")},
    ret_from_vwap_around_new_york_open={"base_col": source("vwap_around_new_york_open")},
    ret_from_vwap_around_new_york_close={"base_col": source("vwap_around_new_york_close")},

    ret_from_vwap_pre_london_open={"base_col": source("vwap_pre_london_open")},
    ret_from_vwap_around_london_open={"base_col": source("vwap_around_london_open")},
    ret_from_vwap_post_london_open={"base_col": source("vwap_post_london_open")},
    ret_from_vwap_pre_london_close={"base_col": source("vwap_pre_london_close")},
    ret_from_vwap_around_london_close={"base_col": source("vwap_around_london_close")},
    ret_from_vwap_post_london_close={"base_col": source("vwap_post_london_close")},
    ret_from_new_york_last_daily_open_0={"base_col": source("new_york_last_daily_open_0")},
    ret_from_new_york_last_daily_open_1={"base_col": source("new_york_last_daily_open_1")},
    ret_from_new_york_last_daily_open_2={"base_col": source("new_york_last_daily_open_2")},
    ret_from_new_york_last_daily_open_3={"base_col": source("new_york_last_daily_open_3")},
    ret_from_new_york_last_daily_open_4={"base_col": source("new_york_last_daily_open_4")},
    ret_from_new_york_last_daily_open_5={"base_col": source("new_york_last_daily_open_5")},
    ret_from_new_york_last_daily_open_6={"base_col": source("new_york_last_daily_open_6")},
    ret_from_new_york_last_daily_open_7={"base_col": source("new_york_last_daily_open_7")},
    ret_from_new_york_last_daily_open_8={"base_col": source("new_york_last_daily_open_8")},
    ret_from_new_york_last_daily_open_9={"base_col": source("new_york_last_daily_open_9")},
    ret_from_new_york_last_daily_open_10={"base_col": source("new_york_last_daily_open_10")},
    ret_from_new_york_last_daily_open_11={"base_col": source("new_york_last_daily_open_11")},
    ret_from_new_york_last_daily_open_12={"base_col": source("new_york_last_daily_open_12")},
    ret_from_new_york_last_daily_open_13={"base_col": source("new_york_last_daily_open_13")},
    ret_from_new_york_last_daily_open_14={"base_col": source("new_york_last_daily_open_14")},
    ret_from_new_york_last_daily_open_15={"base_col": source("new_york_last_daily_open_15")},
    ret_from_new_york_last_daily_open_16={"base_col": source("new_york_last_daily_open_16")},
    ret_from_new_york_last_daily_open_17={"base_col": source("new_york_last_daily_open_17")},
    ret_from_new_york_last_daily_open_18={"base_col": source("new_york_last_daily_open_18")},
    ret_from_new_york_last_daily_open_19={"base_col": source("new_york_last_daily_open_19")},
    ret_from_new_york_last_daily_close_0={"base_col": source("new_york_last_daily_close_0")},
    ret_from_new_york_last_daily_close_1={"base_col": source("new_york_last_daily_close_1")},
    ret_from_new_york_last_daily_close_2={"base_col": source("new_york_last_daily_close_2")},
    ret_from_new_york_last_daily_close_3={"base_col": source("new_york_last_daily_close_3")},
    ret_from_new_york_last_daily_close_4={"base_col": source("new_york_last_daily_close_4")},
    ret_from_new_york_last_daily_close_5={"base_col": source("new_york_last_daily_close_5")},
    ret_from_new_york_last_daily_close_6={"base_col": source("new_york_last_daily_close_6")},
    ret_from_new_york_last_daily_close_7={"base_col": source("new_york_last_daily_close_7")},
    ret_from_new_york_last_daily_close_8={"base_col": source("new_york_last_daily_close_8")},
    ret_from_new_york_last_daily_close_9={"base_col": source("new_york_last_daily_close_9")},
    ret_from_new_york_last_daily_close_10={"base_col": source("new_york_last_daily_close_10")},
    ret_from_new_york_last_daily_close_11={"base_col": source("new_york_last_daily_close_11")},
    ret_from_new_york_last_daily_close_12={"base_col": source("new_york_last_daily_close_12")},
    ret_from_new_york_last_daily_close_13={"base_col": source("new_york_last_daily_close_13")},
    ret_from_new_york_last_daily_close_14={"base_col": source("new_york_last_daily_close_14")},
    ret_from_new_york_last_daily_close_15={"base_col": source("new_york_last_daily_close_15")},
    ret_from_new_york_last_daily_close_16={"base_col": source("new_york_last_daily_close_16")},
    ret_from_new_york_last_daily_close_17={"base_col": source("new_york_last_daily_close_17")},
    ret_from_new_york_last_daily_close_18={"base_col": source("new_york_last_daily_close_18")},
    ret_from_new_york_last_daily_close_19={"base_col": source("new_york_last_daily_close_19")},
    ret_from_london_last_daily_open_0={"base_col": source("london_last_daily_open_0")},
    ret_from_london_last_daily_open_1={"base_col": source("london_last_daily_open_1")},
    ret_from_london_last_daily_open_2={"base_col": source("london_last_daily_open_2")},
    ret_from_london_last_daily_open_3={"base_col": source("london_last_daily_open_3")},
    ret_from_london_last_daily_open_4={"base_col": source("london_last_daily_open_4")},
    ret_from_london_last_daily_open_5={"base_col": source("london_last_daily_open_5")},
    ret_from_london_last_daily_open_6={"base_col": source("london_last_daily_open_6")},
    ret_from_london_last_daily_open_7={"base_col": source("london_last_daily_open_7")},
    ret_from_london_last_daily_open_8={"base_col": source("london_last_daily_open_8")},
    ret_from_london_last_daily_open_9={"base_col": source("london_last_daily_open_9")},
    ret_from_london_last_daily_open_10={"base_col": source("london_last_daily_open_10")},
    ret_from_london_last_daily_open_11={"base_col": source("london_last_daily_open_11")},
    ret_from_london_last_daily_open_12={"base_col": source("london_last_daily_open_12")},
    ret_from_london_last_daily_open_13={"base_col": source("london_last_daily_open_13")},
    ret_from_london_last_daily_open_14={"base_col": source("london_last_daily_open_14")},
    ret_from_london_last_daily_open_15={"base_col": source("london_last_daily_open_15")},
    ret_from_london_last_daily_open_16={"base_col": source("london_last_daily_open_16")},
    ret_from_london_last_daily_open_17={"base_col": source("london_last_daily_open_17")},
    ret_from_london_last_daily_open_18={"base_col": source("london_last_daily_open_18")},
    ret_from_london_last_daily_open_19={"base_col": source("london_last_daily_open_19")},
    ret_from_london_last_daily_close_0={"base_col": source("london_last_daily_close_0")},
    ret_from_london_last_daily_close_1={"base_col": source("london_last_daily_close_1")},
    ret_from_london_last_daily_close_2={"base_col": source("london_last_daily_close_2")},
    ret_from_london_last_daily_close_3={"base_col": source("london_last_daily_close_3")},
    ret_from_london_last_daily_close_4={"base_col": source("london_last_daily_close_4")},
    ret_from_london_last_daily_close_5={"base_col": source("london_last_daily_close_5")},
    ret_from_london_last_daily_close_6={"base_col": source("london_last_daily_close_6")},
    ret_from_london_last_daily_close_7={"base_col": source("london_last_daily_close_7")},
    ret_from_london_last_daily_close_8={"base_col": source("london_last_daily_close_8")},
    ret_from_london_last_daily_close_9={"base_col": source("london_last_daily_close_9")},
    ret_from_london_last_daily_close_10={"base_col": source("london_last_daily_close_10")},
    ret_from_london_last_daily_close_11={"base_col": source("london_last_daily_close_11")},
    ret_from_london_last_daily_close_12={"base_col": source("london_last_daily_close_12")},
    ret_from_london_last_daily_close_13={"base_col": source("london_last_daily_close_13")},
    ret_from_london_last_daily_close_14={"base_col": source("london_last_daily_close_14")},
    ret_from_london_last_daily_close_15={"base_col": source("london_last_daily_close_15")},
    ret_from_london_last_daily_close_16={"base_col": source("london_last_daily_close_16")},
    ret_from_london_last_daily_close_17={"base_col": source("london_last_daily_close_17")},
    ret_from_london_last_daily_close_18={"base_col": source("london_last_daily_close_18")},
    ret_from_london_last_daily_close_19={"base_col": source("london_last_daily_close_19")},
    ret_from_last_weekly_close_0={"base_col": source("last_weekly_close_0")},
    ret_from_last_weekly_close_1={"base_col": source("last_weekly_close_1")},
    ret_from_last_weekly_close_2={"base_col": source("last_weekly_close_2")},
    ret_from_last_weekly_close_3={"base_col": source("last_weekly_close_3")},
    ret_from_last_weekly_close_4={"base_col": source("last_weekly_close_4")},
    ret_from_last_weekly_close_5={"base_col": source("last_weekly_close_5")},
    ret_from_last_weekly_close_6={"base_col": source("last_weekly_close_6")},
    ret_from_last_weekly_close_7={"base_col": source("last_weekly_close_7")},
    ret_from_last_weekly_close_8={"base_col": source("last_weekly_close_8")},
    ret_from_last_weekly_close_9={"base_col": source("last_weekly_close_9")},
    ret_from_last_weekly_close_10={"base_col": source("last_weekly_close_10")},
    ret_from_last_weekly_close_11={"base_col": source("last_weekly_close_11")},
    ret_from_last_weekly_close_12={"base_col": source("last_weekly_close_12")},
    ret_from_last_weekly_close_13={"base_col": source("last_weekly_close_13")},
    ret_from_last_weekly_close_14={"base_col": source("last_weekly_close_14")},
    ret_from_last_weekly_close_15={"base_col": source("last_weekly_close_15")},
    ret_from_last_weekly_close_16={"base_col": source("last_weekly_close_16")},
    ret_from_last_weekly_close_17={"base_col": source("last_weekly_close_17")},
    ret_from_last_weekly_close_18={"base_col": source("last_weekly_close_18")},
    ret_from_last_weekly_close_19={"base_col": source("last_weekly_close_19")},
    ret_from_last_monthly_close_0={"base_col": source("last_monthly_close_0")},
    ret_from_last_monthly_close_1={"base_col": source("last_monthly_close_1")},
    ret_from_last_monthly_close_2={"base_col": source("last_monthly_close_2")},
    ret_from_last_monthly_close_3={"base_col": source("last_monthly_close_3")},
    ret_from_last_monthly_close_4={"base_col": source("last_monthly_close_4")},
    ret_from_last_monthly_close_5={"base_col": source("last_monthly_close_5")},
    ret_from_last_monthly_close_6={"base_col": source("last_monthly_close_6")},
    ret_from_last_monthly_close_7={"base_col": source("last_monthly_close_7")},
    ret_from_last_monthly_close_8={"base_col": source("last_monthly_close_8")},
    ret_from_last_monthly_close_9={"base_col": source("last_monthly_close_9")},
    ret_from_last_monthly_close_10={"base_col": source("last_monthly_close_10")},
    ret_from_last_monthly_close_11={"base_col": source("last_monthly_close_11")},
    ret_from_last_monthly_close_12={"base_col": source("last_monthly_close_12")},
    ret_from_last_monthly_close_13={"base_col": source("last_monthly_close_13")},
    ret_from_last_monthly_close_14={"base_col": source("last_monthly_close_14")},
    ret_from_last_monthly_close_15={"base_col": source("last_monthly_close_15")},
    ret_from_last_monthly_close_16={"base_col": source("last_monthly_close_16")},
    ret_from_last_monthly_close_17={"base_col": source("last_monthly_close_17")},
    ret_from_last_monthly_close_18={"base_col": source("last_monthly_close_18")},
    ret_from_last_monthly_close_19={"base_col": source("last_monthly_close_19")},
    ret_to_next_new_york_close={"base_col": source("next_new_york_close")},
    ret_to_next_weekly_close={"base_col": source("next_weekly_close")},
    ret_to_next_monthly_close={"base_col": source("next_monthly_close")},
    ret_from_bb_high_5_2={"base_col": source("bb_high_5_2")},
    ret_from_bb_high_5_3={"base_col": source("bb_high_5_3")},
    ret_from_bb_high_10_2={"base_col": source("bb_high_10_2")},
    ret_from_bb_high_10_3={"base_col": source("bb_high_10_3")},
    ret_from_bb_high_20_2={"base_col": source("bb_high_20_2")},
    ret_from_bb_high_20_3={"base_col": source("bb_high_20_3")},
    ret_from_bb_high_50_2={"base_col": source("bb_high_50_2")},
    ret_from_bb_high_50_3={"base_col": source("bb_high_50_3")},
    ret_from_bb_high_100_2={"base_col": source("bb_high_100_2")},
    ret_from_bb_high_100_3={"base_col": source("bb_high_100_3")},
    ret_from_bb_high_200_2={"base_col": source("bb_high_200_2")},
    ret_from_bb_high_200_3={"base_col": source("bb_high_200_3")},

    ret_from_bb_low_5_2={"base_col": source("bb_low_5_2")},
    ret_from_bb_low_5_3={"base_col": source("bb_low_5_3")},
    ret_from_bb_low_10_2={"base_col": source("bb_low_10_2")},
    ret_from_bb_low_10_3={"base_col": source("bb_low_10_3")},
    ret_from_bb_low_20_2={"base_col": source("bb_low_20_2")},
    ret_from_bb_low_20_3={"base_col": source("bb_low_20_3")},
    ret_from_bb_low_50_2={"base_col": source("bb_low_50_2")},
    ret_from_bb_low_50_3={"base_col": source("bb_low_50_3")},
    ret_from_bb_low_100_2={"base_col": source("bb_low_100_2")},
    ret_from_bb_low_100_3={"base_col": source("bb_low_100_3")},
    ret_from_bb_low_200_2={"base_col": source("bb_low_200_2")},
    ret_from_bb_low_200_3={"base_col": source("bb_low_200_3")},

    ret_from_bb_high_5d_2={"base_col": source("bb_high_5d_2")},
    ret_from_bb_high_5d_3={"base_col": source("bb_high_5d_3")},
    ret_from_bb_high_10d_2={"base_col": source("bb_high_10d_2")},
    ret_from_bb_high_10d_3={"base_col": source("bb_high_10d_3")},
    ret_from_bb_high_20d_2={"base_col": source("bb_high_20d_2")},
    ret_from_bb_high_20d_3={"base_col": source("bb_high_20d_3")},
    ret_from_bb_high_50d_2={"base_col": source("bb_high_50d_2")},
    ret_from_bb_high_50d_3={"base_col": source("bb_high_50d_3")},
    ret_from_bb_high_100d_2={"base_col": source("bb_high_100d_2")},
    ret_from_bb_high_100d_3={"base_col": source("bb_high_100d_3")},
    ret_from_bb_high_200d_2={"base_col": source("bb_high_200d_2")},
    ret_from_bb_high_200d_3={"base_col": source("bb_high_200d_3")},

    ret_from_bb_low_5d_2={"base_col": source("bb_low_5d_2")},
    ret_from_bb_low_5d_3={"base_col": source("bb_low_5d_3")},
    ret_from_bb_low_10d_2={"base_col": source("bb_low_10d_2")},
    ret_from_bb_low_10d_3={"base_col": source("bb_low_10d_3")},
    ret_from_bb_low_20d_2={"base_col": source("bb_low_20d_2")},
    ret_from_bb_low_20d_3={"base_col": source("bb_low_20d_3")},
    ret_from_bb_low_50d_2={"base_col": source("bb_low_50d_2")},
    ret_from_bb_low_50d_3={"base_col": source("bb_low_50d_3")},
    ret_from_bb_low_100d_2={"base_col": source("bb_low_100d_2")},
    ret_from_bb_low_100d_3={"base_col": source("bb_low_100d_3")},
    ret_from_bb_low_200d_2={"base_col": source("bb_low_200d_2")},
    ret_from_bb_low_200d_3={"base_col": source("bb_low_200d_3")},
    ret_from_sma_5={"base_col": source("sma_5")},
    ret_from_sma_10={"base_col": source("sma_10")},
    ret_from_sma_20={"base_col": source("sma_20")},
    ret_from_sma_50={"base_col": source("sma_50")},
    ret_from_sma_100={"base_col": source("sma_100")},
    ret_from_sma_200={"base_col": source("sma_200")},
    ret_from_sma_5d={"base_col": source("sma_5d")},
    ret_from_sma_10d={"base_col": source("sma_10d")},
    ret_from_sma_20d={"base_col": source("sma_20d")},
    ret_from_sma_50d={"base_col": source("sma_50d")},
    ret_from_sma_100d={"base_col": source("sma_100d")},
    ret_from_sma_200d={"base_col": source("sma_200d")},
)
def ret_from_price(close: pd.Series, base_col: pd.Series, base_price:float) -> pd.Series:
    return np.log(close+base_price) - np.log(base_col+base_price)

@parameterize(
    ret_velocity_from_high_5={"ret_col": source("ret_from_high_5"), "time_col": source("time_to_high_5_ff")},
    ret_velocity_from_high_11={"ret_col": source("ret_from_high_11"), "time_col": source("time_to_high_11_ff")},
    ret_velocity_from_high_21={"ret_col": source("ret_from_high_21"), "time_col": source("time_to_high_21_ff")},
    ret_velocity_from_high_51={"ret_col": source("ret_from_high_51"), "time_col": source("time_to_high_51_ff")},
    ret_velocity_from_high_101={"ret_col": source("ret_from_high_101"), "time_col": source("time_to_high_101_ff")},
    ret_velocity_from_high_201={"ret_col": source("ret_from_high_201"), "time_col": source("time_to_high_201_ff")},
    ret_velocity_from_low_5={"ret_col": source("ret_from_low_5"), "time_col": source("time_to_low_5_ff")},
    ret_velocity_from_low_11={"ret_col": source("ret_from_low_11"), "time_col": source("time_to_low_11_ff")},
    ret_velocity_from_low_21={"ret_col": source("ret_from_low_21"), "time_col": source("time_to_low_21_ff")},
    ret_velocity_from_low_51={"ret_col": source("ret_from_low_51"), "time_col": source("time_to_low_51_ff")},
    ret_velocity_from_low_101={"ret_col": source("ret_from_low_101"), "time_col": source("time_to_low_101_ff")},
    ret_velocity_from_low_201={"ret_col": source("ret_from_low_201"), "time_col": source("time_to_low_201_ff")},
)
def ret_velocity_tmpl(ret_col: pd.Series, time_col: pd.Series) -> pd.Series:
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
                           group_features:pd.DataFrame, config:DictConfig,
                           time_features: pd.DataFrame,
                           daily_returns:pd.Series, daily_returns_5:pd.Series,daily_returns_10:pd.Series,daily_returns_20:pd.Series,
                           daily_vol:pd.Series,daily_vol_5:pd.Series,daily_vol_10:pd.Series,daily_vol_20:pd.Series,
                           daily_skew:pd.Series,daily_skew_5:pd.Series,daily_skew_10:pd.Series,daily_skew_20:pd.Series,
                           daily_kurt:pd.Series,daily_kurt_5:pd.Series,daily_kurt_10:pd.Series,daily_kurt_20:pd.Series,
                           daily_returns_1d:pd.Series, daily_returns_5d:pd.Series,daily_returns_10d:pd.Series,daily_returns_20d:pd.Series,
                           daily_vol_1d:pd.Series,daily_vol_5d:pd.Series,daily_vol_10d:pd.Series,daily_vol_20d:pd.Series,
                           daily_skew_1d:pd.Series,daily_skew_5d:pd.Series,daily_skew_10d:pd.Series,daily_skew_20d:pd.Series,
                           daily_kurt_1d:pd.Series,daily_kurt_5d:pd.Series,daily_kurt_10d:pd.Series,daily_kurt_20d:pd.Series,
                           close: pd.Series,
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
                           new_york_last_daily_open_0: pd.Series,
                           new_york_last_daily_open_1: pd.Series,
                           new_york_last_daily_open_2: pd.Series,
                           new_york_last_daily_open_3: pd.Series,
                           new_york_last_daily_open_4: pd.Series,
                           new_york_last_daily_open_5: pd.Series,
                           new_york_last_daily_open_6: pd.Series,
                           new_york_last_daily_open_7: pd.Series,
                           new_york_last_daily_open_8: pd.Series,
                           new_york_last_daily_open_9: pd.Series,
                           new_york_last_daily_open_10: pd.Series,
                           new_york_last_daily_open_11: pd.Series,
                           new_york_last_daily_open_12: pd.Series,
                           new_york_last_daily_open_13: pd.Series,
                           new_york_last_daily_open_14: pd.Series,
                           new_york_last_daily_open_15: pd.Series,
                           new_york_last_daily_open_16: pd.Series,
                           new_york_last_daily_open_17: pd.Series,
                           new_york_last_daily_open_18: pd.Series,
                           new_york_last_daily_open_19: pd.Series,
                           new_york_last_daily_close_0: pd.Series,
                           new_york_last_daily_close_1: pd.Series,
                           new_york_last_daily_close_2: pd.Series,
                           new_york_last_daily_close_3: pd.Series,
                           new_york_last_daily_close_4: pd.Series,
                           new_york_last_daily_close_5: pd.Series,
                           new_york_last_daily_close_6: pd.Series,
                           new_york_last_daily_close_7: pd.Series,
                           new_york_last_daily_close_8: pd.Series,
                           new_york_last_daily_close_9: pd.Series,
                           new_york_last_daily_close_10: pd.Series,
                           new_york_last_daily_close_11: pd.Series,
                           new_york_last_daily_close_12: pd.Series,
                           new_york_last_daily_close_13: pd.Series,
                           new_york_last_daily_close_14: pd.Series,
                           new_york_last_daily_close_15: pd.Series,
                           new_york_last_daily_close_16: pd.Series,
                           new_york_last_daily_close_17: pd.Series,
                           new_york_last_daily_close_18: pd.Series,
                           new_york_last_daily_close_19: pd.Series,
                           london_last_daily_open_0: pd.Series,
                           london_last_daily_open_1: pd.Series,
                           london_last_daily_open_2: pd.Series,
                           london_last_daily_open_3: pd.Series,
                           london_last_daily_open_4: pd.Series,
                           london_last_daily_open_5: pd.Series,
                           london_last_daily_open_6: pd.Series,
                           london_last_daily_open_7: pd.Series,
                           london_last_daily_open_8: pd.Series,
                           london_last_daily_open_9: pd.Series,
                           london_last_daily_open_10: pd.Series,
                           london_last_daily_open_11: pd.Series,
                           london_last_daily_open_12: pd.Series,
                           london_last_daily_open_13: pd.Series,
                           london_last_daily_open_14: pd.Series,
                           london_last_daily_open_15: pd.Series,
                           london_last_daily_open_16: pd.Series,
                           london_last_daily_open_17: pd.Series,
                           london_last_daily_open_18: pd.Series,
                           london_last_daily_open_19: pd.Series,
                           london_last_daily_close_0: pd.Series,
                           london_last_daily_close_1: pd.Series,
                           london_last_daily_close_2: pd.Series,
                           london_last_daily_close_3: pd.Series,
                           london_last_daily_close_4: pd.Series,
                           london_last_daily_close_5: pd.Series,
                           london_last_daily_close_6: pd.Series,
                           london_last_daily_close_7: pd.Series,
                           london_last_daily_close_8: pd.Series,
                           london_last_daily_close_9: pd.Series,
                           london_last_daily_close_10: pd.Series,
                           london_last_daily_close_11: pd.Series,
                           london_last_daily_close_12: pd.Series,
                           london_last_daily_close_13: pd.Series,
                           london_last_daily_close_14: pd.Series,
                           london_last_daily_close_15: pd.Series,
                           london_last_daily_close_16: pd.Series,
                           london_last_daily_close_17: pd.Series,
                           london_last_daily_close_18: pd.Series,
                           london_last_daily_close_19: pd.Series,
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
                           ret_from_new_york_last_daily_open_0: pd.Series,
                           ret_from_new_york_last_daily_open_1: pd.Series,
                           ret_from_new_york_last_daily_open_2: pd.Series,
                           ret_from_new_york_last_daily_open_3: pd.Series,
                           ret_from_new_york_last_daily_open_4: pd.Series,
                           ret_from_new_york_last_daily_open_5: pd.Series,
                           ret_from_new_york_last_daily_open_6: pd.Series,
                           ret_from_new_york_last_daily_open_7: pd.Series,
                           ret_from_new_york_last_daily_open_8: pd.Series,
                           ret_from_new_york_last_daily_open_9: pd.Series,
                           ret_from_new_york_last_daily_open_10: pd.Series,
                           ret_from_new_york_last_daily_open_11: pd.Series,
                           ret_from_new_york_last_daily_open_12: pd.Series,
                           ret_from_new_york_last_daily_open_13: pd.Series,
                           ret_from_new_york_last_daily_open_14: pd.Series,
                           ret_from_new_york_last_daily_open_15: pd.Series,
                           ret_from_new_york_last_daily_open_16: pd.Series,
                           ret_from_new_york_last_daily_open_17: pd.Series,
                           ret_from_new_york_last_daily_open_18: pd.Series,
                           ret_from_new_york_last_daily_open_19: pd.Series,
                           ret_from_new_york_last_daily_close_0: pd.Series,
                           ret_from_new_york_last_daily_close_1: pd.Series,
                           ret_from_new_york_last_daily_close_2: pd.Series,
                           ret_from_new_york_last_daily_close_3: pd.Series,
                           ret_from_new_york_last_daily_close_4: pd.Series,
                           ret_from_new_york_last_daily_close_5: pd.Series,
                           ret_from_new_york_last_daily_close_6: pd.Series,
                           ret_from_new_york_last_daily_close_7: pd.Series,
                           ret_from_new_york_last_daily_close_8: pd.Series,
                           ret_from_new_york_last_daily_close_9: pd.Series,
                           ret_from_new_york_last_daily_close_10: pd.Series,
                           ret_from_new_york_last_daily_close_11: pd.Series,
                           ret_from_new_york_last_daily_close_12: pd.Series,
                           ret_from_new_york_last_daily_close_13: pd.Series,
                           ret_from_new_york_last_daily_close_14: pd.Series,
                           ret_from_new_york_last_daily_close_15: pd.Series,
                           ret_from_new_york_last_daily_close_16: pd.Series,
                           ret_from_new_york_last_daily_close_17: pd.Series,
                           ret_from_new_york_last_daily_close_18: pd.Series,
                           ret_from_new_york_last_daily_close_19: pd.Series,
                           ret_from_london_last_daily_open_0: pd.Series,
                           ret_from_london_last_daily_open_1: pd.Series,
                           ret_from_london_last_daily_open_2: pd.Series,
                           ret_from_london_last_daily_open_3: pd.Series,
                           ret_from_london_last_daily_open_4: pd.Series,
                           ret_from_london_last_daily_open_5: pd.Series,
                           ret_from_london_last_daily_open_6: pd.Series,
                           ret_from_london_last_daily_open_7: pd.Series,
                           ret_from_london_last_daily_open_8: pd.Series,
                           ret_from_london_last_daily_open_9: pd.Series,
                           ret_from_london_last_daily_open_10: pd.Series,
                           ret_from_london_last_daily_open_11: pd.Series,
                           ret_from_london_last_daily_open_12: pd.Series,
                           ret_from_london_last_daily_open_13: pd.Series,
                           ret_from_london_last_daily_open_14: pd.Series,
                           ret_from_london_last_daily_open_15: pd.Series,
                           ret_from_london_last_daily_open_16: pd.Series,
                           ret_from_london_last_daily_open_17: pd.Series,
                           ret_from_london_last_daily_open_18: pd.Series,
                           ret_from_london_last_daily_open_19: pd.Series,
                           ret_from_london_last_daily_close_0: pd.Series,
                           ret_from_london_last_daily_close_1: pd.Series,
                           ret_from_london_last_daily_close_2: pd.Series,
                           ret_from_london_last_daily_close_3: pd.Series,
                           ret_from_london_last_daily_close_4: pd.Series,
                           ret_from_london_last_daily_close_5: pd.Series,
                           ret_from_london_last_daily_close_6: pd.Series,
                           ret_from_london_last_daily_close_7: pd.Series,
                           ret_from_london_last_daily_close_8: pd.Series,
                           ret_from_london_last_daily_close_9: pd.Series,
                           ret_from_london_last_daily_close_10: pd.Series,
                           ret_from_london_last_daily_close_11: pd.Series,
                           ret_from_london_last_daily_close_12: pd.Series,
                           ret_from_london_last_daily_close_13: pd.Series,
                           ret_from_london_last_daily_close_14: pd.Series,
                           ret_from_london_last_daily_close_15: pd.Series,
                           ret_from_london_last_daily_close_16: pd.Series,
                           ret_from_london_last_daily_close_17: pd.Series,
                           ret_from_london_last_daily_close_18: pd.Series,
                           ret_from_london_last_daily_close_19: pd.Series,
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

    raw_data = group_features.copy()
    add_daily_rolling_features=config.model.features.add_daily_rolling_features
    interval_mins = config.dataset.interval_mins
    new_york_cal = mcal.get_calendar("NYSE")
    lse_cal = mcal.get_calendar("LSE")

    raw_data["week_of_year"] = week_of_year
    raw_data["month_of_year"] = month_of_year

    raw_data["weekly_close_time"] = weekly_close_time
    raw_data["last_weekly_close_time"] = last_weekly_close_time
    raw_data["monthly_close_time"] = monthly_close_time
    raw_data["last_monthly_close_time"] = last_monthly_close_time
    raw_data["option_expiration_time"] = option_expiration_time
    raw_data["last_option_expiration_time"] = last_option_expiration_time

    if macro_data_builder.add_macro_event:
        raw_data["last_macro_event_time_imp1"] = last_macro_event_time_imp1
        raw_data["next_macro_event_time_imp1"] = next_macro_event_time_imp1
        raw_data["last_macro_event_time_imp2"] = last_macro_event_time_imp2
        raw_data["next_macro_event_time_imp2"] = next_macro_event_time_imp2
        raw_data["last_macro_event_time_imp3"] = last_macro_event_time_imp3
        raw_data["next_macro_event_time_imp3"] = next_macro_event_time_imp3
    
    raw_data["last_weekly_close_time_0"] = last_weekly_close_time_0
    raw_data["last_weekly_close_time_1"] = last_weekly_close_time_1
    raw_data["last_weekly_close_time_2"] = last_weekly_close_time_2
    raw_data["last_weekly_close_time_3"] = last_weekly_close_time_3
    raw_data["last_weekly_close_time_4"] = last_weekly_close_time_4
    raw_data["last_weekly_close_time_5"] = last_weekly_close_time_5
    raw_data["last_weekly_close_time_6"] = last_weekly_close_time_6
    raw_data["last_weekly_close_time_7"] = last_weekly_close_time_7
    raw_data["last_weekly_close_time_8"] = last_weekly_close_time_8
    raw_data["last_weekly_close_time_9"] = last_weekly_close_time_9
    raw_data["last_weekly_close_time_10"] = last_weekly_close_time_10
    raw_data["last_weekly_close_time_11"] = last_weekly_close_time_11
    raw_data["last_weekly_close_time_12"] = last_weekly_close_time_12
    raw_data["last_weekly_close_time_13"] = last_weekly_close_time_13
    raw_data["last_weekly_close_time_14"] = last_weekly_close_time_14
    raw_data["last_weekly_close_time_15"] = last_weekly_close_time_15
    raw_data["last_weekly_close_time_16"] = last_weekly_close_time_16
    raw_data["last_weekly_close_time_17"] = last_weekly_close_time_17
    raw_data["last_weekly_close_time_18"] = last_weekly_close_time_18
    raw_data["last_weekly_close_time_19"] = last_weekly_close_time_19

    raw_data["last_monthly_close_time_0"] = last_monthly_close_time_0
    raw_data["last_monthly_close_time_1"] = last_monthly_close_time_1
    raw_data["last_monthly_close_time_2"] = last_monthly_close_time_2
    raw_data["last_monthly_close_time_3"] = last_monthly_close_time_3
    raw_data["last_monthly_close_time_4"] = last_monthly_close_time_4
    raw_data["last_monthly_close_time_5"] = last_monthly_close_time_5
    raw_data["last_monthly_close_time_6"] = last_monthly_close_time_6
    raw_data["last_monthly_close_time_7"] = last_monthly_close_time_7
    raw_data["last_monthly_close_time_8"] = last_monthly_close_time_8
    raw_data["last_monthly_close_time_9"] = last_monthly_close_time_9
    raw_data["last_monthly_close_time_10"] = last_monthly_close_time_10
    raw_data["last_monthly_close_time_11"] = last_monthly_close_time_11
    raw_data["last_monthly_close_time_12"] = last_monthly_close_time_12
    raw_data["last_monthly_close_time_13"] = last_monthly_close_time_13
    raw_data["last_monthly_close_time_14"] = last_monthly_close_time_14
    raw_data["last_monthly_close_time_15"] = last_monthly_close_time_15
    raw_data["last_monthly_close_time_16"] = last_monthly_close_time_16
    raw_data["last_monthly_close_time_17"] = last_monthly_close_time_17
    raw_data["last_monthly_close_time_18"] = last_monthly_close_time_18
    raw_data["last_monthly_close_time_19"] = last_monthly_close_time_19
    
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
        raw_data["vwap_pre_macro_event_imp1"] = vwap_pre_macro_event_imp1
        raw_data["ret_from_vwap_pre_macro_event_imp1"] = ret_from_vwap_pre_macro_event_imp1
        raw_data["vwap_post_macro_event_imp1"] = vwap_post_macro_event_imp1
        raw_data["ret_from_vwap_post_macro_event_imp1"] = ret_from_vwap_post_macro_event_imp1
        raw_data["vwap_around_macro_event_imp1"] = vwap_around_macro_event_imp1
        raw_data["ret_from_vwap_around_macro_event_imp1"] = ret_from_vwap_around_macro_event_imp1

        raw_data["ret_from_vwap_since_last_macro_event_imp2"] = ret_from_vwap_since_last_macro_event_imp2
        raw_data["vwap_pre_macro_event_imp2"] = vwap_pre_macro_event_imp2
        raw_data["ret_from_vwap_pre_macro_event_imp2"] = ret_from_vwap_pre_macro_event_imp2
        raw_data["vwap_post_macro_event_imp2"] = vwap_post_macro_event_imp2
        raw_data["ret_from_vwap_post_macro_event_imp2"] = ret_from_vwap_post_macro_event_imp2
        raw_data["vwap_around_macro_event_imp2"] = vwap_around_macro_event_imp2
        raw_data["ret_from_vwap_around_macro_event_imp2"] = ret_from_vwap_around_macro_event_imp2

        raw_data["ret_from_vwap_since_last_macro_event_imp3"] = ret_from_vwap_since_last_macro_event_imp3
        raw_data["vwap_pre_macro_event_imp3"] = vwap_pre_macro_event_imp3
        raw_data["ret_from_vwap_pre_macro_event_imp3"] = ret_from_vwap_pre_macro_event_imp3
        raw_data["vwap_post_macro_event_imp3"] = vwap_post_macro_event_imp3
        raw_data["ret_from_vwap_post_macro_event_imp3"] = ret_from_vwap_post_macro_event_imp3
        raw_data["vwap_around_macro_event_imp3"] = vwap_around_macro_event_imp3
        raw_data["ret_from_vwap_around_macro_event_imp3"] = ret_from_vwap_around_macro_event_imp3
        
    raw_data["ret_from_vwap_since_new_york_open"] = ret_from_vwap_since_new_york_open
    raw_data["ret_from_vwap_since_london_open"] = ret_from_vwap_since_london_open
    
    raw_data["vwap_pre_new_york_open"] = vwap_pre_new_york_open
    raw_data["ret_from_vwap_pre_new_york_open"] = ret_from_vwap_pre_new_york_open

    raw_data["vwap_post_new_york_open"] = vwap_post_new_york_open
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
        interval_per_day = int(23 * 60 / interval_minutes)        
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
        raw_data["daily_vol_diff_1_20d"] = raw_data["daily_vol_1d"]*diff_mul - raw_data["daily_vol_20d"]
        raw_data["daily_skew_1d"] = daily_skew_1d
        raw_data["daily_skew_5d"] = daily_skew_5d
        raw_data["daily_skew_10d"] = daily_skew_10d
        raw_data["daily_skew_20d"] = daily_skew_20d
        raw_data["daily_kurt_1d"] = daily_kurt_1d
        raw_data["daily_kurt_5d"] = daily_kurt_5d
        raw_data["daily_kurt_10d"] = daily_kurt_10d
        raw_data["daily_kurt_20d"] = daily_kurt_20d

    raw_data[f"new_york_last_daily_open_0"] = new_york_last_daily_open_0
    raw_data[f"new_york_last_daily_close_0"] = new_york_last_daily_close_0
    raw_data[f"new_york_last_daily_open_1"] = new_york_last_daily_open_1
    raw_data[f"new_york_last_daily_close_1"] = new_york_last_daily_close_1
    raw_data[f"new_york_last_daily_open_2"] = new_york_last_daily_open_2
    raw_data[f"new_york_last_daily_close_2"] = new_york_last_daily_close_2
    raw_data[f"new_york_last_daily_open_3"] = new_york_last_daily_open_3
    raw_data[f"new_york_last_daily_close_3"] = new_york_last_daily_close_3
    raw_data[f"new_york_last_daily_open_4"] = new_york_last_daily_open_4
    raw_data[f"new_york_last_daily_close_4"] = new_york_last_daily_close_4
    raw_data[f"new_york_last_daily_open_5"] = new_york_last_daily_open_5
    raw_data[f"new_york_last_daily_close_5"] = new_york_last_daily_close_5
    raw_data[f"new_york_last_daily_open_6"] = new_york_last_daily_open_6
    raw_data[f"new_york_last_daily_close_6"] = new_york_last_daily_close_6
    raw_data[f"new_york_last_daily_open_7"] = new_york_last_daily_open_7
    raw_data[f"new_york_last_daily_close_7"] = new_york_last_daily_close_8
    raw_data[f"new_york_last_daily_open_8"] = new_york_last_daily_open_8
    raw_data[f"new_york_last_daily_close_8"] = new_york_last_daily_close_8
    raw_data[f"new_york_last_daily_open_9"] = new_york_last_daily_open_9
    raw_data[f"new_york_last_daily_close_9"] = new_york_last_daily_close_9
    raw_data[f"new_york_last_daily_open_10"] = new_york_last_daily_open_10
    raw_data[f"new_york_last_daily_close_10"] = new_york_last_daily_close_10
    raw_data[f"new_york_last_daily_open_11"] = new_york_last_daily_open_1
    raw_data[f"new_york_last_daily_close_11"] = new_york_last_daily_close_11
    raw_data[f"new_york_last_daily_open_12"] = new_york_last_daily_open_12
    raw_data[f"new_york_last_daily_close_12"] = new_york_last_daily_close_12
    raw_data[f"new_york_last_daily_open_13"] = new_york_last_daily_open_13
    raw_data[f"new_york_last_daily_close_13"] = new_york_last_daily_close_13
    raw_data[f"new_york_last_daily_open_14"] = new_york_last_daily_open_14
    raw_data[f"new_york_last_daily_close_14"] = new_york_last_daily_close_14
    raw_data[f"new_york_last_daily_open_15"] = new_york_last_daily_open_15
    raw_data[f"new_york_last_daily_close_15"] = new_york_last_daily_close_15
    raw_data[f"new_york_last_daily_open_16"] = new_york_last_daily_open_16
    raw_data[f"new_york_last_daily_close_16"] = new_york_last_daily_close_16
    raw_data[f"new_york_last_daily_open_17"] = new_york_last_daily_open_17
    raw_data[f"new_york_last_daily_close_17"] = new_york_last_daily_close_18
    raw_data[f"new_york_last_daily_open_18"] = new_york_last_daily_open_18
    raw_data[f"new_york_last_daily_close_18"] = new_york_last_daily_close_18
    raw_data[f"new_york_last_daily_open_19"] = new_york_last_daily_open_19
    raw_data[f"new_york_last_daily_close_19"] = new_york_last_daily_close_19

    raw_data[f"london_last_daily_open_0"] = london_last_daily_open_0
    raw_data[f"london_last_daily_close_0"] = london_last_daily_close_0
    raw_data[f"london_last_daily_open_1"] = london_last_daily_open_1
    raw_data[f"london_last_daily_close_1"] = london_last_daily_close_1
    raw_data[f"london_last_daily_open_2"] = london_last_daily_open_2
    raw_data[f"london_last_daily_close_2"] = london_last_daily_close_2
    raw_data[f"london_last_daily_open_3"] = london_last_daily_open_3
    raw_data[f"london_last_daily_close_3"] = london_last_daily_close_3
    raw_data[f"london_last_daily_open_4"] = london_last_daily_open_4
    raw_data[f"london_last_daily_close_4"] = london_last_daily_close_4
    raw_data[f"london_last_daily_open_5"] = london_last_daily_open_5
    raw_data[f"london_last_daily_close_5"] = london_last_daily_close_5
    raw_data[f"london_last_daily_open_6"] = london_last_daily_open_6
    raw_data[f"london_last_daily_close_6"] = london_last_daily_close_6
    raw_data[f"london_last_daily_open_7"] = london_last_daily_open_7
    raw_data[f"london_last_daily_close_7"] = london_last_daily_close_8
    raw_data[f"london_last_daily_open_8"] = london_last_daily_open_8
    raw_data[f"london_last_daily_close_8"] = london_last_daily_close_8
    raw_data[f"london_last_daily_open_9"] = london_last_daily_open_9
    raw_data[f"london_last_daily_close_9"] = london_last_daily_close_9
    raw_data[f"london_last_daily_open_10"] = london_last_daily_open_10
    raw_data[f"london_last_daily_close_10"] = london_last_daily_close_10
    raw_data[f"london_last_daily_open_11"] = london_last_daily_open_1
    raw_data[f"london_last_daily_close_11"] = london_last_daily_close_11
    raw_data[f"london_last_daily_open_12"] = london_last_daily_open_12
    raw_data[f"london_last_daily_close_12"] = london_last_daily_close_12
    raw_data[f"london_last_daily_open_13"] = london_last_daily_open_13
    raw_data[f"london_last_daily_close_13"] = london_last_daily_close_13
    raw_data[f"london_last_daily_open_14"] = london_last_daily_open_14
    raw_data[f"london_last_daily_close_14"] = london_last_daily_close_14
    raw_data[f"london_last_daily_open_15"] = london_last_daily_open_15
    raw_data[f"london_last_daily_close_15"] = london_last_daily_close_15
    raw_data[f"london_last_daily_open_16"] = london_last_daily_open_16
    raw_data[f"london_last_daily_close_16"] = london_last_daily_close_16
    raw_data[f"london_last_daily_open_17"] = london_last_daily_open_17
    raw_data[f"london_last_daily_close_17"] = london_last_daily_close_18
    raw_data[f"london_last_daily_open_18"] = london_last_daily_open_18
    raw_data[f"london_last_daily_close_18"] = london_last_daily_close_18
    raw_data[f"london_last_daily_open_19"] = london_last_daily_open_19
    raw_data[f"london_last_daily_close_19"] = london_last_daily_close_19
                        
    raw_data[f"ret_from_new_york_last_daily_open_0"] = ret_from_new_york_last_daily_open_0
    raw_data[f"ret_from_new_york_last_daily_open_1"] = ret_from_new_york_last_daily_open_1
    raw_data[f"ret_from_new_york_last_daily_open_2"] = ret_from_new_york_last_daily_open_2
    raw_data[f"ret_from_new_york_last_daily_open_3"] = ret_from_new_york_last_daily_open_3
    raw_data[f"ret_from_new_york_last_daily_open_4"] = ret_from_new_york_last_daily_open_4
    raw_data[f"ret_from_new_york_last_daily_open_5"] = ret_from_new_york_last_daily_open_5
    raw_data[f"ret_from_new_york_last_daily_open_6"] = ret_from_new_york_last_daily_open_6
    raw_data[f"ret_from_new_york_last_daily_open_7"] = ret_from_new_york_last_daily_open_7
    raw_data[f"ret_from_new_york_last_daily_open_8"] = ret_from_new_york_last_daily_open_8
    raw_data[f"ret_from_new_york_last_daily_open_9"] = ret_from_new_york_last_daily_open_9
    raw_data[f"ret_from_new_york_last_daily_open_10"] = ret_from_new_york_last_daily_open_10
    raw_data[f"ret_from_new_york_last_daily_open_11"] = ret_from_new_york_last_daily_open_11
    raw_data[f"ret_from_new_york_last_daily_open_12"] = ret_from_new_york_last_daily_open_12
    raw_data[f"ret_from_new_york_last_daily_open_13"] = ret_from_new_york_last_daily_open_13
    raw_data[f"ret_from_new_york_last_daily_open_14"] = ret_from_new_york_last_daily_open_14
    raw_data[f"ret_from_new_york_last_daily_open_15"] = ret_from_new_york_last_daily_open_15
    raw_data[f"ret_from_new_york_last_daily_open_16"] = ret_from_new_york_last_daily_open_16
    raw_data[f"ret_from_new_york_last_daily_open_17"] = ret_from_new_york_last_daily_open_17
    raw_data[f"ret_from_new_york_last_daily_open_18"] = ret_from_new_york_last_daily_open_18
    raw_data[f"ret_from_new_york_last_daily_open_19"] = ret_from_new_york_last_daily_open_19

    raw_data[f"ret_from_new_york_last_daily_close_0"] = ret_from_new_york_last_daily_close_0
    raw_data[f"ret_from_new_york_last_daily_close_1"] = ret_from_new_york_last_daily_close_1
    raw_data[f"ret_from_new_york_last_daily_close_2"] = ret_from_new_york_last_daily_close_2
    raw_data[f"ret_from_new_york_last_daily_close_3"] = ret_from_new_york_last_daily_close_3
    raw_data[f"ret_from_new_york_last_daily_close_4"] = ret_from_new_york_last_daily_close_4
    raw_data[f"ret_from_new_york_last_daily_close_5"] = ret_from_new_york_last_daily_close_5
    raw_data[f"ret_from_new_york_last_daily_close_6"] = ret_from_new_york_last_daily_close_6
    raw_data[f"ret_from_new_york_last_daily_close_7"] = ret_from_new_york_last_daily_close_7
    raw_data[f"ret_from_new_york_last_daily_close_8"] = ret_from_new_york_last_daily_close_8
    raw_data[f"ret_from_new_york_last_daily_close_9"] = ret_from_new_york_last_daily_close_9
    raw_data[f"ret_from_new_york_last_daily_close_10"] = ret_from_new_york_last_daily_close_10
    raw_data[f"ret_from_new_york_last_daily_close_11"] = ret_from_new_york_last_daily_close_11
    raw_data[f"ret_from_new_york_last_daily_close_12"] = ret_from_new_york_last_daily_close_12
    raw_data[f"ret_from_new_york_last_daily_close_13"] = ret_from_new_york_last_daily_close_13
    raw_data[f"ret_from_new_york_last_daily_close_14"] = ret_from_new_york_last_daily_close_14
    raw_data[f"ret_from_new_york_last_daily_close_15"] = ret_from_new_york_last_daily_close_15
    raw_data[f"ret_from_new_york_last_daily_close_16"] = ret_from_new_york_last_daily_close_16
    raw_data[f"ret_from_new_york_last_daily_close_17"] = ret_from_new_york_last_daily_close_17
    raw_data[f"ret_from_new_york_last_daily_close_18"] = ret_from_new_york_last_daily_close_18
    raw_data[f"ret_from_new_york_last_daily_close_19"] = ret_from_new_york_last_daily_close_19

    raw_data[f"ret_from_london_last_daily_open_0"] = ret_from_london_last_daily_open_0
    raw_data[f"ret_from_london_last_daily_open_1"] = ret_from_london_last_daily_open_1
    raw_data[f"ret_from_london_last_daily_open_2"] = ret_from_london_last_daily_open_2
    raw_data[f"ret_from_london_last_daily_open_3"] = ret_from_london_last_daily_open_3
    raw_data[f"ret_from_london_last_daily_open_4"] = ret_from_london_last_daily_open_4
    raw_data[f"ret_from_london_last_daily_open_5"] = ret_from_london_last_daily_open_5
    raw_data[f"ret_from_london_last_daily_open_6"] = ret_from_london_last_daily_open_6
    raw_data[f"ret_from_london_last_daily_open_7"] = ret_from_london_last_daily_open_7
    raw_data[f"ret_from_london_last_daily_open_8"] = ret_from_london_last_daily_open_8
    raw_data[f"ret_from_london_last_daily_open_9"] = ret_from_london_last_daily_open_9
    raw_data[f"ret_from_london_last_daily_open_10"] = ret_from_london_last_daily_open_10
    raw_data[f"ret_from_london_last_daily_open_11"] = ret_from_london_last_daily_open_11
    raw_data[f"ret_from_london_last_daily_open_12"] = ret_from_london_last_daily_open_12
    raw_data[f"ret_from_london_last_daily_open_13"] = ret_from_london_last_daily_open_13
    raw_data[f"ret_from_london_last_daily_open_14"] = ret_from_london_last_daily_open_14
    raw_data[f"ret_from_london_last_daily_open_15"] = ret_from_london_last_daily_open_15
    raw_data[f"ret_from_london_last_daily_open_16"] = ret_from_london_last_daily_open_16
    raw_data[f"ret_from_london_last_daily_open_17"] = ret_from_london_last_daily_open_17
    raw_data[f"ret_from_london_last_daily_open_18"] = ret_from_london_last_daily_open_18
    raw_data[f"ret_from_london_last_daily_open_19"] = ret_from_london_last_daily_open_19

    raw_data[f"ret_from_london_last_daily_close_0"] = ret_from_london_last_daily_close_0
    raw_data[f"ret_from_london_last_daily_close_1"] = ret_from_london_last_daily_close_1
    raw_data[f"ret_from_london_last_daily_close_2"] = ret_from_london_last_daily_close_2
    raw_data[f"ret_from_london_last_daily_close_3"] = ret_from_london_last_daily_close_3
    raw_data[f"ret_from_london_last_daily_close_4"] = ret_from_london_last_daily_close_4
    raw_data[f"ret_from_london_last_daily_close_5"] = ret_from_london_last_daily_close_5
    raw_data[f"ret_from_london_last_daily_close_6"] = ret_from_london_last_daily_close_6
    raw_data[f"ret_from_london_last_daily_close_7"] = ret_from_london_last_daily_close_7
    raw_data[f"ret_from_london_last_daily_close_8"] = ret_from_london_last_daily_close_8
    raw_data[f"ret_from_london_last_daily_close_9"] = ret_from_london_last_daily_close_9
    raw_data[f"ret_from_london_last_daily_close_10"] = ret_from_london_last_daily_close_10
    raw_data[f"ret_from_london_last_daily_close_11"] = ret_from_london_last_daily_close_11
    raw_data[f"ret_from_london_last_daily_close_12"] = ret_from_london_last_daily_close_12
    raw_data[f"ret_from_london_last_daily_close_13"] = ret_from_london_last_daily_close_13
    raw_data[f"ret_from_london_last_daily_close_14"] = ret_from_london_last_daily_close_14
    raw_data[f"ret_from_london_last_daily_close_15"] = ret_from_london_last_daily_close_15
    raw_data[f"ret_from_london_last_daily_close_16"] = ret_from_london_last_daily_close_16
    raw_data[f"ret_from_london_last_daily_close_17"] = ret_from_london_last_daily_close_17
    raw_data[f"ret_from_london_last_daily_close_18"] = ret_from_london_last_daily_close_18
    raw_data[f"ret_from_london_last_daily_close_19"] = ret_from_london_last_daily_close_19

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

    raw_data["next_new_york_close"] = next_new_york_close
    raw_data["next_weekly_close"] = next_weekly_close
    raw_data["next_monthly_close"] = next_monthly_close
    raw_data["ret_to_next_new_york_close"] = ret_to_next_new_york_close
    raw_data["ret_to_next_weekly_close"] = ret_to_next_weekly_close
    raw_data["ret_to_next_monthly_close"] = ret_to_next_monthly_close
    raw_data["last_option_expiration_close"] = last_option_expiration_close

                                                        
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

    
    logging.error(f"sampled_raw:{raw_data.iloc[-10:]}")
    return raw_data
