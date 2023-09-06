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

@parameterize(
    time_to_new_york_open={"diff_time": source("new_york_open_time")},
    time_to_new_york_last_open={"diff_time": source("new_york_last_open_time")},
    time_to_new_york_last_close={"diff_time": source("new_york_last_close_time")},
    time_to_new_york_close={"diff_time": source("new_york_close_time")},
    time_to_london_open={"diff_time": source("london_open_time")},
    time_to_london_last_open={"diff_time": source("london_last_open_time")},
    time_to_london_last_close={"diff_time": source("london_last_close_time")},
    time_to_london_close={"diff_time": source("london_close_time")},
    time_to_weekly_close={"diff_time": source("weekly_close_time")},
    time_to_monthly_close={"diff_time": source("monthly_close_time")},
    time_to_option_expiration={"diff_time": source("option_expiration_time")},
    time_to_last_macro_event_imp1={"diff_time": source("last_macro_event_time_imp1")},
    time_to_last_macro_event_imp2={"diff_time": source("last_macro_event_time_imp2")},
    time_to_last_macro_event_imp3={"diff_time": source("last_macro_event_time_imp3")},
    time_to_next_macro_event_imp1={"diff_time": source("next_macro_event_time_imp1")},
    time_to_next_macro_event_imp2={"diff_time": source("next_macro_event_time_imp2")},
    time_to_next_macro_event_imp3={"diff_time": source("next_macro_event_time_imp3")},
    time_to_high_1_ff={"diff_time": source("time_high_1_ff")},
    time_to_high_5_ff={"diff_time": source("time_high_5_ff")},
    time_to_high_11_ff={"diff_time": source("time_high_11_ff")},
    time_to_high_21_ff={"diff_time": source("time_high_21_ff")},
    time_to_high_51_ff={"diff_time": source("time_high_51_ff")},
    time_to_high_101_ff={"diff_time": source("time_high_101_ff")},
    time_to_high_201_ff={"diff_time": source("time_high_201_ff")},
    time_to_low_1_ff={"diff_time": source("time_low_1_ff")},
    time_to_low_5_ff={"diff_time": source("time_low_5_ff")},
    time_to_low_11_ff={"diff_time": source("time_low_11_ff")},
    time_to_low_21_ff={"diff_time": source("time_low_21_ff")},
    time_to_low_51_ff={"diff_time": source("time_low_51_ff")},
    time_to_low_101_ff={"diff_time": source("time_low_101_ff")},
    time_to_low_201_ff={"diff_time": source("time_low_201_ff")},
    time_to_high_1d_ff_shift_1d={"diff_time": source("time_high_1d_ff_shift_1d")},
    time_to_low_1d_ff_shift_1d={"diff_time": source("time_low_1d_ff_shift_1d")},
    time_to_high_5d_ff_shift_5d={"diff_time": source("time_high_5d_ff_shift_5d")},
    time_to_low_5d_ff_shift_5d={"diff_time": source("time_low_5d_ff_shift_5d")},
    time_to_high_11d_ff_shift_11d={"diff_time": source("time_high_11d_ff_shift_11d")},
    time_to_low_11d_ff_shift_11d={"diff_time": source("time_low_11d_ff_shift_11d")},
    time_to_high_21d_ff_shift_21d={"diff_time": source("time_high_21d_ff_shift_21d")},
    time_to_low_21d_ff_shift_21d={"diff_time": source("time_low_21d_ff_shift_21d")},
    time_to_high_51d_ff_shift_51d={"diff_time": source("time_high_51d_ff_shift_51d")},
    time_to_low_51d_ff_shift_51d={"diff_time": source("time_low_51d_ff_shift_51d")},
    time_to_high_101d_ff_shift_101d={"diff_time": source("time_high_101d_ff_shift_101d")},
    time_to_low_101d_ff_shift_101d={"diff_time": source("time_low_101d_ff_shift_101d")},
    time_to_high_201d_ff_shift_201d={"diff_time": source("time_high_201d_ff_shift_201d")},
    time_to_low_201d_ff_shift_201d={"diff_time": source("time_low_201d_ff_shift_201d")},
    time_to_high_1d_ff={"diff_time": source("time_high_1d_ff")},
    time_to_low_1d_ff={"diff_time": source("time_low_1d_ff")},
    time_to_high_5d_ff={"diff_time": source("time_high_5d_ff")},
    time_to_low_5d_ff={"diff_time": source("time_low_5d_ff")},
    time_to_high_11d_ff={"diff_time": source("time_high_11d_ff")},
    time_to_low_11d_ff={"diff_time": source("time_low_11d_ff")},
    time_to_high_21d_ff={"diff_time": source("time_high_21d_ff")},
    time_to_low_21d_ff={"diff_time": source("time_low_21d_ff")},
    time_to_high_51d_ff={"diff_time": source("time_high_51d_ff")},
    time_to_low_51d_ff={"diff_time": source("time_low_51d_ff")},
    time_to_high_101d_ff={"diff_time": source("time_high_101d_ff")},
    time_to_low_101d_ff={"diff_time": source("time_low_101d_ff")},
    time_to_high_201d_ff={"diff_time": source("time_high_201d_ff")},
    time_to_low_201d_ff={"diff_time": source("time_low_201d_ff")},
)
def time_to(timestamp:pd.Series, diff_time:pd.Series) -> pd.Series:
    return timestamp-diff_time


def example_level_features(group_features:pd.DataFrame, cal:CMEEquityExchangeCalendar,
                           macro_data_builder:MacroDataBuilder, config:DictConfig,
                           time_features: pd.DataFrame,
                           month: pd.Series, year:pd.Series, hour_of_day:pd.Series,
                           time_to_low_1d_ff_shift_1d: pd.Series, time_to_low_5d_ff_shift_5d: pd.Series,
                           time_to_low_11d_ff_shift_11d: pd.Series, time_to_low_21d_ff_shift_21d: pd.Series,
                           time_to_high_1d_ff_shift_1d: pd.Series, time_to_high_5d_ff_shift_5d: pd.Series,
                           time_to_high_11d_ff_shift_11d: pd.Series, time_to_high_21d_ff_shift_21d: pd.Series,
                           time_to_low_1_ff: pd.Series, time_to_low_5_ff: pd.Series,
                           time_to_low_11_ff: pd.Series, time_to_low_21_ff: pd.Series,
                           time_to_low_51_ff: pd.Series, time_to_low_101_ff: pd.Series,
                           time_to_low_201_ff: pd.Series,
                           time_to_high_1_ff: pd.Series, time_to_high_5_ff: pd.Series,
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
                           time_to_london_last_close:pd.Series) -> pd.DataFrame:
    add_daily_rolling_features=config.model.features.add_daily_rolling_features
    new_york_cal = mcal.get_calendar("NYSE")
    lse_cal = mcal.get_calendar("LSE")
    raw_data = group_features
    
    raw_data["week_of_year"] = time_features["week_of_year"]
    raw_data["month_of_year"] = time_features["month_of_year"]

    raw_data["weekly_close_time"] = time_features["weekly_close_time"]
    raw_data["last_weekly_close_time"] = time_features["last_weekly_close_time"]
    raw_data["monthly_close_time"] = time_features["monthly_close_time"]
    raw_data["last_monthly_close_time"] = time_features["last_monthly_close_time"]
    raw_data["option_expiration_time"] = time_features["option_expiration_time"]
    raw_data["last_option_expiration_time"] = time_features["last_option_expiration_time"]

    if macro_data_builder.add_macro_event:
        raw_data["last_macro_event_time_imp1"] = time_features["last_macro_event_time_imp1"]
        raw_data["next_macro_event_time_imp1"] = time_features["next_macro_event_time_imp1"]
        raw_data["last_macro_event_time_imp2"] = time_features["last_macro_event_time_imp2"]
        raw_data["next_macro_event_time_imp2"] = time_features["next_macro_event_time_imp2"]
        raw_data["last_macro_event_time_imp3"] = time_features["last_macro_event_time_imp3"]
        raw_data["next_macro_event_time_imp3"] = time_features["next_macro_event_time_imp3"]

    for idx in range(config.model.daily_lookback):
        raw_data[f"new_york_last_open_time_{idx}"] = time_features[f"new_york_last_open_time_{idx}"] 
        raw_data[f"new_york_last_close_time_{idx}"] = time_features[f"new_york_last_close_time_{idx}"]
        raw_data[f"london_last_open_time_{idx}"] = time_features[f"london_last_open_time_{idx}"]
        raw_data[f"london_last_close_time_{idx}"] = time_features[f"london_last_close_time_{idx}"]

    raw_data[f"new_york_last_open_time"] = raw_data[f"new_york_last_open_time_0"]
    raw_data[f"new_york_last_close_time"] = raw_data[f"new_york_last_close_time_0"]
    raw_data[f"london_last_open_time"] = raw_data[f"london_last_open_time_0"]
    raw_data[f"london_last_close_time"] = raw_data[f"london_last_close_time_0"]

    raw_data["new_york_open_time"] = time_features[f"new_york_open_time"]
    raw_data["new_york_close_time"] = time_features[f"new_york_close_time"]
    raw_data["london_open_time"] = time_features[f"london_open_time"]
    raw_data["london_close_time"] = time_features[f"london_close_time"]
    raw_data["time_to_new_york_open"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="new_york_open_time"
    )
    raw_data["time_to_new_york_last_open"] = time_to_new_york_last_open
    raw_data["time_to_new_york_last_close"] = time_to_new_york_last_close
    raw_data["time_to_new_york_close"] = time_to_new_york_close
    raw_data["time_to_london_open"] = time_to_london_open
    raw_data["time_to_london_last_open"] = time_to_london_last_open
    raw_data["time_to_london_last_close"] = time_to_london_last_close
    raw_data["time_to_london_close"] = time_to_london_close

    raw_data["month"] = month
    raw_data["year"] = year
    raw_data["hour_of_day"] = hour_of_day
    raw_data["day_of_week"] = day_of_week
    raw_data["day_of_month"] = day_of_month
    # TODO: use business time instead of calendar time. this changes a lot
    # during new year.
    raw_data["time_to_weekly_close"] = time_to_weekly_close
    raw_data["time_to_monthly_close"] = time_to_monthly_close
    raw_data["time_to_option_expiration"] = time_to_option_expiration
    if macro_data_builder.add_macro_event:
        raw_data["time_to_last_macro_event_imp1"] = time_to_last_macro_event_imp1
        raw_data["time_to_next_macro_event_imp1"] = time_to_next_macro_event_imp1
        raw_data["time_to_last_macro_event_imp2"] = time_to_last_macro_event_imp2
        raw_data["time_to_next_macro_event_imp2"] = time_to_next_macro_event_imp2
        raw_data["time_to_last_macro_event_imp3"] = time_to_last_macro_event_imp3
        raw_data["time_to_next_macro_event_imp3"] = time_to_next_macro_event_imp3

    raw_data["time_to_high_5_ff"] = time_to_high_5_ff
    raw_data["time_to_low_5_ff"] = time_to_low_5_ff
    raw_data["time_to_high_11_ff"] = time_to_high_11_ff
    raw_data["time_to_low_11_ff"] = time_to_low_11_ff
    raw_data["time_to_high_21_ff"] = time_to_high_21_ff
    raw_data["time_to_low_21_ff"] = time_to_low_21_ff
    raw_data["time_to_high_51_ff"] = time_to_high_51_ff
    raw_data["time_to_low_51_ff"] = time_to_low_51_ff
    raw_data["time_to_high_101_ff"] = time_to_high_101_ff
    raw_data["time_to_low_101_ff"] = time_to_low_101_ff
    raw_data["time_to_high_201_ff"] = time_to_high_201_ff
    raw_data["time_to_low_201_ff"] = time_to_low_201_ff

    if add_daily_rolling_features:
        raw_data["time_to_high_1d_ff_shift_1d"] = time_to_high_1d_ff_shift_1d
        raw_data["time_to_low_1d_ff_shift_1d"] = time_to_low_1d_ff_shift_1d
        raw_data["time_to_low_1d_ff_shift_1d"] = raw_data.time_to_low_1d_ff_shift_1d.apply(
            lambda x : (x-20000)/12000)
        raw_data["time_to_high_1d_ff_shift_1d"] = raw_data.time_to_high_1d_ff_shift_1d.apply(
            lambda x : (x-20000)/12000)
        raw_data["time_to_high_1d_ff"] = time_to_high_1d_ff
        raw_data["time_to_low_1d_ff"] = time_to_low_1d_ff
        raw_data["time_to_high_5d_ff"] = time_to_high_5d_ff
        raw_data["time_to_low_5d_ff"] = time_to_low_5d_ff
        raw_data["time_to_high_5d_ff_shift_5d"] = time_to_high_5d_ff_shift_5d
        raw_data["time_to_low_5d_ff_shift_5d"] = time_to_low_5d_ff_shift_5d
        raw_data["time_to_high_11d_ff_shift_11d"] = time_to_high_11d_ff_shift_11d
        raw_data["time_to_low_11d_ff_shift_11d"] = time_to_low_11d_ff_shift_11d
        raw_data["time_to_high_11d_ff"] = time_to_high_11d_ff
        raw_data["time_to_low_11d_ff"] = time_to_low_11d_ff
        raw_data["time_to_high_21d_ff"] = time_to_high_21d_ff
        raw_data["time_to_low_21d_ff"] = time_to_low_21d_ff
        raw_data["time_to_high_21d_ff_shift_21d"] = time_to_high_21d_ff_shift_21d
        raw_data["time_to_low_21d_ff_shift_21d"] = time_to_low_21d_ff_shift_21d
        raw_data["time_to_high_51d_ff"] = time_to_high_51d_ff
        raw_data["time_to_low_51d_ff"] = time_to_low_51d_ff
        raw_data["time_to_high_201d_ff"] = time_to_high_201d_ff
        raw_data["time_to_low_201d_ff"] = time_to_low_201d_ff

    return raw_data

def pre_macro_event_cum_dv_imp1(example_level_features:pd.DataFrame) -> pd.Series:
    return example_level_features.apply(fill_cum_dv, time_col="last_macro_event_time_imp1",
                          pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)

def pre_macro_event_cum_volume_imp1(example_level_features:pd.DataFrame) -> pd.Series:
    return example_level_features.apply(fill_cum_volume, time_col="last_macro_event_time_imp1",
                                        pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)
def pre_macro_event_cum_dv_imp2(example_level_features:pd.DataFrame) -> pd.Series:
    return example_level_features.apply(fill_cum_dv, time_col="last_macro_event_time_imp2",
                          pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)

def pre_macro_event_cum_volume_imp2(example_level_features:pd.DataFrame) -> pd.Series:
    return example_level_features.apply(fill_cum_volume, time_col="last_macro_event_time_imp2",
                                        pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)

def pre_macro_event_cum_dv_imp3(example_level_features:pd.DataFrame) -> pd.Series:
    return example_level_features.apply(fill_cum_dv, time_col="last_macro_event_time_imp3",
                          pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)

def pre_macro_event_cum_volume_imp3(example_level_features:pd.DataFrame) -> pd.Series:
    return example_level_features.apply(fill_cum_volume, time_col="last_macro_event_time_imp3",
                                        pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)

@parameterize(
    ret_from_vwap_since_last_macro_event_imp1={"time_col": value("last_macro_event_time_imp1")},
    ret_from_vwap_since_last_macro_event_imp2={"time_col": value("last_macro_event_time_imp2")},
    ret_from_vwap_since_last_macro_event_imp3={"time_col": value("last_macro_event_time_imp3")},
    ret_from_vwap_since_new_york_open={"time_col": value("new_york_last_open_time")},
    ret_from_vwap_since_london_open={"time_col": value("london_last_open_time")},
)
def ret_from_vwap(example_level_features:pd.DataFrame, time_col:str, interval_mins: int) ->pd.Series:
    raw_data = example_level_features
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
def around_vwap(example_level_features:pd.DataFrame, time_col:str, interval_mins:int) ->pd.Series:
    raw_data = example_level_features
    cum_dv_col = f"around_{time_col}_cum_dv"
    cum_volume_col = f"around_{time_col}_cum_volume"
    raw_data[cum_dv_col] = raw_data.apply(fill_cum_dv, time_col=time_col,
                                          pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3, axis=1)
    raw_data[cum_volume_col] = raw_data.apply(fill_cum_volume, time_col=time_col,
                                              pre_interval_mins=interval_mins*3, post_interval_mins=interval_mins*3, axis=1)
    rol = raw_data[dv_col].rolling(window=2)

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
def post_vwap(example_level_features:pd.DataFrame, time_col:str) ->pd.Series:
    raw_data = example_level_features
    cum_dv_col = f"post_{time_col}_cum_dv"
    cum_volume_col = f"post_{time_col}_cum_volume"
    raw_data[cum_dv_col] = raw_data.apply(fill_cum_dv, time_col=time_col,
                                          pre_interval_mins=0, post_interval_mins=interval_mins*3, axis=1)
    raw_data[cum_volume_col] = raw_data.apply(fill_cum_volume, time_col=time_col,
                                              pre_interval_mins=0, post_interval_mins=interval_mins*3, axis=1)
    rol = raw_data[dv_col].rolling(window=2)

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
def pre_vwap(example_level_features:pd.DataFrame, time_col:str) ->pd.Series:
    raw_data = example_level_features
    cum_dv_col = f"pre_{time_col}_cum_dv"
    cum_volume_col = f"pre_{time_col}_cum_volume"
    raw_data[cum_dv_col] = raw_data.apply(fill_cum_dv, time_col=time_col,
                                          pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)
    raw_data[cum_volume_col] = raw_data.apply(fill_cum_volume, time_col=time_col,
                                              pre_interval_mins=interval_mins*3, post_interval_mins=0, axis=1)
    rol = raw_data[dv_col].rolling(window=2)

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

        
def close_back_cumsum(example_level_features:pd.DataFrame) -> pd.Series:
    return example_level_features["close_back_cumsum"]

@parameterize(
    close_back_cumsum_high_5d_ff={"lookback_days": value(5)},
    close_back_cumsum_high_11d_ff={"lookback_days": value(11)},
    close_back_cumsum_high_21d_ff={"lookback_days": value(21)},
    close_back_cumsum_high_51d_ff={"lookback_days": value(51)},
    close_back_cumsum_high_101d_ff={"lookback_days": value(101)},
    close_back_cumsum_high_201d_ff={"lookback_days": value(201)},
)
def close_back_cumsum_high_ff(close_back_cumsum: pd.Series, interval_per_day: int, lookback_days: int) -> pd.Series:
    close_back_cumsum_rolling_max = close_back_cumsum.rolling(lookback_days*interval_per_day).max()
    return close_back_cumsum_rolling_max.ffill()

@parameterize(
    close_back_cumsum_low_5d_ff={"lookback_days": value(5)},
    close_back_cumsum_low_11d_ff={"lookback_days": value(11)},
    close_back_cumsum_low_21d_ff={"lookback_days": value(21)},
    close_back_cumsum_low_51d_ff={"lookback_days": value(51)},
    close_back_cumsum_low_101d_ff={"lookback_days": value(101)},
    close_back_cumsum_low_201d_ff={"lookback_days": value(201)},
)
def close_back_cumsum_low_ff(close_back_cumsum: pd.Series, interval_per_day: int, lookback_days: int) -> pd.Series:
    close_back_cumsum_rolling_min = close_back_cumsum.rolling(lookback_days*interval_per_day).min()
    return close_back_cumsum_rolling_min.ffill()

@parameterize(
    ret_from_close_cumsum_high_5d={"close_back_cumsum_high_ff": source("close_back_cumsum_high_5d_ff")},
    ret_from_close_cumsum_high_11d={"close_back_cumsum_high_ff": source("close_back_cumsum_high_11d_ff")},
    ret_from_close_cumsum_high_21d={"close_back_cumsum_high_ff": source("close_back_cumsum_high_21d_ff")},
    ret_from_close_cumsum_high_51d={"close_back_cumsum_high_ff": source("close_back_cumsum_high_51d_ff")},
    ret_from_close_cumsum_high_101d={"close_back_cumsum_high_ff": source("close_back_cumsum_high_101d_ff")},
    ret_from_close_cumsum_high_201d={"close_back_cumsum_high_ff": source("close_back_cumsum_high_201d_ff")},
)
def ret_from_close_cumsum_high(close_back_cumsum: pd.Series, close_back_cumsum_high_ff: pd.Series) -> pd.Series:
    return close_back_cumsum_high_ff - close_back_cumsum

@parameterize(
    ret_from_close_cumsum_low_5d={"close_back_cumsum_low_ff": source("close_back_cumsum_low_5d_ff")},
    ret_from_close_cumsum_low_11d={"close_back_cumsum_low_ff": source("close_back_cumsum_low_11d_ff")},
    ret_from_close_cumsum_low_21d={"close_back_cumsum_low_ff": source("close_back_cumsum_low_21d_ff")},
    ret_from_close_cumsum_low_51d={"close_back_cumsum_low_ff": source("close_back_cumsum_low_51d_ff")},
    ret_from_close_cumsum_low_101d={"close_back_cumsum_low_ff": source("close_back_cumsum_low_101d_ff")},
    ret_from_close_cumsum_low_201d={"close_back_cumsum_low_ff": source("close_back_cumsum_low_201d_ff")},
)
def ret_from_close_cumsum_low(close_back_cumsum: pd.Series, close_back_cumsum_low_ff: pd.Series) -> pd.Series:
    return close_back_cumsum_low_ff - close_back_cumsum


#@profile_util.profile
def example_group_features(cal:CMEEquityExchangeCalendar, macro_data_builder:MacroDataBuilder,
                           example_level_features:pd.DataFrame, config:DictConfig,
                           time_to_new_york_open: pd.Series,
                           time_to_new_york_close: pd.Series,
                           time_to_london_open: pd.Series,
                           time_to_london_close: pd.Series,
                           time_to_new_york_last_open: pd.Series,
                           time_to_new_york_last_close: pd.Series,
                           time_to_london_last_open: pd.Series,
                           time_to_london_last_close: pd.Series,
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
                           ret_from_vwap_since_new_york_open: pd.Series,
                           ret_from_vwap_since_london_open: pd.Series,
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
                           vwap_post_london_close: pd.Series) ->pd.DataFrame:
    raw_data = example_level_features
    add_daily_rolling_features=config.model.features.add_daily_rolling_features
    interval_mins = config.dataset.interval_mins
    new_york_cal = mcal.get_calendar("NYSE")
    lse_cal = mcal.get_calendar("LSE")

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
        raw_data["ret_from_vwap_since_last_macro_event_imp1"] = ret_from_vwap_since_last_macro_event_imp1
        raw_data["vwap_pre_macro_event_imp1"] = vwap_pre_macro_event_imp1
        raw_data["ret_from_vwap_pre_macro_event_imp1"] = raw_data.apply(compute_ret, base_col="vwap_pre_macro_event_imp1", axis=1)
        raw_data["vwap_post_macro_event_imp1"] = vwap_post_macro_event_imp1
        raw_data["ret_from_vwap_post_macro_event_imp1"] = raw_data.apply(compute_ret, base_col="vwap_post_macro_event_imp1", axis=1)
        raw_data["vwap_around_macro_event_imp1"] = vwap_around_macro_event_imp1
        raw_data["ret_from_vwap_around_macro_event_imp1"] = raw_data.apply(compute_ret, base_col="vwap_around_macro_event_imp1", axis=1)

        raw_data["ret_from_vwap_since_last_macro_event_imp2"] = ret_from_vwap_since_last_macro_event_imp2
        raw_data["vwap_pre_macro_event_imp2"] = vwap_pre_macro_event_imp2
        raw_data["ret_from_vwap_pre_macro_event_imp2"] = raw_data.apply(compute_ret, base_col="vwap_pre_macro_event_imp2", axis=1)
        raw_data["vwap_post_macro_event_imp2"] = vwap_post_macro_event_imp2
        raw_data["ret_from_vwap_post_macro_event_imp2"] = raw_data.apply(compute_ret, base_col="vwap_post_macro_event_imp2", axis=1)
        raw_data["vwap_around_macro_event_imp2"] = vwap_around_macro_event_imp2
        raw_data["ret_from_vwap_around_macro_event_imp2"] = raw_data.apply(compute_ret, base_col="vwap_around_macro_event_imp2", axis=1)

        raw_data["ret_from_vwap_since_last_macro_event_imp3"] = ret_from_vwap_since_last_macro_event_imp3
        raw_data["vwap_pre_macro_event_imp3"] = vwap_pre_macro_event_imp3
        raw_data["ret_from_vwap_pre_macro_event_imp3"] = raw_data.apply(compute_ret, base_col="vwap_pre_macro_event_imp3", axis=1)
        raw_data["vwap_post_macro_event_imp3"] = vwap_post_macro_event_imp3
        raw_data["ret_from_vwap_post_macro_event_imp3"] = raw_data.apply(compute_ret, base_col="vwap_post_macro_event_imp3", axis=1)
        raw_data["vwap_around_macro_event_imp3"] = vwap_around_macro_event_imp3
        raw_data["ret_from_vwap_around_macro_event_imp3"] = raw_data.apply(compute_ret, base_col="vwap_around_macro_event_imp3", axis=1)
        
    raw_data["ret_from_vwap_since_new_york_open"] = ret_from_vwap_since_new_york_open
    raw_data["ret_from_vwap_since_london_open"] = ret_from_vwap_since_london_open
    
    raw_data["vwap_pre_new_york_open"] = vwap_pre_new_york_open
    raw_data["ret_from_vwap_pre_new_york_open"] = raw_data.apply(compute_ret, base_col="vwap_pre_new_york_open", axis=1)

    raw_data["vwap_post_new_york_open"] = vwap_post_new_york_open
    raw_data["ret_from_vwap_post_new_york_open"] = raw_data.apply(compute_ret, base_col="vwap_post_new_york_open", axis=1)

    for idx in range(config.model.daily_lookback):
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

                                                        
    raw_data["vwap_around_new_york_open"] = vwap_around_new_york_open
    raw_data["ret_from_vwap_around_new_york_open"] = raw_data.apply(compute_ret, base_col="vwap_around_new_york_open", axis=1)

    raw_data["vwap_pre_new_york_close"] = vwap_pre_new_york_close
    raw_data["ret_from_vwap_pre_new_york_close"] = raw_data.apply(compute_ret, base_col="vwap_pre_new_york_close", axis=1)

    raw_data["vwap_post_new_york_close"] = vwap_post_new_york_close
    raw_data["ret_from_vwap_post_new_york_close"] = raw_data.apply(compute_ret, base_col="vwap_post_new_york_close", axis=1)

    raw_data["vwap_around_new_york_close"] = vwap_around_new_york_close
    raw_data["ret_from_vwap_around_new_york_close"] = raw_data.apply(compute_ret, base_col="vwap_around_new_york_close", axis=1)

    raw_data["vwap_pre_london_open"] = vwap_pre_london_open
    raw_data["ret_from_vwap_pre_london_open"] = raw_data.apply(compute_ret, base_col="vwap_pre_london_open", axis=1)

    raw_data["vwap_post_london_open"] = vwap_post_london_open
    raw_data["ret_from_vwap_post_london_open"] = raw_data.apply(compute_ret, base_col="vwap_post_london_open", axis=1)

    raw_data["vwap_around_london_open"] = vwap_around_london_open
    raw_data["ret_from_vwap_around_london_open"] = raw_data.apply(compute_ret, base_col="vwap_around_london_open", axis=1)

    raw_data["vwap_pre_london_close"] = vwap_pre_london_close
    raw_data["ret_from_vwap_pre_london_close"] = raw_data.apply(compute_ret, base_col="vwap_pre_london_close", axis=1)

    raw_data["vwap_post_london_close"] = vwap_post_london_close
    raw_data["ret_from_vwap_post_london_close"] = raw_data.apply(compute_ret, base_col="vwap_post_london_close", axis=1)

    raw_data["vwap_around_london_close"] = vwap_around_london_close
    raw_data["ret_from_vwap_around_london_close"] = raw_data.apply(compute_ret, base_col="vwap_around_london_close", axis=1)

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

    logging.error(f"sampled_raw:{raw_data.iloc[-10:]}")
    return raw_data
