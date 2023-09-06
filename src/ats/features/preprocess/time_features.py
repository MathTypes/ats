import datetime
import logging
import math
import os
from typing import Dict

from hamilton.function_modifiers import tag, does, extract_columns, parameterize, source, value

from numba import njit
import numpy as np
import pandas as pd
from pyarrow import csv
import ray
import ta

from omegaconf.dictconfig import DictConfig
from pandas_market_calendars.exchange_calendar_cme import CMEEquityExchangeCalendar
from ats.event.macro_indicator import MacroDataBuilder
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
from ats.features.preprocess.feature_utils import *

VOL_THRESHOLD = 5  # multiple to winsorise by
HALFLIFE_WINSORISE = 252

def timestamp(sorted_data: pd.DataFrame) -> pd.Series:
    return sorted_data["timestamp"]

def time(sorted_data: pd.DataFrame) -> pd.Series:
    return sorted_data["time"]

@tag(cache="parquet")
def new_york_open_time(timestamp: pd.Series) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_next_open_time, cal=new_york_cal
    )

@tag(cache="parquet")
def new_york_close_time(timestamp: pd.Series) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_next_close_time, cal=new_york_cal
    )

@tag(cache="parquet")
def london_open_time(timestamp: pd.Series) -> pd.Series:
    lse_cal = mcal.get_calendar("LSE")
    return timestamp.apply(
        market_time.compute_next_open_time, cal=lse_cal
    )

@tag(cache="parquet")
def london_close_time(timestamp: pd.Series) -> pd.Series:
    lse_cal = mcal.get_calendar("LSE")
    return timestamp.apply(
        market_time.compute_next_close_time, cal=lse_cal
    )

@tag(cache="parquet")
def weekly_close_time(timestamp: pd.Series, cal:CMEEquityExchangeCalendar) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_weekly_close_time, cal=new_york_cal
    )

@tag(cache="parquet")
def new_york_last_open_time_0(timestamp: pd.Series) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_last_open_time, cal=new_york_cal, k=0
    )

@tag(cache="parquet")
def month(time: pd.Series) -> pd.Series:
    return time.dt.month  # categories have be strings

@tag(cache="parquet")
def year(time: pd.Series) -> pd.Series:
    return time.dt.year  # categories have be strings

@tag(cache="parquet")
def hour_of_day(time: pd.Series) -> pd.Series:
    return time.apply(lambda x: x.hour)

@tag(cache="parquet")
def day_of_week(time: pd.Series) -> pd.Series:
    return time.apply(lambda x: x.dayofweek)

@tag(cache="parquet")
def day_of_month(time: pd.Series) -> pd.Series:
    return time.apply(lambda x: x.day)

@tag(cache="parquet")
def new_york_last_close_time_0(timestamp: pd.Series) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_last_open_time, cal=new_york_cal, k=0
    )

@tag(cache="parquet")
def new_york_last_open_time_1(timestamp: pd.Series) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_last_open_time, cal=new_york_cal, k=1
    )

@tag(cache="parquet")
def new_york_last_close_time_1(timestamp: pd.Series) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_last_open_time, cal=new_york_cal, k=1
    )

@tag(cache="parquet")
def london_last_open_time_0(timestamp: pd.Series) -> pd.Series:
    lse_cal = mcal.get_calendar("LSE")
    return timestamp.apply(
        market_time.compute_last_open_time, cal=lse_cal, k=0
    )
@tag(cache="parquet")
def london_last_close_time_0(timestamp: pd.Series) -> pd.Series:
    lse_cal = mcal.get_calendar("LSE")
    return timestamp.apply(
        market_time.compute_last_open_time, cal=lse_cal, k=0
    )
@tag(cache="parquet")
def london_last_open_time_1(timestamp: pd.Series) -> pd.Series:
    lse_cal = mcal.get_calendar("LSE")
    return timestamp.apply(
        market_time.compute_last_open_time, cal=lse_cal, k=1
    )

@tag(cache="parquet")
def london_last_close_time(london_last_close_time_0: pd.Series) -> pd.Series:
    return london_last_close_time_0

@tag(cache="parquet")
def london_last_open_time(london_last_open_time_0: pd.Series) -> pd.Series:
    return london_last_open_time_0

@tag(cache="parquet")
def new_york_last_close_time(new_york_last_close_time_0: pd.Series) -> pd.Series:
    return new_york_last_close_time_0

@tag(cache="parquet")
def new_york_last_open_time(new_york_last_open_time_0: pd.Series) -> pd.Series:
    return new_york_last_open_time_0

@tag(cache="parquet")
def london_last_close_time_1(timestamp: pd.Series) -> pd.Series:
    lse_cal = mcal.get_calendar("LSE")
    return timestamp.apply(
        market_time.compute_last_open_time, cal=lse_cal, k=1
    )
@tag(cache="parquet")
def last_weekly_close_time(timestamp: pd.Series, cal:CMEEquityExchangeCalendar) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_last_weekly_close_time, cal=new_york_cal
    )

@tag(cache="parquet")
def monthly_close_time(timestamp: pd.Series, cal:CMEEquityExchangeCalendar) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_monthly_close_time, cal=new_york_cal
    )

@tag(cache="parquet")
def last_monthly_close_time(timestamp: pd.Series, cal:CMEEquityExchangeCalendar) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_last_monthly_close_time, cal=new_york_cal
    )

@tag(cache="parquet")
def option_expiration_time(timestamp: pd.Series, cal:CMEEquityExchangeCalendar) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_option_expiration_time, cal=new_york_cal
    )

@tag(cache="parquet")
def last_option_expiration_time(timestamp: pd.Series, cal:CMEEquityExchangeCalendar) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_last_option_expiration_time, cal=new_york_cal
    )

@tag(cache="parquet")
def last_macro_event_time_imp1(timestamp: pd.Series, macro_data_builder:MacroDataBuilder) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_last_macro_event_time, cal=new_york_cal,
        mdb=macro_data_builder, imp=1)

@tag(cache="parquet")
def next_macro_event_time_imp1(timestamp: pd.Series, macro_data_builder:MacroDataBuilder) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_next_macro_event_time, cal=new_york_cal,
        mdb=macro_data_builder, imp=1)

@tag(cache="parquet")
def last_macro_event_time_imp2(timestamp: pd.Series, macro_data_builder:MacroDataBuilder) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_last_macro_event_time, cal=new_york_cal,
        mdb=macro_data_builder, imp=2)

@tag(cache="parquet")
def next_macro_event_time_imp2(timestamp: pd.Series, macro_data_builder:MacroDataBuilder) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_next_macro_event_time, cal=new_york_cal,
        mdb=macro_data_builder, imp=2)

@tag(cache="parquet")
def last_macro_event_time_imp3(timestamp: pd.Series, macro_data_builder:MacroDataBuilder) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_last_macro_event_time, cal=new_york_cal,
        mdb=macro_data_builder, imp=3)

@tag(cache="parquet")
def next_macro_event_time_imp3(timestamp: pd.Series, macro_data_builder:MacroDataBuilder) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_next_macro_event_time, cal=new_york_cal,
        mdb=macro_data_builder, imp=3)

@parameterize(
    time_high_1d_ff_shift_1d={"time_col": source("time_high_1d_ff"), "lookback_days":value(1)},
    time_high_5d_ff_shift_5d={"time_col": source("time_high_5d_ff"), "lookback_days":value(5)},
    time_high_11d_ff_shift_11d={"time_col": source("time_high_11d_ff"), "lookback_days":value(11)},
    time_high_21d_ff_shift_21d={"time_col": source("time_high_21d_ff"), "lookback_days":value(21)},
    time_low_1d_ff_shift_1d={"time_col": source("time_low_1d_ff"), "lookback_days":value(1)},
    time_low_5d_ff_shift_5d={"time_col": source("time_low_5d_ff"), "lookback_days":value(5)},
    time_low_11d_ff_shift_11d={"time_col": source("time_low_11d_ff"), "lookback_days":value(11)},
    time_low_21d_ff_shift_21d={"time_col": source("time_low_21d_ff"), "lookback_days":value(21)},
)
def time_shift(time_col:pd.Series, lookback_days:int, interval_per_day: int) -> pd.Series:
    return time_col.shift(-lookback_days*interval_per_day)

@parameterize(
    time_high_1_ff={"close_col": value("close_high_1_ff")},
    time_high_5_ff={"close_col": value("close_high_5_ff")},
    time_high_11_ff={"close_col": value("close_high_11_ff")},
    time_high_21_ff={"close_col": value("close_high_21_ff")},
    time_high_51_ff={"close_col": value("close_high_51_ff")},
    time_high_101_ff={"close_col": value("close_high_101_ff")},
    time_high_201_ff={"close_col": value("close_high_201_ff")},
    time_low_1_ff={"close_col": value("close_low_1_ff")},
    time_low_5_ff={"close_col": value("close_low_5_ff")},
    time_low_11_ff={"close_col": value("close_low_11_ff")},
    time_low_21_ff={"close_col": value("close_low_21_ff")},
    time_low_51_ff={"close_col": value("close_low_51_ff")},
    time_low_101_ff={"close_col": value("close_low_101_ff")},
    time_low_201_ff={"close_col": value("close_low_201_ff")},
    time_high_1d_ff={"close_col": value("close_high_1d_ff")},
    time_high_5d_ff={"close_col": value("close_high_5d_ff")},
    time_high_11d_ff={"close_col": value("close_high_11d_ff")},
    time_high_21d_ff={"close_col": value("close_high_21d_ff")},
    time_high_51d_ff={"close_col": value("close_high_51d_ff")},
    time_high_101d_ff={"close_col": value("close_high_101d_ff")},
    time_high_201d_ff={"close_col": value("close_high_201d_ff")},
    time_low_1d_ff={"close_col": value("close_low_1d_ff")},
    time_low_5d_ff={"close_col": value("close_low_5d_ff")},
    time_low_11d_ff={"close_col": value("close_low_11d_ff")},
    time_low_21d_ff={"close_col": value("close_low_21d_ff")},
    time_low_51d_ff={"close_col": value("close_low_51d_ff")},
    time_low_101d_ff={"close_col": value("close_low_101d_ff")},
    time_low_201d_ff={"close_col": value("close_low_201d_ff")},
)
def time_at(sorted_data:pd.DataFrame, close_col:str) -> pd.Series:
    time_series = sorted_data.apply(lambda x: get_close_time(x, close_col=close_col), axis=1)
    time_series = time_series.ffill()
    return time_series

def time_features(group_features:pd.DataFrame, cal:CMEEquityExchangeCalendar,
                  macro_data_builder:MacroDataBuilder, config:DictConfig,
                  weekly_close_time: pd.Series, last_weekly_close_time: pd.Series,
                  monthly_close_time: pd.Series, last_monthly_close_time: pd.Series,
                  option_expiration_time: pd.Series, last_option_expiration_time: pd.Series,
                  last_macro_event_time_imp1: pd.Series, next_macro_event_time_imp1: pd.Series,
                  last_macro_event_time_imp2: pd.Series, next_macro_event_time_imp2: pd.Series,
                  last_macro_event_time_imp3: pd.Series, next_macro_event_time_imp3: pd.Series,
                  new_york_last_open_time_0: pd.Series, new_york_last_close_time_0: pd.Series,
                  new_york_last_open_time_1: pd.Series, new_york_last_close_time_1: pd.Series,
                  london_last_open_time_0: pd.Series, london_last_close_time_0: pd.Series,
                  london_last_open_time_1: pd.Series, london_last_close_time_1: pd.Series,
                  new_york_open_time: pd.Series, new_york_close_time: pd.Series,
                  london_open_time: pd.Series, london_close_time: pd.Series) -> pd.DataFrame:
    add_daily_rolling_features=config.model.features.add_daily_rolling_features
    new_york_cal = mcal.get_calendar("NYSE")

    raw_data = group_features
    raw_data["week_of_year"] = raw_data["time"].apply(lambda x: x.isocalendar()[1])
    raw_data["month_of_year"] = raw_data["time"].apply(lambda x: x.month)

    raw_data["weekly_close_time"] = weekly_close_time
    raw_data["last_weekly_close_time"] = last_weekly_close_time
    raw_data["monthly_close_time"] = monthly_close_time
    raw_data["last_monthly_close_time"] = last_monthly_close_time
    raw_data["option_expiration_time"] = option_expiration_time
    raw_data["last_option_expiration_time"] = last_option_expiration_time
    raw_data["last_macro_event_time_imp1"] = last_macro_event_time_imp1
    raw_data["next_macro_event_time_imp1"] = next_macro_event_time_imp1
    raw_data["last_macro_event_time_imp2"] = last_macro_event_time_imp2
    raw_data["next_macro_event_time_imp2"] = next_macro_event_time_imp2
    raw_data["last_macro_event_time_imp3"] = last_macro_event_time_imp3
    raw_data["next_macro_event_time_imp3"] = next_macro_event_time_imp3
    raw_data["new_york_last_open_time_0"] = new_york_last_open_time_0
    raw_data["new_york_last_open_time_1"] = new_york_last_open_time_1
    raw_data["new_york_last_close_time_0"] = new_york_last_close_time_0
    raw_data["new_york_last_close_time_1"] = new_york_last_close_time_1
    raw_data["london_last_open_time_0"] = london_last_open_time_0
    raw_data["london_last_open_time_1"] = london_last_open_time_1
    raw_data["london_last_close_time_0"] = london_last_close_time_0
    raw_data["london_last_close_time_1"] = london_last_close_time_1
    raw_data["new_york_open_time"] = new_york_open_time
    raw_data["new_york_close_time"] = new_york_close_time
    raw_data["london_open_time"] = london_open_time
    raw_data["london_close_time"] = london_close_time
    return raw_data
