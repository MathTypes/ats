import datetime
import logging
import math
import os
import traceback
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

def timestamp(clean_sorted_data: pd.DataFrame) -> pd.Series:
    return clean_sorted_data["idx_timestamp"]
    #logging.error(f"clean_sorted_data:{clean_sorted_data.iloc[:3]}")
    #series = clean_sorted_data[["timestamp"]]
    #series = clean_sorted_data["timestamp"]
    #series = clean_sorted_data.index.get_level_values(0).to_series()
    #logging.error(f"series:{series}")
    #return series

def time(timestamp: pd.Series) -> pd.Series:
    series = timestamp.apply(lambda x:datetime.datetime.fromtimestamp(x, tz=datetime.timezone.utc))
    #logging.error(f"time:{clean_sorted_data.iloc[5:10]}")
    #series =  clean_sorted_data.index.get_level_values(0).to_series()
    logging.error(f"time series:{series}")
    return series

def week_of_year(time: pd.Series) -> pd.Series:
    return time.apply(lambda x: x.isocalendar().week)

def month_of_year(time: pd.Series) -> pd.Series:
    return time.apply(lambda x: x.month)

@tag(cache="parquet")
def new_york_open_time(timestamp: pd.Series) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_next_open_time, cal=new_york_cal
    ).rename("new_york_open_time")

@tag(cache="parquet")
def new_york_close_time(timestamp: pd.Series) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_next_close_time, cal=new_york_cal
    ).rename("new_york_close_time")

@tag(cache="parquet")
def is_new_york_close_time(timestamp: pd.Series, new_york_close_time: pd.Series) -> pd.Series:
    return timestamp == new_york_close_time

@tag(cache="parquet")
def is_weekly_close_time(timestamp: pd.Series, weekly_close_time: pd.Series) -> pd.Series:
    return timestamp == weekly_close_time

@tag(cache="parquet")
def is_monthyl_close_time(timestamp: pd.Series, last_monthly_close_time_0: pd.Series) -> pd.Series:
    return timestamp == last_monthly_close_time_0

@tag(cache="parquet")
def london_open_time(timestamp: pd.Series) -> pd.Series:
    lse_cal = mcal.get_calendar("LSE")
    return timestamp.apply(
        market_time.compute_next_open_time, cal=lse_cal
    ).rename("london_open_time")

@tag(cache="parquet")
def london_close_time(timestamp: pd.Series) -> pd.Series:
    lse_cal = mcal.get_calendar("LSE")
    return timestamp.apply(
        market_time.compute_next_close_time, cal=lse_cal
    ).rename("london_close_time")

@tag(cache="parquet")
def weekly_close_time(timestamp: pd.Series, cal:CMEEquityExchangeCalendar) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_weekly_close_time, cal=new_york_cal
    ).rename("weekly_close_time")


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

@parameterize(
    new_york_last_open_time_0={"k":value(0)},
    new_york_last_open_time_1={"k":value(1)},
    new_york_last_open_time_2={"k":value(2)},
    new_york_last_open_time_3={"k":value(3)},
    new_york_last_open_time_4={"k":value(4)},
    new_york_last_open_time_5={"k":value(5)},
    new_york_last_open_time_6={"k":value(6)},
    new_york_last_open_time_7={"k":value(7)},
    new_york_last_open_time_8={"k":value(8)},
    new_york_last_open_time_9={"k":value(9)},
    new_york_last_open_time_10={"k":value(10)},
    new_york_last_open_time_11={"k":value(11)},
    new_york_last_open_time_12={"k":value(12)},
    new_york_last_open_time_13={"k":value(13)},
    new_york_last_open_time_14={"k":value(14)},
    new_york_last_open_time_15={"k":value(15)},
    new_york_last_open_time_16={"k":value(16)},
    new_york_last_open_time_17={"k":value(17)},
    new_york_last_open_time_18={"k":value(18)},
    new_york_last_open_time_19={"k":value(19)},
)
def new_york_last_open_time_tmpl(timestamp: pd.Series, k:int) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_last_open_time, cal=new_york_cal, k=k
    ).rename("new_york_last_open_time")


@parameterize(
    new_york_last_close_time_0={"k":value(0)},
    new_york_last_close_time_1={"k":value(1)},
    new_york_last_close_time_2={"k":value(2)},
    new_york_last_close_time_3={"k":value(3)},
    new_york_last_close_time_4={"k":value(4)},
    new_york_last_close_time_5={"k":value(5)},
    new_york_last_close_time_6={"k":value(6)},
    new_york_last_close_time_7={"k":value(7)},
    new_york_last_close_time_8={"k":value(8)},
    new_york_last_close_time_9={"k":value(9)},
    new_york_last_close_time_10={"k":value(10)},
    new_york_last_close_time_11={"k":value(11)},
    new_york_last_close_time_12={"k":value(12)},
    new_york_last_close_time_13={"k":value(13)},
    new_york_last_close_time_14={"k":value(14)},
    new_york_last_close_time_15={"k":value(15)},
    new_york_last_close_time_16={"k":value(16)},
    new_york_last_close_time_17={"k":value(17)},
    new_york_last_close_time_18={"k":value(18)},
    new_york_last_close_time_19={"k":value(19)},
)
def new_york_last_close_time_tmpl(timestamp: pd.Series, k:int) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_last_close_time, cal=new_york_cal, k=k
    ).rename("new_york_last_close_time")

def last_daily_close_time(timestamp: pd.Series) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_last_close_time, cal=new_york_cal, k=0
    ).rename("last_daily_close_time")

def next_daily_close_time(timestamp: pd.Series) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_next_close_time, cal=new_york_cal, k=0
    ).rename("next_daily_close_time")

def next_weekly_close_time(timestamp: pd.Series) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_next_weekly_close_time, cal=new_york_cal
    ).rename("next_weekly_close_time")

def next_monthly_close_time(timestamp: pd.Series) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_next_monthly_close_time, cal=new_york_cal
    ).rename("next_monthly_close_time")

@parameterize(
    last_weekly_close_time_0={"k":value(0)},
    last_weekly_close_time_1={"k":value(1)},
    last_weekly_close_time_2={"k":value(2)},
    last_weekly_close_time_3={"k":value(3)},
    last_weekly_close_time_4={"k":value(4)},
    last_weekly_close_time_5={"k":value(5)},
    last_weekly_close_time_6={"k":value(6)},
    last_weekly_close_time_7={"k":value(7)},
    last_weekly_close_time_8={"k":value(8)},
    last_weekly_close_time_9={"k":value(9)},
    last_weekly_close_time_10={"k":value(10)},
    last_weekly_close_time_11={"k":value(11)},
    last_weekly_close_time_12={"k":value(12)},
    last_weekly_close_time_13={"k":value(13)},
    last_weekly_close_time_14={"k":value(14)},
    last_weekly_close_time_15={"k":value(15)},
    last_weekly_close_time_16={"k":value(16)},
    last_weekly_close_time_17={"k":value(17)},
    last_weekly_close_time_18={"k":value(18)},
    last_weekly_close_time_19={"k":value(19)},
)
def last_weekly_close_time_tmpl(timestamp: pd.Series, k:int) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_last_weekly_close_time, cal=new_york_cal, k=1
    ).rename("last_weekly_close_time")

def last_weekly_close_time(timestamp: pd.Series) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_last_weekly_close_time, cal=new_york_cal, k=0
    ).rename("last_weekly_close_time")

@parameterize(
    last_monthly_close_time_0={"k":value(0)},
    last_monthly_close_time_1={"k":value(1)},
    last_monthly_close_time_2={"k":value(2)},
    last_monthly_close_time_3={"k":value(3)},
    last_monthly_close_time_4={"k":value(4)},
    last_monthly_close_time_5={"k":value(5)},
    last_monthly_close_time_6={"k":value(6)},
    last_monthly_close_time_7={"k":value(7)},
    last_monthly_close_time_8={"k":value(8)},
    last_monthly_close_time_9={"k":value(9)},
    last_monthly_close_time_10={"k":value(10)},
    last_monthly_close_time_11={"k":value(11)},
    last_monthly_close_time_12={"k":value(12)},
    last_monthly_close_time_13={"k":value(13)},
    last_monthly_close_time_14={"k":value(14)},
    last_monthly_close_time_15={"k":value(15)},
    last_monthly_close_time_16={"k":value(16)},
    last_monthly_close_time_17={"k":value(17)},
    last_monthly_close_time_18={"k":value(18)},
    last_monthly_close_time_19={"k":value(19)},
)
def last_monthly_close_time_tmpl(timestamp: pd.Series, k:int) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_last_monthly_close_time, cal=new_york_cal, k=k
    ).rename("last_monthly_close_time")

def last_monthly_close_time(timestamp: pd.Series) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_last_monthly_close_time, cal=new_york_cal, k=0
    ).rename("last_monthly_close_time")

@parameterize(
    london_last_open_time_0={"k":value(0)},
    london_last_open_time_1={"k":value(1)},
    london_last_open_time_2={"k":value(2)},
    london_last_open_time_3={"k":value(3)},
    london_last_open_time_4={"k":value(4)},
    london_last_open_time_5={"k":value(5)},
    london_last_open_time_6={"k":value(6)},
    london_last_open_time_7={"k":value(7)},
    london_last_open_time_8={"k":value(8)},
    london_last_open_time_9={"k":value(9)},
    london_last_open_time_10={"k":value(10)},
    london_last_open_time_11={"k":value(11)},
    london_last_open_time_12={"k":value(12)},
    london_last_open_time_13={"k":value(13)},
    london_last_open_time_14={"k":value(14)},
    london_last_open_time_15={"k":value(15)},
    london_last_open_time_16={"k":value(16)},
    london_last_open_time_17={"k":value(17)},
    london_last_open_time_18={"k":value(18)},
    london_last_open_time_19={"k":value(19)},
)
def london_last_open_time_tmpl(timestamp: pd.Series, k:int) -> pd.Series:
    lse_cal = mcal.get_calendar("LSE")
    return timestamp.apply(
        market_time.compute_last_open_time, cal=lse_cal, k=k
    ).rename("london_last_open_time")

@parameterize(
    london_last_close_time_0={"k":value(0)},
    london_last_close_time_1={"k":value(1)},
    london_last_close_time_2={"k":value(2)},
    london_last_close_time_3={"k":value(3)},
    london_last_close_time_4={"k":value(4)},
    london_last_close_time_5={"k":value(5)},
    london_last_close_time_6={"k":value(6)},
    london_last_close_time_7={"k":value(7)},
    london_last_close_time_8={"k":value(8)},
    london_last_close_time_9={"k":value(9)},
    london_last_close_time_10={"k":value(10)},
    london_last_close_time_11={"k":value(11)},
    london_last_close_time_12={"k":value(12)},
    london_last_close_time_13={"k":value(13)},
    london_last_close_time_14={"k":value(14)},
    london_last_close_time_15={"k":value(15)},
    london_last_close_time_16={"k":value(16)},
    london_last_close_time_17={"k":value(17)},
    london_last_close_time_18={"k":value(18)},
    london_last_close_time_19={"k":value(19)},
)
def london_last_close_time_tmpl(timestamp: pd.Series, k:int) -> pd.Series:
    lse_cal = mcal.get_calendar("LSE")
    return timestamp.apply(
        market_time.compute_last_open_time, cal=lse_cal, k=k
    ).rename("london_last_close_time")

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
def last_weekly_close_time(timestamp: pd.Series, cal:CMEEquityExchangeCalendar) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_last_weekly_close_time, cal=new_york_cal
    ).rename("last_weekly_close_time")

@tag(cache="parquet")
def monthly_close_time(timestamp: pd.Series, cal:CMEEquityExchangeCalendar) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_monthly_close_time, cal=new_york_cal
    ).rename("monthly_close_time")

@tag(cache="parquet")
def last_monthly_close_time(timestamp: pd.Series, cal:CMEEquityExchangeCalendar) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_last_monthly_close_time, cal=new_york_cal
    ).rename("last_monthly_close_time")

@tag(cache="parquet")
def option_expiration_time(timestamp: pd.Series, ticker: pd.Series, cal:CMEEquityExchangeCalendar) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    series = timestamp.apply(
        market_time.compute_option_expiration_time, cal=new_york_cal
    ).rename("option_expiration_time")
    #logging.error(f"ticker:{ticker}")
    df = series.reset_index()
    #logging.error(f"df_before_reset:{df}")
    df["ticker"] = ticker.values
    df = df.set_index(["timestamp", "ticker"])
    #logging.error(f"option_expiration_time_df:{df}")
    series = df["option_expiration_time"]
    #logging.error(f"option_expiration_time_series:{series}")
    return series

@tag(cache="parquet")
def last_option_expiration_time(timestamp: pd.Series, cal:CMEEquityExchangeCalendar) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_last_option_expiration_time, cal=new_york_cal
    ).rename("last_option_expiration_time")

@tag(cache="parquet")
def last_macro_event_time_imp1(timestamp: pd.Series, macro_data_builder:MacroDataBuilder) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_last_macro_event_time, cal=new_york_cal,
        mdb=macro_data_builder, imp=1).rename("last_macro_event_time_imp1")

@tag(cache="parquet")
def next_macro_event_time_imp1(timestamp: pd.Series, macro_data_builder:MacroDataBuilder) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_next_macro_event_time, cal=new_york_cal,
        mdb=macro_data_builder, imp=1).rename("next_macro_event_time_imp1")

@tag(cache="parquet")
def last_macro_event_time_imp2(timestamp: pd.Series, macro_data_builder:MacroDataBuilder) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_last_macro_event_time, cal=new_york_cal,
        mdb=macro_data_builder, imp=2).rename("last_macro_event_time_imp2")

@tag(cache="parquet")
def next_macro_event_time_imp2(timestamp: pd.Series, macro_data_builder:MacroDataBuilder) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_next_macro_event_time, cal=new_york_cal,
        mdb=macro_data_builder, imp=2).rename("next_macro_event_time_imp2")

@tag(cache="parquet")
def last_macro_event_time_imp3(timestamp: pd.Series, macro_data_builder:MacroDataBuilder) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_last_macro_event_time, cal=new_york_cal,
        mdb=macro_data_builder, imp=3).rename("last_macro_event_time_imp3")

@tag(cache="parquet")
def next_macro_event_time_imp3(timestamp: pd.Series, macro_data_builder:MacroDataBuilder) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_next_macro_event_time, cal=new_york_cal,
        mdb=macro_data_builder, imp=3).rename("next_macro_event_time_imp3")

@parameterize(
    time_to_new_york_open={"diff_time": source("new_york_open_time"),"diff_col":value("new_york_open_time")},
    time_to_new_york_last_open={"diff_time": source("new_york_last_open_time"),"diff_col":value("new_york_last_open_time")},
    time_to_new_york_last_close={"diff_time": source("new_york_last_close_time"),"diff_col":value("new_york_last_close_time")},
    time_to_new_york_close={"diff_time": source("new_york_close_time"),"diff_col":value("new_york_close_time")},
    time_to_london_open={"diff_time": source("london_open_time"),"diff_col":value("london_open_time")},
    time_to_london_last_open={"diff_time": source("london_last_open_time"),"diff_col":value("london_last_open_time")},
    time_to_london_last_close={"diff_time": source("london_last_close_time"),"diff_col":value("london_last_close_time")},
    time_to_london_close={"diff_time": source("london_close_time"),"diff_col":value("london_close_time")},
    time_to_weekly_close={"diff_time": source("weekly_close_time"),"diff_col":value("weekly_close_time")},
    time_to_monthly_close={"diff_time": source("monthly_close_time"),"diff_col":value("monthly_close_time")},
    time_to_option_expiration={"diff_time": source("option_expiration_time"),"diff_col":value("option_expiration_time")},
    time_to_last_macro_event_imp1={"diff_time": source("last_macro_event_time_imp1"),"diff_col":value("last_macro_event_time_imp1")},
    time_to_last_macro_event_imp2={"diff_time": source("last_macro_event_time_imp2"),"diff_col":value("last_macro_event_time_imp2")},
    time_to_last_macro_event_imp3={"diff_time": source("last_macro_event_time_imp3"),"diff_col":value("last_macro_event_time_imp3")},
    time_to_next_macro_event_imp1={"diff_time": source("next_macro_event_time_imp1"),"diff_col":value("next_macro_event_time_imp1")},
    time_to_next_macro_event_imp2={"diff_time": source("next_macro_event_time_imp2"),"diff_col":value("next_macro_event_time_imp2")},
    time_to_next_macro_event_imp3={"diff_time": source("next_macro_event_time_imp3"),"diff_col":value("next_macro_event_time_imp3")},
    time_to_high_5_ff={"diff_time": source("time_high_5_ff"),"diff_col":value("time_high_5_ff")},
    time_to_high_11_ff={"diff_time": source("time_high_11_ff"),"diff_col":value("time_high_11_ff")},
    time_to_high_21_ff={"diff_time": source("time_high_21_ff"),"diff_col":value("time_high_21_ff")},
    time_to_high_51_ff={"diff_time": source("time_high_51_ff"),"diff_col":value("time_high_51_ff")},
    time_to_high_101_ff={"diff_time": source("time_high_101_ff"),"diff_col":value("time_high_101_ff")},
    time_to_high_201_ff={"diff_time": source("time_high_201_ff"),"diff_col":value("time_high_201_ff")},
    time_to_low_5_ff={"diff_time": source("time_low_5_ff"),"diff_col":value("time_low_5_ff")},
    time_to_low_11_ff={"diff_time": source("time_low_11_ff"),"diff_col":value("time_low_11_ff")},
    time_to_low_21_ff={"diff_time": source("time_low_21_ff"),"diff_col":value("time_low_21_ff")},
    time_to_low_51_ff={"diff_time": source("time_low_51_ff"),"diff_col":value("time_low_51_ff")},
    time_to_low_101_ff={"diff_time": source("time_low_101_ff"),"diff_col":value("time_low_101_ff")},
    time_to_low_201_ff={"diff_time": source("time_low_201_ff"),"diff_col":value("time_low_201_ff")},
    time_to_high_1d_ff_shift_1d={"diff_time": source("time_high_1d_ff_shift_1d"),"diff_col":value("time_high_1d_ff_shift_1d")},
    time_to_low_1d_ff_shift_1d={"diff_time": source("time_low_1d_ff_shift_1d"),"diff_col":value("time_low_1d_ff_shift_1d")},
    time_to_high_5d_ff_shift_5d={"diff_time": source("time_high_5d_ff_shift_5d"),"diff_col":value("time_high_5d_ff_shift_5d")},
    time_to_low_5d_ff_shift_5d={"diff_time": source("time_low_5d_ff_shift_5d"),"diff_col":value("time_low_5d_ff_shift_5d")},
    time_to_high_11d_ff_shift_11d={"diff_time": source("time_high_11d_ff_shift_11d"),"diff_col":value("time_high_11d_ff_shift_11d")},
    time_to_low_11d_ff_shift_11d={"diff_time": source("time_low_11d_ff_shift_11d"),"diff_col":value("time_low_11d_ff_shift_11d")},
    time_to_high_21d_ff_shift_21d={"diff_time": source("time_high_21d_ff_shift_21d"),"diff_col":value("time_high_21d_ff_shift_21d")},
    time_to_low_21d_ff_shift_21d={"diff_time": source("time_low_21d_ff_shift_21d"),"diff_col":value("time_low_21d_ff_shift_21d")},
    time_to_high_1d_ff={"diff_time": source("time_high_1d_ff"),"diff_col":value("time_high_1d_ff")},
    time_to_low_1d_ff={"diff_time": source("time_low_1d_ff"),"diff_col":value("time_low_1d_ff")},
    time_to_high_5d_ff={"diff_time": source("time_high_5d_ff"),"diff_col":value("time_high_5d_ff")},
    time_to_low_5d_ff={"diff_time": source("time_low_5d_ff"),"diff_col":value("time_low_5d_ff")},
    time_to_high_11d_ff={"diff_time": source("time_high_11d_ff"),"diff_col":value("time_high_11d_ff")},
    time_to_low_11d_ff={"diff_time": source("time_low_11d_ff"),"diff_col":value("time_low_11d_ff")},
    time_to_high_21d_ff={"diff_time": source("time_high_21d_ff"),"diff_col":value("time_high_21d_ff")},
    time_to_low_21d_ff={"diff_time": source("time_low_21d_ff"),"diff_col":value("time_low_21d_ff")},
    time_to_high_51d_ff={"diff_time": source("time_high_51d_ff"),"diff_col":value("time_high_51d_ff")},
    time_to_low_51d_ff={"diff_time": source("time_low_51d_ff"),"diff_col":value("time_low_51d_ff")},
    time_to_high_101d_ff={"diff_time": source("time_high_101d_ff"),"diff_col":value("time_high_101d_ff")},
    time_to_low_101d_ff={"diff_time": source("time_low_101d_ff"),"diff_col":value("time_low_101d_ff")},
    time_to_high_201d_ff={"diff_time": source("time_high_201d_ff"),"diff_col":value("time_high_201d_ff")},
    time_to_low_201d_ff={"diff_time": source("time_low_201d_ff"),"diff_col":value("time_low_201d_ff")},
)
def time_to(timestamp:pd.Series, diff_time:pd.Series, ticker:pd.Series, diff_col:str) -> pd.Series:
    #traceback.print_stack()
    logging.error(f"time_to_diff_col:{diff_col}, timestamp:{timestamp}, diff_col:{diff_col}")
    logging.error(f"time_to_diff_col:{diff_col}, ticker:{ticker}")
    #timestamp = timestamp.reset_index()
    diff_time = diff_time.reset_index()
    logging.error(f"time_to_diff_col:{diff_col}, diff_time:{diff_time}, diff_col:{diff_col}")
    if "ticker" in diff_time.columns:
        df = pd.concat([diff_time], axis=1)
    else:
        df = pd.concat([diff_time, ticker.index.to_series()], axis=1)
    logging.error(f"time_to_df before setting column:{df.iloc[-10:]}")
    
    df.columns = ["timestamp", "ticker", diff_col]
    logging.error(f"time_to_df:{df.iloc[-10:]}")
    df["time_to"]=df[diff_col] - df["timestamp"]
    df = df.set_index(["timestamp","ticker"])
    #diff_series = diff_series.set_index(["ticker","time"])
    #logging.error(f"time_to_df after diff:{df.iloc[-10:]}")
    diff_series = df["time_to"]
    #logging.error(f"diff_series:{diff_series.iloc[-10:]}")
    return diff_series

def time_features(clean_sorted_data:pd.DataFrame, cal:CMEEquityExchangeCalendar,
                  macro_data_builder:MacroDataBuilder, config:DictConfig,
                  weekly_close_time: pd.Series, last_weekly_close_time: pd.Series,
                  monthly_close_time: pd.Series, last_monthly_close_time: pd.Series,
                  option_expiration_time: pd.Series, last_option_expiration_time: pd.Series,
                  last_macro_event_time_imp1: pd.Series, next_macro_event_time_imp1: pd.Series,
                  last_macro_event_time_imp2: pd.Series, next_macro_event_time_imp2: pd.Series,
                  last_macro_event_time_imp3: pd.Series, next_macro_event_time_imp3: pd.Series,
                  new_york_last_open_time_0: pd.Series,
                  new_york_last_open_time_1: pd.Series,
                  new_york_last_open_time_2: pd.Series,
                  new_york_last_open_time_3: pd.Series,
                  new_york_last_open_time_4: pd.Series,
                  new_york_last_open_time_5: pd.Series,
                  new_york_last_open_time_6: pd.Series,
                  new_york_last_open_time_7: pd.Series,
                  new_york_last_open_time_8: pd.Series,
                  new_york_last_open_time_9: pd.Series,
                  new_york_last_open_time_10: pd.Series,
                  new_york_last_open_time_11: pd.Series,
                  new_york_last_open_time_12: pd.Series,
                  new_york_last_open_time_13: pd.Series,
                  new_york_last_open_time_14: pd.Series,
                  new_york_last_open_time_15: pd.Series,
                  new_york_last_open_time_16: pd.Series,
                  new_york_last_open_time_17: pd.Series,
                  new_york_last_open_time_18: pd.Series,
                  new_york_last_open_time_19: pd.Series,
                  new_york_last_close_time_0: pd.Series,
                  new_york_last_close_time_1: pd.Series,
                  new_york_last_close_time_2: pd.Series,
                  new_york_last_close_time_3: pd.Series,
                  new_york_last_close_time_4: pd.Series,
                  new_york_last_close_time_5: pd.Series,
                  new_york_last_close_time_6: pd.Series,
                  new_york_last_close_time_7: pd.Series,
                  new_york_last_close_time_8: pd.Series,
                  new_york_last_close_time_9: pd.Series,
                  new_york_last_close_time_10: pd.Series,
                  new_york_last_close_time_11: pd.Series,
                  new_york_last_close_time_12: pd.Series,
                  new_york_last_close_time_13: pd.Series,
                  new_york_last_close_time_14: pd.Series,
                  new_york_last_close_time_15: pd.Series,
                  new_york_last_close_time_16: pd.Series,
                  new_york_last_close_time_17: pd.Series,
                  new_york_last_close_time_18: pd.Series,
                  new_york_last_close_time_19: pd.Series,
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
                  london_last_open_time_0: pd.Series,
                  london_last_open_time_1: pd.Series,
                  london_last_open_time_2: pd.Series,
                  london_last_open_time_3: pd.Series,
                  london_last_open_time_4: pd.Series,
                  london_last_open_time_5: pd.Series,
                  london_last_open_time_6: pd.Series,
                  london_last_open_time_7: pd.Series,
                  london_last_open_time_8: pd.Series,
                  london_last_open_time_9: pd.Series,
                  london_last_open_time_10: pd.Series,
                  london_last_open_time_11: pd.Series,
                  london_last_open_time_12: pd.Series,
                  london_last_open_time_13: pd.Series,
                  london_last_open_time_14: pd.Series,
                  london_last_open_time_15: pd.Series,
                  london_last_open_time_16: pd.Series,
                  london_last_open_time_17: pd.Series,
                  london_last_open_time_18: pd.Series,
                  london_last_open_time_19: pd.Series,
                  london_last_close_time_0: pd.Series,
                  london_last_close_time_1: pd.Series,
                  london_last_close_time_2: pd.Series,
                  london_last_close_time_3: pd.Series,
                  london_last_close_time_4: pd.Series,
                  london_last_close_time_5: pd.Series,
                  london_last_close_time_6: pd.Series,
                  london_last_close_time_7: pd.Series,
                  london_last_close_time_8: pd.Series,
                  london_last_close_time_9: pd.Series,
                  london_last_close_time_10: pd.Series,
                  london_last_close_time_11: pd.Series,
                  london_last_close_time_12: pd.Series,
                  london_last_close_time_13: pd.Series,
                  london_last_close_time_14: pd.Series,
                  london_last_close_time_15: pd.Series,
                  london_last_close_time_16: pd.Series,
                  london_last_close_time_17: pd.Series,
                  london_last_close_time_18: pd.Series,
                  london_last_close_time_19: pd.Series,
                  new_york_open_time: pd.Series, new_york_close_time: pd.Series,
                  london_open_time: pd.Series, london_close_time: pd.Series) -> pd.DataFrame:
    add_daily_rolling_features=config.model.features.add_daily_rolling_features
    new_york_cal = mcal.get_calendar("NYSE")

    raw_data = clean_sorted_data.copy()

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
    raw_data["new_york_last_open_time"] = new_york_last_open_time_0
    raw_data["new_york_last_open_time_0"] = new_york_last_open_time_0
    raw_data["new_york_last_open_time_1"] = new_york_last_open_time_1
    raw_data["new_york_last_open_time_2"] = new_york_last_open_time_2
    raw_data["new_york_last_open_time_3"] = new_york_last_open_time_3
    raw_data["new_york_last_open_time_4"] = new_york_last_open_time_4
    raw_data["new_york_last_open_time_5"] = new_york_last_open_time_5
    raw_data["new_york_last_open_time_6"] = new_york_last_open_time_6
    raw_data["new_york_last_open_time_7"] = new_york_last_open_time_7
    raw_data["new_york_last_open_time_8"] = new_york_last_open_time_8
    raw_data["new_york_last_open_time_9"] = new_york_last_open_time_9
    raw_data["new_york_last_open_time_10"] = new_york_last_open_time_10
    raw_data["new_york_last_open_time_11"] = new_york_last_open_time_11
    raw_data["new_york_last_open_time_12"] = new_york_last_open_time_12
    raw_data["new_york_last_open_time_13"] = new_york_last_open_time_13
    raw_data["new_york_last_open_time_14"] = new_york_last_open_time_14
    raw_data["new_york_last_open_time_15"] = new_york_last_open_time_15
    raw_data["new_york_last_open_time_16"] = new_york_last_open_time_16
    raw_data["new_york_last_open_time_17"] = new_york_last_open_time_17
    raw_data["new_york_last_open_time_18"] = new_york_last_open_time_18
    raw_data["new_york_last_open_time_19"] = new_york_last_open_time_19
    raw_data["new_york_last_close_time"] = new_york_last_close_time_0
    raw_data["new_york_last_close_time_0"] = new_york_last_close_time_0
    raw_data["new_york_last_close_time_1"] = new_york_last_close_time_1
    raw_data["new_york_last_close_time_2"] = new_york_last_close_time_2
    raw_data["new_york_last_close_time_3"] = new_york_last_close_time_3
    raw_data["new_york_last_close_time_4"] = new_york_last_close_time_4
    raw_data["new_york_last_close_time_5"] = new_york_last_close_time_5
    raw_data["new_york_last_close_time_6"] = new_york_last_close_time_6
    raw_data["new_york_last_close_time_7"] = new_york_last_close_time_7
    raw_data["new_york_last_close_time_8"] = new_york_last_close_time_8
    raw_data["new_york_last_close_time_9"] = new_york_last_close_time_9
    raw_data["new_york_last_close_time_10"] = new_york_last_close_time_10
    raw_data["new_york_last_close_time_11"] = new_york_last_close_time_11
    raw_data["new_york_last_close_time_12"] = new_york_last_close_time_12
    raw_data["new_york_last_close_time_13"] = new_york_last_close_time_13
    raw_data["new_york_last_close_time_14"] = new_york_last_close_time_14
    raw_data["new_york_last_close_time_15"] = new_york_last_close_time_15
    raw_data["new_york_last_close_time_16"] = new_york_last_close_time_16
    raw_data["new_york_last_close_time_17"] = new_york_last_close_time_17
    raw_data["new_york_last_close_time_18"] = new_york_last_close_time_18
    raw_data["new_york_last_close_time_19"] = new_york_last_close_time_19
    raw_data["london_last_open_time"] = london_last_open_time_0
    raw_data["london_last_open_time_0"] = london_last_open_time_0
    raw_data["london_last_open_time_1"] = london_last_open_time_1
    raw_data["london_last_open_time_2"] = london_last_open_time_2
    raw_data["london_last_open_time_3"] = london_last_open_time_3
    raw_data["london_last_open_time_4"] = london_last_open_time_4
    raw_data["london_last_open_time_5"] = london_last_open_time_5
    raw_data["london_last_open_time_6"] = london_last_open_time_6
    raw_data["london_last_open_time_7"] = london_last_open_time_7
    raw_data["london_last_open_time_8"] = london_last_open_time_8
    raw_data["london_last_open_time_9"] = london_last_open_time_9
    raw_data["london_last_open_time_10"] = london_last_open_time_10
    raw_data["london_last_open_time_11"] = london_last_open_time_11
    raw_data["london_last_open_time_12"] = london_last_open_time_12
    raw_data["london_last_open_time_13"] = london_last_open_time_13
    raw_data["london_last_open_time_14"] = london_last_open_time_14
    raw_data["london_last_open_time_15"] = london_last_open_time_15
    raw_data["london_last_open_time_16"] = london_last_open_time_16
    raw_data["london_last_open_time_17"] = london_last_open_time_17
    raw_data["london_last_open_time_18"] = london_last_open_time_18
    raw_data["london_last_open_time_19"] = london_last_open_time_19

    raw_data["london_last_close_time"] = london_last_close_time_0
    raw_data["london_last_close_time_0"] = london_last_close_time_0
    raw_data["london_last_close_time_1"] = london_last_close_time_1
    raw_data["london_last_close_time_2"] = london_last_close_time_2
    raw_data["london_last_close_time_3"] = london_last_close_time_3
    raw_data["london_last_close_time_4"] = london_last_close_time_4
    raw_data["london_last_close_time_5"] = london_last_close_time_5
    raw_data["london_last_close_time_6"] = london_last_close_time_6
    raw_data["london_last_close_time_7"] = london_last_close_time_7
    raw_data["london_last_close_time_8"] = london_last_close_time_8
    raw_data["london_last_close_time_9"] = london_last_close_time_9
    raw_data["london_last_close_time_10"] = london_last_close_time_10
    raw_data["london_last_close_time_11"] = london_last_close_time_11
    raw_data["london_last_close_time_12"] = london_last_close_time_12
    raw_data["london_last_close_time_13"] = london_last_close_time_13
    raw_data["london_last_close_time_14"] = london_last_close_time_14
    raw_data["london_last_close_time_15"] = london_last_close_time_15
    raw_data["london_last_close_time_16"] = london_last_close_time_16
    raw_data["london_last_close_time_17"] = london_last_close_time_17
    raw_data["london_last_close_time_18"] = london_last_close_time_18
    raw_data["london_last_close_time_19"] = london_last_close_time_19

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

    raw_data["new_york_open_time"] = new_york_open_time
    raw_data["new_york_close_time"] = new_york_close_time
    raw_data["london_open_time"] = london_open_time
    raw_data["london_close_time"] = london_close_time
    return raw_data

@parameterize(
    time_high_1d_ff_shift_1d={"steps": value(1), "shift_col":source("time_high_1d_ff"),"col_name":value("time_high_1d_ff")},
    time_low_1d_ff_shift_1d={"steps": value(1), "shift_col":source("time_low_1d_ff"),"col_name":value("time_low_1d_ff")},
    time_high_5d_ff_shift_5d={"steps": value(5), "shift_col":source("time_high_5d_ff"),"col_name":value("time_high_5d_ff")},
    time_low_5d_ff_shift_5d={"steps": value(5), "shift_col":source("time_low_5d_ff"),"col_name":value("time_low_5d_ff")},
    time_high_11d_ff_shift_11d={"steps": value(11), "shift_col":source("time_high_11d_ff"),"col_name":value("time_high_11d_ff")},
    time_low_11d_ff_shift_11d={"steps": value(11), "shift_col":source("time_low_11d_ff"),"col_name":value("time_low_11d_ff")},
    time_high_21d_ff_shift_21d={"steps": value(21), "shift_col":source("time_high_21d_ff"),"col_name":value("time_high_21d_ff")},
    time_low_21d_ff_shift_21d={"steps": value(21), "shift_col":source("time_low_21d_ff"),"col_name":value("time_low_21d_ff")},
    time_high_51d_ff_shift_51d={"steps": value(51), "shift_col":source("time_high_51d_ff"),"col_name":value("time_high_51d_ff")},
    time_low_51d_ff_shift_51d={"steps": value(51), "shift_col":source("time_low_51d_ff"),"col_name":value("time_low_51d_ff")},
    time_high_101d_ff_shift_101d={"steps": value(101), "shift_col":source("time_high_101d_ff"),"col_name":value("time_high_101d_ff")},
    time_low_101d_ff_shift_101d={"steps": value(101), "shift_col":source("time_low_101d_ff"),"col_name":value("time_low_101d_ff")},
    time_high_201d_ff_shift_201d={"steps": value(201), "shift_col":source("time_high_201d_ff"),"col_name":value("time_high_201d_ff")},
    time_low_201d_ff_shift_201d={"steps": value(201), "shift_col":source("time_low_201d_ff"),"col_name":value("time_low_201d_ff")},
)
def shift_time_tmpl(steps:int, shift_col:pd.Series, timestamp:pd.Series, interval_per_day:int, col_name:str, ticker:pd.Series) -> pd.Series:
    #logging.error(f"shift_tmpl, col_name:{col_name}, before reset_index timestamp:{timestamp}")
    #timestamp = timestamp.reset_index()
    #logging.error(f"shift_tmpl, col_name:{col_name}, shift_col:{shift_col.iloc[50:100]}")
    #logging.error(f"shift_tmpl, col_name:{col_name}, timestamp:{timestamp}")
    #logging.error(f"shift_tmpl, col_name:{col_name}, ticker:{ticker}")
    df = pd.concat([shift_col], axis=1)
    #logging.error(f"shift_tmpl_shift_col, col_name:{col_name}, df:{df.iloc[50:100]}")
    #logging.error(f"df after reset:{df}")
    series = df.groupby(by='ticker', group_keys=True).transform(lambda x:x.shift(-steps*interval_per_day)).reset_index()
    #logging.error(f"shift_tmpl series:{series.iloc[50:100]}")
    series = series.set_index(["timestamp","ticker"])
    #logging.error(f"shift_tmpl_col_name:{col_name}, series:{series.iloc[50:100]}, interval_per_day:{interval_per_day}, steps:{steps}")
    return series
