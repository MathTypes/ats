import datetime
import logging
import math
import os
from typing import Dict

from hamilton.function_modifiers import tag

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

@tag(cache="parquet")
def weekly_close_time(timestamp: pd.Series, cal:CMEEquityExchangeCalendar) -> pd.Series:
    new_york_cal = mcal.get_calendar("NYSE")
    return timestamp.apply(
        market_time.compute_weekly_close_time, cal=new_york_cal
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


def time_features(group_features:pd.DataFrame, cal:CMEEquityExchangeCalendar,
                  macro_data_builder:MacroDataBuilder, config:DictConfig,
                  weekly_close_time: pd.Series, last_weekly_close_time: pd.Series,
                  monthly_close_time: pd.Series, last_monthly_close_time: pd.Series,
                  option_expiration_time: pd.Series, last_option_expiration_time: pd.Series) -> pd.DataFrame:
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
    return raw_data
