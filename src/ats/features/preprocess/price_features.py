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

@parameterize(
    _close_high_1d_ff={"col_name": value("close_high_1d_ff")},
    _close_high_5d_ff={"col_name": value("close_high_5d_ff")},
    _close_high_11d_ff={"col_name": value("close_high_11d_ff")},
    _close_high_21d_ff={"col_name": value("close_high_21d_ff")},
    _close_high_51d_ff={"col_name": value("close_high_51d_ff")},
    _close_high_101d_ff={"col_name": value("close_high_101d_ff")},
    _close_high_201d_ff={"col_name": value("close_high_201d_ff")},
    _close_low_1d_ff={"col_name": value("close_low_1d_ff")},
    _close_low_5d_ff={"col_name": value("close_low_5d_ff")},
    _close_low_11d_ff={"col_name": value("close_low_11d_ff")},
    _close_low_21d_ff={"col_name": value("close_low_21d_ff")},
    _close_low_51d_ff={"col_name": value("close_low_51d_ff")},
    _close_low_101d_ff={"col_name": value("close_low_101d_ff")},
    _close_low_201d_ff={"col_name": value("close_low_201d_ff")},
    _close_high_1_ff={"col_name": value("close_high_1_ff")},
    _close_high_5_ff={"col_name": value("close_high_5_ff")},
    _close_high_11_ff={"col_name": value("close_high_11_ff")},
    _close_high_21_ff={"col_name": value("close_high_21_ff")},
    _close_high_51_ff={"col_name": value("close_high_51_ff")},
    _close_high_101_ff={"col_name": value("close_high_101_ff")},
    _close_high_201_ff={"col_name": value("close_high_201_ff")},
    _close_low_1_ff={"col_name": value("close_low_1_ff")},
    _close_low_5_ff={"col_name": value("close_low_5_ff")},
    _close_low_11_ff={"col_name": value("close_low_11_ff")},
    _close_low_21_ff={"col_name": value("close_low_21_ff")},
    _close_low_51_ff={"col_name": value("close_low_51_ff")},
    _close_low_101_ff={"col_name": value("close_low_101_ff")},
    _close_low_201_ff={"col_name": value("close_low_201_ff")},
    _time_high_1d_ff={"col_name": value("time_high_1d_ff")},
    _time_low_1d_ff={"col_name": value("time_low_1d_ff")},
    _time_high_5d_ff={"col_name": value("time_high_5d_ff")},
    _time_low_5d_ff={"col_name": value("time_low_5d_ff")},
    _time_high_11d_ff={"col_name": value("time_high_11d_ff")},
    _time_low_11d_ff={"col_name": value("time_low_11d_ff")},
    _time_high_21d_ff={"col_name": value("time_high_21d_ff")},
    _time_low_21d_ff={"col_name": value("time_low_21d_ff")},
    _time_high_51d_ff={"col_name": value("time_high_51d_ff")},
    _time_low_51d_ff={"col_name": value("time_low_51d_ff")},
    _time_high_101d_ff={"col_name": value("time_high_101d_ff")},
    _time_low_101d_ff={"col_name": value("time_low_101d_ff")},
    _time_high_201d_ff={"col_name": value("time_high_201d_ff")},
    _time_low_201d_ff={"col_name": value("time_low_201d_ff")},
    _time_high_5_ff={"col_name": value("time_high_5_ff")},
    _time_low_5_ff={"col_name": value("time_low_5_ff")},
    _time_high_11_ff={"col_name": value("time_high_11_ff")},
    _time_low_11_ff={"col_name": value("time_low_11_ff")},
    _time_high_21_ff={"col_name": value("time_high_21_ff")},
    _time_low_21_ff={"col_name": value("time_low_21_ff")},
    _time_high_51_ff={"col_name": value("time_high_51_ff")},
    _time_low_51_ff={"col_name": value("time_low_51_ff")},
    _time_high_101_ff={"col_name": value("time_high_101_ff")},
    _time_low_101_ff={"col_name": value("time_low_101_ff")},
    _time_high_201_ff={"col_name": value("time_high_201_ff")},
    _time_low_201_ff={"col_name": value("time_low_201_ff")},
    _time_high_1d_ff_shift_1d={"col_name": value("time_high_1d_ff_shift_1d")},
    _time_high_5d_ff_shift_5d={"col_name": value("time_high_5d_ff_shift_5d")},
    _time_high_11d_ff_shift_11d={"col_name": value("time_high_11d_ff_shift_11d")},
    _time_high_21d_ff_shift_21d={"col_name": value("time_high_21d_ff_shift_21d")},
    _time_low_1d_ff_shift_1d={"col_name": value("time_low_1d_ff_shift_1d")},
    _time_low_5d_ff_shift_5d={"col_name": value("time_low_5d_ff_shift_5d")},
    _time_low_11d_ff_shift_11d={"col_name": value("time_low_11d_ff_shift_11d")},
    _time_low_21d_ff_shift_21d={"col_name": value("time_low_21d_ff_shift_21d")},
)
def group_feature_colt(group_features:pd.DataFrame, col_name:str) -> pd.Series:
    return group_features[col_name]


@parameterize(
    next_new_york_close={"time_col": value("new_york_close_time"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    next_weekly_close={"time_col": value("weekly_close_time"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    next_monthly_close={"time_col": value("monthly_close_time"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_option_expiration_close={"time_col": value("last_option_expiration_time"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_open_0={"time_col": value("new_york_last_open_time_0"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_open_1={"time_col": value("new_york_last_open_time_1"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_open_2={"time_col": value("new_york_last_open_time_2"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_open_3={"time_col": value("new_york_last_open_time_3"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_open_4={"time_col": value("new_york_last_open_time_4"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_open_5={"time_col": value("new_york_last_open_time_5"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_open_6={"time_col": value("new_york_last_open_time_6"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_open_7={"time_col": value("new_york_last_open_time_7"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_open_8={"time_col": value("new_york_last_open_time_8"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_open_9={"time_col": value("new_york_last_open_time_9"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_open_10={"time_col": value("new_york_last_open_time_10"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_open_11={"time_col": value("new_york_last_open_time_11"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_open_12={"time_col": value("new_york_last_open_time_12"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_open_13={"time_col": value("new_york_last_open_time_13"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_open_14={"time_col": value("new_york_last_open_time_14"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_open_15={"time_col": value("new_york_last_open_time_15"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_open_16={"time_col": value("new_york_last_open_time_16"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_open_17={"time_col": value("new_york_last_open_time_17"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_open_18={"time_col": value("new_york_last_open_time_18"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_open_19={"time_col": value("new_york_last_open_time_19"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_close_0={"time_col": value("new_york_last_close_time_0"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_close_1={"time_col": value("new_york_last_close_time_1"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_close_2={"time_col": value("new_york_last_close_time_2"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_close_3={"time_col": value("new_york_last_close_time_3"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_close_4={"time_col": value("new_york_last_close_time_4"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_close_5={"time_col": value("new_york_last_close_time_5"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_close_6={"time_col": value("new_york_last_close_time_6"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_close_7={"time_col": value("new_york_last_close_time_7"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_close_8={"time_col": value("new_york_last_close_time_8"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_close_9={"time_col": value("new_york_last_close_time_9"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_close_10={"time_col": value("new_york_last_close_time_10"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_close_11={"time_col": value("new_york_last_close_time_11"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_close_12={"time_col": value("new_york_last_close_time_12"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_close_13={"time_col": value("new_york_last_close_time_13"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_close_14={"time_col": value("new_york_last_close_time_14"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_close_15={"time_col": value("new_york_last_close_time_15"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_close_16={"time_col": value("new_york_last_close_time_16"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_close_17={"time_col": value("new_york_last_close_time_17"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_close_18={"time_col": value("new_york_last_close_time_18"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    new_york_last_daily_close_19={"time_col": value("new_york_last_close_time_19"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_open_0={"time_col": value("london_last_open_time_0"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_open_1={"time_col": value("london_last_open_time_1"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_open_2={"time_col": value("london_last_open_time_2"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_open_3={"time_col": value("london_last_open_time_3"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_open_4={"time_col": value("london_last_open_time_4"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_open_5={"time_col": value("london_last_open_time_5"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_open_6={"time_col": value("london_last_open_time_6"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_open_7={"time_col": value("london_last_open_time_7"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_open_8={"time_col": value("london_last_open_time_8"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_open_9={"time_col": value("london_last_open_time_9"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_open_10={"time_col": value("london_last_open_time_10"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_open_11={"time_col": value("london_last_open_time_11"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_open_12={"time_col": value("london_last_open_time_12"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_open_13={"time_col": value("london_last_open_time_13"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_open_14={"time_col": value("london_last_open_time_14"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_open_15={"time_col": value("london_last_open_time_15"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_open_16={"time_col": value("london_last_open_time_16"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_open_17={"time_col": value("london_last_open_time_17"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_open_18={"time_col": value("london_last_open_time_18"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_open_19={"time_col": value("london_last_open_time_19"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_close_0={"time_col": value("london_last_close_time_0"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_close_1={"time_col": value("london_last_close_time_1"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_close_2={"time_col": value("london_last_close_time_2"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_close_3={"time_col": value("london_last_close_time_3"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_close_4={"time_col": value("london_last_close_time_4"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_close_5={"time_col": value("london_last_close_time_5"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_close_6={"time_col": value("london_last_close_time_6"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_close_7={"time_col": value("london_last_close_time_7"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_close_8={"time_col": value("london_last_close_time_8"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_close_9={"time_col": value("london_last_close_time_9"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_close_10={"time_col": value("london_last_close_time_10"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_close_11={"time_col": value("london_last_close_time_11"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_close_12={"time_col": value("london_last_close_time_12"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_close_13={"time_col": value("london_last_close_time_13"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_close_14={"time_col": value("london_last_close_time_14"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_close_15={"time_col": value("london_last_close_time_15"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_close_16={"time_col": value("london_last_close_time_16"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_close_17={"time_col": value("london_last_close_time_17"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_close_18={"time_col": value("london_last_close_time_18"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    london_last_daily_close_19={"time_col": value("london_last_close_time_19"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_weekly_close_0={"time_col": value("last_weekly_close_time_0"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_weekly_close_1={"time_col": value("last_weekly_close_time_1"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_weekly_close_2={"time_col": value("last_weekly_close_time_2"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_weekly_close_3={"time_col": value("last_weekly_close_time_3"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_weekly_close_4={"time_col": value("last_weekly_close_time_4"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_weekly_close_5={"time_col": value("last_weekly_close_time_5"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_weekly_close_6={"time_col": value("last_weekly_close_time_6"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_weekly_close_7={"time_col": value("last_weekly_close_time_7"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_weekly_close_8={"time_col": value("last_weekly_close_time_8"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_weekly_close_9={"time_col": value("last_weekly_close_time_9"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_weekly_close_10={"time_col": value("last_weekly_close_time_10"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_weekly_close_11={"time_col": value("last_weekly_close_time_11"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_weekly_close_12={"time_col": value("last_weekly_close_time_12"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_weekly_close_13={"time_col": value("last_weekly_close_time_13"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_weekly_close_14={"time_col": value("last_weekly_close_time_14"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_weekly_close_15={"time_col": value("last_weekly_close_time_15"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_weekly_close_16={"time_col": value("last_weekly_close_time_16"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_weekly_close_17={"time_col": value("last_weekly_close_time_17"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_weekly_close_18={"time_col": value("last_weekly_close_time_18"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_weekly_close_19={"time_col": value("last_weekly_close_time_19"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_monthly_close_0={"time_col": value("last_monthly_close_time_0"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_monthly_close_1={"time_col": value("last_monthly_close_time_1"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_monthly_close_2={"time_col": value("last_monthly_close_time_2"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_monthly_close_3={"time_col": value("last_monthly_close_time_3"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_monthly_close_4={"time_col": value("last_monthly_close_time_4"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_monthly_close_5={"time_col": value("last_monthly_close_time_5"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_monthly_close_6={"time_col": value("last_monthly_close_time_6"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_monthly_close_7={"time_col": value("last_monthly_close_time_7"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_monthly_close_8={"time_col": value("last_monthly_close_time_8"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_monthly_close_9={"time_col": value("last_monthly_close_time_9"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_monthly_close_10={"time_col": value("last_monthly_close_time_10"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_monthly_close_11={"time_col": value("last_monthly_close_time_11"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_monthly_close_12={"time_col": value("last_monthly_close_time_12"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_monthly_close_13={"time_col": value("last_monthly_close_time_13"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_monthly_close_14={"time_col": value("last_monthly_close_time_14"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_monthly_close_15={"time_col": value("last_monthly_close_time_15"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_monthly_close_16={"time_col": value("last_monthly_close_time_16"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_monthly_close_17={"time_col": value("last_monthly_close_time_17"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_monthly_close_18={"time_col": value("last_monthly_close_time_18"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_monthly_close_19={"time_col": value("last_monthly_close_time_19"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
)
def price_at_tmpl(time_features: pd.DataFrame, time_col: str,
                  pre_interval_mins:int, post_interval_mins:int, interval_mins:int) -> pd.Series:
    return time_features.apply(fill_close, time_col=time_col,
                               pre_interval_mins=pre_interval_mins,
                               post_interval_mins=post_interval_mins,
                               axis=1).ffill()

@parameterize(
    close_back_cumsum_high_1d_ff={"lookback_days": value(1)},
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
    close_back_cumsum_low_1d_ff={"lookback_days": value(1)},
    close_back_cumsum_low_5d_ff={"lookback_days": value(5)},
    close_back_cumsum_low_11d_ff={"lookback_days": value(11)},
    close_back_cumsum_low_21d_ff={"lookback_days": value(21)},
    close_back_cumsum_low_51d_ff={"lookback_days": value(51)},
    close_back_cumsum_low_101d_ff={"lookback_days": value(101)},
    close_back_cumsum_low_201d_ff={"lookback_days": value(201)},
)
def close_back_cumsum_low_ff_tmpl(close_back_cumsum: pd.Series, interval_per_day: int, lookback_days: int) -> pd.Series:
    close_back_cumsum_rolling_min = close_back_cumsum.rolling(lookback_days*interval_per_day).min()
    return close_back_cumsum_rolling_min.ffill()

@parameterize(
    close_back_cumsum_high_5_ff ={"steps": value(5)},
    close_back_cumsum_high_11_ff ={"steps": value(11)},
    close_back_cumsum_high_21_ff ={"steps": value(21)},
    close_back_cumsum_high_51_ff ={"steps": value(51)},
    close_back_cumsum_high_101_ff ={"steps": value(101)},
    close_back_cumsum_high_201_ff ={"steps": value(201)},
)
def close_back_cumsum_max_tmpl(steps:int, close_back_cumsum:pd.Series) -> pd.Series:
    return close_back_cumsum.rolling(steps).max().ffill()

@parameterize(
    close_back_cumsum_low_5_ff ={"steps": value(5)},
    close_back_cumsum_low_11_ff ={"steps": value(11)},
    close_back_cumsum_low_21_ff ={"steps": value(21)},
    close_back_cumsum_low_51_ff ={"steps": value(51)},
    close_back_cumsum_low_101_ff ={"steps": value(101)},
    close_back_cumsum_low_201_ff ={"steps": value(201)},
)
def close_back_cumsum_min_tmpl(steps:int, close_back_cumsum:pd.Series) -> pd.Series:
    return close_back_cumsum.rolling(steps).min().ffill()


@parameterize(
    close_high_5d_bf ={"lookback_days": value(5)},
    close_high_11d_bf ={"lookback_days": value(11)},
    close_high_21d_bf ={"lookback_days": value(21)},
    close_high_51d_bf ={"lookback_days": value(51)},
    close_high_101d_bf ={"lookback_days": value(101)},
    close_high_201d_bf ={"lookback_days": value(201)},
)
def close_high_bf_tmpl(lookback_days:int, close:pd.Series, interval_per_day:int) -> pd.Series:
    return close.rolling(lookback_days*interval_per_day).max().bfill()

@parameterize(
    close_low_5d_bf ={"lookback_days": value(5)},
    close_low_11d_bf ={"lookback_days": value(11)},
    close_low_21d_bf ={"lookback_days": value(21)},
    close_low_51d_bf ={"lookback_days": value(51)},
    close_low_101d_bf ={"lookback_days": value(101)},
    close_low_201d_bf ={"lookback_days": value(201)},
)
def close_low_bf_tmpl(lookback_days:int, close:pd.Series, interval_per_day:int) -> pd.Series:
    return close.rolling(lookback_days*interval_per_day).min().bfill()
