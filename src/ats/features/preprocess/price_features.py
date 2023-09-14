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
    bollinger_5d_2={"window": value(5), "window_dev":value(2)},
    bollinger_5d_3={"window": value(5), "window_dev":value(3)},
    bollinger_10d_2={"window": value(10), "window_dev":value(2)},
    bollinger_10d_3={"window": value(10), "window_dev":value(3)},
    bollinger_20d_2={"window": value(20), "window_dev":value(2)},
    bollinger_20d_3={"window": value(20), "window_dev":value(3)},
    bollinger_50d_2={"window": value(50), "window_dev":value(2)},
    bollinger_50d_3={"window": value(50), "window_dev":value(3)},
    bollinger_100d_2={"window": value(100), "window_dev":value(2)},
    bollinger_100d_3={"window": value(100), "window_dev":value(3)},
    bollinger_200d_2={"window": value(200), "window_dev":value(2)},
    bollinger_200d_3={"window": value(200), "window_dev":value(3)},
)
def bollinger_day_tmpl(close:pd.Series, window:int, window_dev:int, interval_per_day:int) -> pd.Series:
    logging.error(f"bollinger_day_tmpl_close:{close}")
    return ta.volatility.BollingerBands(close, window=window*interval_per_day, window_dev=window_dev)

@parameterize(
    bollinger_5_2={"window": value(5), "window_dev":value(2)},
    bollinger_5_3={"window": value(5), "window_dev":value(3)},
    bollinger_10_2={"window": value(10), "window_dev":value(2)},
    bollinger_10_3={"window": value(10), "window_dev":value(3)},
    bollinger_20_2={"window": value(20), "window_dev":value(2)},
    bollinger_20_3={"window": value(20), "window_dev":value(3)},
    bollinger_50_2={"window": value(50), "window_dev":value(2)},
    bollinger_50_3={"window": value(50), "window_dev":value(3)},
    bollinger_100_2={"window": value(100), "window_dev":value(2)},
    bollinger_100_3={"window": value(100), "window_dev":value(3)},
    bollinger_200_2={"window": value(200), "window_dev":value(2)},
    bollinger_200_3={"window": value(200), "window_dev":value(3)},
)
def bollinger_tmpl(close:pd.Series, window:int, window_dev:int) -> pd.Series:
    return ta.volatility.BollingerBands(close, window=window, window_dev=window_dev)


@parameterize(
    bb_high_5d_2={"bollinger": source("bollinger_5d_2")},
    bb_high_5d_3={"bollinger": source("bollinger_5d_3")},
    bb_high_10d_2={"bollinger": source("bollinger_10d_2")},
    bb_high_10d_3={"bollinger": source("bollinger_10d_3")},
    bb_high_20d_2={"bollinger": source("bollinger_20d_2")},
    bb_high_20d_3={"bollinger": source("bollinger_20d_3")},
    bb_high_50d_2={"bollinger": source("bollinger_50d_2")},
    bb_high_50d_3={"bollinger": source("bollinger_50d_3")},
    bb_high_100d_2={"bollinger": source("bollinger_100d_2")},
    bb_high_100d_3={"bollinger": source("bollinger_100d_3")},
    bb_high_200d_2={"bollinger": source("bollinger_200d_2")},
    bb_high_200d_3={"bollinger": source("bollinger_200d_3")},
)
def bb_high_day_tmpl(bollinger:pd.Series) -> pd.Series:
    logging.error(f"bollinger:{bollinger}")
    return bollinger.bollinger_hband()

@parameterize(
    bb_low_5d_2={"bollinger": source("bollinger_5d_2")},
    bb_low_5d_3={"bollinger": source("bollinger_5d_3")},
    bb_low_10d_2={"bollinger": source("bollinger_10d_2")},
    bb_low_10d_3={"bollinger": source("bollinger_10d_3")},
    bb_low_20d_2={"bollinger": source("bollinger_20d_2")},
    bb_low_20d_3={"bollinger": source("bollinger_20d_3")},
    bb_low_50d_2={"bollinger": source("bollinger_50d_2")},
    bb_low_50d_3={"bollinger": source("bollinger_50d_3")},
    bb_low_100d_2={"bollinger": source("bollinger_100d_2")},
    bb_low_100d_3={"bollinger": source("bollinger_100d_3")},
    bb_low_200d_2={"bollinger": source("bollinger_200d_2")},
    bb_low_200d_3={"bollinger": source("bollinger_200d_3")},
)
def bb_low_day_tmpl(bollinger:pd.Series) -> pd.Series:
    return bollinger.bollinger_lband()

@parameterize(
    rsi_14d={"lookback_days":value(14)},
    rsi_28d={"lookback_days":value(28)},
    rsi_42d={"lookback_days":value(42)},
)
def rsi_day_tmpl(lookback_days:int, close:pd.Series, interval_per_day:int) -> pd.Series:
    return ta.momentum.RSIIndicator(close=close, window=lookback_days*interval_per_day).rsi() 

@parameterize(
    rsi_14={"lookback_days":value(14)},
    rsi_28={"lookback_days":value(28)},
    rsi_42={"lookback_days":value(42)},
)
def rsi_tmpl(lookback_days:int, close:pd.Series) -> pd.Series:
    return ta.momentum.RSIIndicator(close=close, window=lookback_days).rsi() 

@parameterize(
    sma_5d={"lookback_days":value(5)},
    sma_10d={"lookback_days":value(10)},
    sma_20d={"lookback_days":value(20)},
    sma_50d={"lookback_days":value(50)},
    sma_100d={"lookback_days":value(100)},
    sma_200d={"lookback_days":value(200)},
)
def sma_day_tmpl(lookback_days:int, close:pd.Series, interval_per_day:int) -> pd.Series:
    return ta.trend.SMAIndicator(close=close, window=lookback_days*interval_per_day).sma_indicator()

@parameterize(
    sma_5={"lookback_days":value(5)},
    sma_10={"lookback_days":value(10)},
    sma_20={"lookback_days":value(20)},
    sma_50={"lookback_days":value(50)},
    sma_100={"lookback_days":value(100)},
    sma_200={"lookback_days":value(200)}
)
def sma_tmpl(lookback_days:int, close:pd.Series) -> pd.Series:
    return ta.trend.SMAIndicator(close=close, window=lookback_days).sma_indicator()

@parameterize(
    bb_high_5_2={"bollinger": source("bollinger_5_2")},
    bb_high_5_3={"bollinger": source("bollinger_5_3")},
    bb_high_10_2={"bollinger": source("bollinger_10_2")},
    bb_high_10_3={"bollinger": source("bollinger_10_3")},
    bb_high_20_2={"bollinger": source("bollinger_20_2")},
    bb_high_20_3={"bollinger": source("bollinger_20_3")},
    bb_high_50_2={"bollinger": source("bollinger_50_2")},
    bb_high_50_3={"bollinger": source("bollinger_50_3")},
    bb_high_100_2={"bollinger": source("bollinger_100_2")},
    bb_high_100_3={"bollinger": source("bollinger_100_3")},
    bb_high_200_2={"bollinger": source("bollinger_200_2")},
    bb_high_200_3={"bollinger": source("bollinger_200_3")},
)
def bb_high_tmpl(bollinger:pd.Series) -> pd.Series:
    return bollinger.bollinger_hband()

@parameterize(
    bb_low_5_2={"bollinger": source("bollinger_5_2")},
    bb_low_5_3={"bollinger": source("bollinger_5_3")},
    bb_low_10_2={"bollinger": source("bollinger_10_2")},
    bb_low_10_3={"bollinger": source("bollinger_10_3")},
    bb_low_20_2={"bollinger": source("bollinger_20_2")},
    bb_low_20_3={"bollinger": source("bollinger_20_3")},
    bb_low_50_2={"bollinger": source("bollinger_50_2")},
    bb_low_50_3={"bollinger": source("bollinger_50_3")},
    bb_low_100_2={"bollinger": source("bollinger_100_2")},
    bb_low_100_3={"bollinger": source("bollinger_100_3")},
    bb_low_200_2={"bollinger": source("bollinger_200_2")},
    bb_low_200_3={"bollinger": source("bollinger_200_3")},
)
def bb_low_tmpl(bollinger:pd.Series) -> pd.Series:
    return bollinger.bollinger_lband()
                   
@parameterize(
    next_new_york_close={"time_col": value("new_york_close_time"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    next_london_close={"time_col": value("london_close_time"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
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
    close_high_5_ff={"steps": value(5)},
    close_high_11_ff={"steps": value(11)},
    close_high_21_ff={"steps": value(21)},
    close_high_51_ff={"steps": value(51)},
    close_high_101_ff={"steps": value(101)},
    close_high_201_ff={"steps": value(201)},
)
def close_high_tmpl(steps:int, close:pd.Series) -> pd.Series:
    return close.groupby(['ticker']).transform(lambda x: x.rolling(steps).max().ffill())

@parameterize(
    close_high_1d_ff={"steps": value(1)},
    close_high_5d_ff={"steps": value(5)},
    close_high_11d_ff={"steps": value(11)},
    close_high_21d_ff={"steps": value(21)},
    close_high_51d_ff={"steps": value(51)},
    close_high_101d_ff={"steps": value(101)},
    close_high_201d_ff={"steps": value(201)},
)
def close_high_day_tmpl(steps:int, close:pd.Series, interval_per_day:int) -> pd.Series:
    logging.error(f"close:{close}")
    series = close.groupby(['ticker']).transform(lambda x: x.rolling(steps*interval_per_day).max().ffill())
    logging.error(f"series:{series}")
    return series

@parameterize(
    close_high_5d_bf={"steps": value(5)},
    close_high_11d_bf={"steps": value(11)},
    close_high_21d_bf={"steps": value(21)},
    close_high_51d_bf={"steps": value(51)},
    close_high_101d_bf={"steps": value(101)},
    close_high_201d_bf={"steps": value(201)},
)
def close_high_day_bf_tmpl(steps:int, close:pd.Series, interval_per_day:int) -> pd.Series:
    res = close.groupby(['ticker']).transform(lambda x: x.rolling(steps*interval_per_day).max().bfill())
    return res
    #close_rolling_max = close.rolling(steps*interval_per_day).max()
    #return close_rolling_max.bfill()

@parameterize(
    close_low_5d_bf={"steps": value(5)},
    close_low_11d_bf={"steps": value(11)},
    close_low_21d_bf={"steps": value(21)},
    close_low_51d_bf={"steps": value(51)},
    close_low_101d_bf={"steps": value(101)},
    close_low_201d_bf={"steps": value(201)},
)
def close_low_day_bf_tmpl(steps:int, close:pd.Series, interval_per_day:int) -> pd.Series:
    res = close.groupby(['ticker']).transform(lambda x: x.rolling(steps*interval_per_day).min().bfill())
    #close_rolling_min = close.rolling(steps*interval_per_day).min()
    #return close_rolling_min.bfill()
    return res

@parameterize(
    close_low_1d_ff={"steps": value(1)},
    close_low_5d_ff={"steps": value(5)},
    close_low_11d_ff={"steps": value(11)},
    close_low_21d_ff={"steps": value(21)},
    close_low_51d_ff={"steps": value(51)},
    close_low_101d_ff={"steps": value(101)},
    close_low_201d_ff={"steps": value(201)},
)
def close_low_day_tmpl(steps:int, close:pd.Series, interval_per_day:int) -> pd.Series:
    res = close.groupby(['ticker']).transform(lambda x: x.rolling(steps*interval_per_day).min().ffill())
    logging.error(f"close_low_day_tmpl:{close_low_day_tmpl}")
    return res

@parameterize(
    close_low_5_ff={"steps": value(5)},
    close_low_11_ff={"steps": value(11)},
    close_low_21_ff={"steps": value(21)},
    close_low_51_ff={"steps": value(51)},
    close_low_101_ff={"steps": value(101)},
    close_low_201_ff={"steps": value(201)},
)
def close_low_tmpl(steps:int, close:pd.Series) -> pd.Series:
    return close.groupby(['ticker']).transform(lambda x: x.rolling(steps).min().ffill())

@parameterize(
    close_high_1d_ff_shift_1d={"steps": value(1),"shift_col":source("close_high_1d_ff"),"col_name":value("close_high_1d_ff")},
    close_low_1d_ff_shift_1d={"steps": value(1), "shift_col":source("close_low_1d_ff"),"col_name":value("close_low_1d_ff")},
    close_high_5d_ff_shift_5d={"steps": value(5), "shift_col":source("close_high_5d_ff"),"col_name":value("close_high_5d_ff")},
    close_low_5d_ff_shift_5d={"steps": value(5), "shift_col":source("close_low_5d_ff"),"col_name":value("close_low_5d_ff")},
    close_high_11d_ff_shift_11d={"steps":value(11),"shift_col":source("close_high_11d_ff"),"col_name":value("close_high_11d_ff")},
    close_low_11d_ff_shift_11d={"steps": value(11), "shift_col":source("close_low_11d_ff"),"col_name":value("close_low_11d_ff")},
    close_high_21d_ff_shift_21d={"steps": value(21), "shift_col":source("close_high_21d_ff"),"col_name":value("close_high_21d_ff")},
    close_low_21d_ff_shift_21d={"steps": value(21), "shift_col":source("close_low_21d_ff"),"col_name":value("close_low_21d_ff")},
    close_high_51d_ff_shift_51d={"steps": value(51), "shift_col":source("close_high_51d_ff"),"col_name":value("close_high_51d_ff")},
    close_low_51d_ff_shift_51d={"steps": value(51), "shift_col":source("close_low_51d_ff"),"col_name":value("close_low_51d_ff")},
    close_high_101d_ff_shift_101d={"steps": value(101), "shift_col":source("close_high_101d_ff"),"col_name":value("close_high_101d_ff")},
    close_low_101d_ff_shift_101d={"steps": value(101), "shift_col":source("close_low_101d_ff"),"col_name":value("close_low_101d_ff")},
    close_high_201d_ff_shift_201d={"steps": value(201), "shift_col":source("close_high_201d_ff"),"col_name":value("close_high_201d_ff")},
    close_low_201d_ff_shift_201d={"steps": value(201), "shift_col":source("close_low_201d_ff"),"col_name":value("close_low_201d_ff")},
)
def shift_price_tmpl(steps:int, shift_col:pd.Series, timestamp:pd.Series, interval_per_day:int, col_name:str, ticker:pd.Series) -> pd.Series:
    timestamp = timestamp.reset_index()
    logging.error(f"shift_price_tmpl, col_name:{col_name}, shift_col:{shift_col.iloc[:50]}")
    logging.error(f"shift_price_tmpl, col_name:{col_name}, timestamp:{timestamp.iloc[:50]}")
    logging.error(f"shift_price_tmpl, col_name:{col_name}, ticker:{ticker}")
    df = pd.concat([timestamp, shift_col], axis=1)
    logging.error(f"shift_price_tmpl_shift_col, col_name:{col_name}, df:{df.iloc[:50]}")
    logging.error(f"df after reset:{df}")
    series = df.groupby(by='ticker', group_keys=True).transform(lambda x:x.shift(-steps*interval_per_day)).reset_index()
    logging.error(f"shift_price_tmpl series:{series.iloc[:50]}")
    #series = series.set_index(["ticker","time"])
    logging.error(f"shift_price_tmpl_col_name:{col_name}, series:{series.iloc[:50]}, interval_per_day:{interval_per_day}, steps:{steps}")
    return series

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

