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
    #logging.error(f"bollinger:{bollinger}")
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

def daily_open(next_daily_close_time:pd.Series, timestamp:pd.Series, ticker:pd.Series, open:pd.Series) -> pd.Series:
    tuples = list(zip(timestamp, ticker, next_daily_close_time))
    index = pd.MultiIndex.from_tuples(tuples, names=["timestamp", "ticker", "close_time"])
    open = open.set_axis(index)
    
    temp = open.loc[index.get_level_values('close_time')==1277496000]
    #logging.error(f"daily_open_temp:{temp}")
    #logging.error(f"daily_open_close:{open.iloc[50:150]}")
    agg = open.sort_index().groupby(["close_time", "ticker"]).agg('first')
    #logging.error(f"daily_open_agg:{agg}")
    return agg

def daily_close(next_daily_close_time:pd.Series, timestamp:pd.Series, ticker:pd.Series, close:pd.Series) -> pd.Series:
    tuples = list(zip(timestamp, ticker, next_daily_close_time))
    index = pd.MultiIndex.from_tuples(tuples, names=["timestamp", "ticker", "close_time"])
    close = close.set_axis(index)
    #sample = close[close.1279742400
    #temp = close[(close.index.close_time==1206648000)]
    temp = close.loc[index.get_level_values('close_time')==1207080000]
    #logging.error(f"daily_close_temp:{temp}")
    #logging.error(f"daily_close_close:{close.iloc[50:150]}")
    agg = close.sort_index().groupby(["close_time", "ticker"]).agg('last')
    #logging.error(f"daily_close_agg:{agg}")
    return agg

def daily_high(next_daily_close_time:pd.Series, timestamp:pd.Series, ticker:pd.Series, high:pd.Series) -> pd.Series:
    tuples = list(zip(timestamp, ticker, next_daily_close_time))
    index = pd.MultiIndex.from_tuples(tuples, names=["timestamp", "ticker", "close_time"])
    high = high.set_axis(index)
    #sample = close[close.1279742400
    #temp = close[(close.index.close_time==1206648000)]
    temp = high.loc[index.get_level_values('close_time')==1207080000]
    #logging.error(f"daily_close_temp:{temp}")
    #logging.error(f"daily_close_close:{high.iloc[50:150]}")
    agg = high.sort_index().groupby(["close_time", "ticker"]).agg('max')
    #logging.error(f"daily_close_agg:{agg}")
    return agg


def daily_low(next_daily_close_time:pd.Series, timestamp:pd.Series, ticker:pd.Series, low:pd.Series) -> pd.Series:
    tuples = list(zip(timestamp, ticker, next_daily_close_time))
    index = pd.MultiIndex.from_tuples(tuples, names=["timestamp", "ticker", "close_time"])
    low = low.set_axis(index)
    agg = low.sort_index().groupby(["close_time", "ticker"]).agg('min')
    return agg

def weekly_open(next_weekly_close_time:pd.Series, timestamp:pd.Series, ticker:pd.Series, open:pd.Series) -> pd.Series:
    tuples = list(zip(timestamp, ticker, next_weekly_close_time))
    index = pd.MultiIndex.from_tuples(tuples, names=["timestamp", "ticker", "close_time"])
    open = open.set_axis(index)
    agg = open.sort_index().groupby(["close_time", "ticker"]).agg('first')
    return agg


def weekly_close(next_weekly_close_time:pd.Series, timestamp:pd.Series, ticker:pd.Series, close:pd.Series) -> pd.Series:
    tuples = list(zip(timestamp, ticker, next_weekly_close_time))
    index = pd.MultiIndex.from_tuples(tuples, names=["timestamp", "ticker", "close_time"])
    temp = close.loc[index.get_level_values('close_time')==1254513600]
    #logging.error(f"weekly_close_temp:{temp}")
    #logging.error(f"weekly_close_close:{close.iloc[50:150]}")
    close = close.set_axis(index)
    agg = close.sort_index().groupby(["close_time", "ticker"]).agg('last')
    #logging.error(f"weekly_close_agg:{agg}")
    return agg


def weekly_high(next_weekly_close_time:pd.Series, timestamp:pd.Series, ticker:pd.Series, high:pd.Series) -> pd.Series:
    tuples = list(zip(timestamp, ticker, next_weekly_close_time))
    index = pd.MultiIndex.from_tuples(tuples, names=["timestamp", "ticker", "close_time"])
    high = high.set_axis(index)
    agg = high.sort_index().groupby(["close_time", "ticker"]).agg('max')
    return agg

def weekly_low(next_weekly_close_time:pd.Series, timestamp:pd.Series, ticker:pd.Series, low:pd.Series) -> pd.Series:
    tuples = list(zip(timestamp, ticker, next_weekly_close_time))
    index = pd.MultiIndex.from_tuples(tuples, names=["timestamp", "ticker", "close_time"])
    low = low.set_axis(index)
    agg = low.groupby(["close_time", "ticker"]).agg('min')
    return agg

def monthly_open(next_monthly_close_time:pd.Series, timestamp:pd.Series, ticker:pd.Series, open:pd.Series) -> pd.Series:
    tuples = list(zip(timestamp, ticker, next_monthly_close_time))
    index = pd.MultiIndex.from_tuples(tuples, names=["timestamp", "ticker", "close_time"])
    open = open.set_axis(index)
    agg = open.groupby(["close_time", "ticker"]).agg('first')
    return agg

def monthly_close(next_monthly_close_time:pd.Series, timestamp:pd.Series, ticker:pd.Series, close:pd.Series) -> pd.Series:
    tuples = list(zip(timestamp, ticker, next_monthly_close_time))
    index = pd.MultiIndex.from_tuples(tuples, names=["timestamp", "ticker", "close_time"])
    temp = close.loc[index.get_level_values('close_time')==1206993600]
    #logging.error(f"monthly_close_temp:{temp}")
    #logging.error(f"monthly_close_close:{close.iloc[50:150]}")
    close = close.set_axis(index)
    agg = close.sort_index().groupby(["close_time", "ticker"]).agg('last')
    return agg

def monthly_high(next_monthly_close_time:pd.Series, timestamp:pd.Series, ticker:pd.Series, high:pd.Series) -> pd.Series:
    tuples = list(zip(timestamp, ticker, next_monthly_close_time))
    index = pd.MultiIndex.from_tuples(tuples, names=["timestamp", "ticker", "close_time"])
    high = high.set_axis(index)
    agg = high.groupby(["close_time", "ticker"]).agg('max')
    return agg

def monthly_low(next_monthly_close_time:pd.Series, timestamp:pd.Series, ticker:pd.Series, low:pd.Series) -> pd.Series:
    tuples = list(zip(timestamp, ticker, next_monthly_close_time))
    index = pd.MultiIndex.from_tuples(tuples, names=["timestamp", "ticker", "close_time"])
    low = low.set_axis(index)
    agg = low.groupby(["close_time", "ticker"]).agg('min')
    return agg

def df_by_daily(last_daily_close_time:pd.Series, timestamp:pd.Series, ticker:pd.Series, open:pd.Series,
                high:pd.Series, close:pd.Series, low:pd.Series) -> pd.DataFrame:
    logging.error(f"last_daily_close_time:{last_daily_close_time}")
    logging.error(f"ticker:{ticker}")
    COLUMN_NAMES = ["last_daily_close_time", "ticker", "open", "high", "close", "low"]
    df_dict = dict.fromkeys(COLUMN_NAMES, [])
    df_dict["last_daily_close_time"] = pd.Series(last_daily_close_time.values)
    df_dict["timestamp"] = pd.Series(timestamp)
    df_dict["ticker"] = pd.Series(ticker)
    df_dict["open"] = pd.Series(open.values)
    df_dict["high"] = pd.Series(high.values)
    df_dict["close"] = pd.Series(close.values)
    df_dict["low"] = pd.Series(low.values)
    logging.error(f"open:{open}")
    logging.error(f"close:{close}")
    logging.error(f"timestamp:{timestamp}")
    logging.error(f"high:{high}")
    logging.error(f"low:{low}")
    df = pd.DataFrame.from_dict(df_dict)[COLUMN_NAMES]
    #df = pd.concat([ticker, last_daily_close_time, open, high, close, low], axis=1)
    #df = pd.concat([ticker, open], axis=1) 
    #df = df.sort_values(["ticker", "timestamp"])
    #logging.error(f"df:{df}")
    #agg = df.groupby(["ticker","last_daily_close_time"]).agg({'open':'first','close':'last','high':'max','low':'min'})
    #logging.error(f"agg:{agg}")
    #agg.columns = ["daily_open", "daily_close", "daily_high", "daily_low"]
    return df

#@extract_columns("daily_open", "daily_close", "daily_high", "daily_low", fill_with=-1)
#def df_by_daily(last_daily_close_time:pd.Series, ticker:pd.Series, open:pd.Series,
#                high:pd.Series, close:pd.Series, low:pd.Series) -> pd.DataFrame:
#    logging.error(f"last_daily_close_time:{last_daily_close_time}")
#    logging.error(f"ticker:{ticker}")
#    df = pd.concat([ticker, last_daily_close_time, open, high, close, low], axis=1)
#    agg = open_df.groupby(["ticker","last_daily_close_time"]).agg({'open':'first','close':'last','high':'max','low':'min'})
#    agg.columns = ["daily_open", "daily_close", "daily_high", "daily_low"]
#    return agg
    
@parameterize(
    next_new_york_close={"time_col": value("new_york_close_time"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    next_london_close={"time_col": value("london_close_time"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    next_weekly_close={"time_col": value("weekly_close_time"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    next_monthly_close={"time_col": value("monthly_close_time"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
    last_option_expiration_close={"time_col": value("last_option_expiration_time"), "pre_interval_mins":value(0), "post_interval_mins":source("interval_mins")},
)
def price_at_tmpl(time_features: pd.DataFrame, time_col: str,
                  pre_interval_mins:int, post_interval_mins:int) -> pd.Series:
    logging.error(f"time_features time_col:{time_col}, pre_interval_mins:{pre_interval_mins}, post_interval_mins:{post_interval_mins}, {time_features.iloc[5:10]}")
    series = time_features.apply(fill_close, time_col=time_col, axis=1).ffill()
    logging.error(f"price_at_tmpl series:{series.iloc[10:15]}")
    return series

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
    #logging.error(f"close:{close}")
    series = close.groupby(['ticker']).transform(lambda x: x.rolling(steps*interval_per_day).max().ffill())
    #logging.error(f"series:{series}")
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
    #logging.error(f"close_low_day_tmpl close:{close.iloc[50:100]}, steps:{steps}")
    res = close.groupby(['ticker']).transform(lambda x: x.rolling(steps*interval_per_day).min().ffill())
    #logging.error(f"close_low_day_tmpl res:{res.iloc[50:100]}, steps:{steps}")
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


def last_daily_close(last_daily_close_0: pd.Series) -> pd.Series:
    return last_daily_close_0

def last_weekly_close(last_weekly_close_0: pd.Series) -> pd.Series:
    return last_weekly_close_0

def last_monthly_close(last_monthly_close_0: pd.Series) -> pd.Series:
    return last_monthly_close_0

@parameterize(
    last_daily_close_0={"steps": value(0),"price_col":source("daily_close")},
    last_daily_close_1={"steps": value(1),"price_col":source("daily_close")},
    last_daily_close_2={"steps": value(2),"price_col":source("daily_close")},
    last_daily_close_3={"steps": value(3),"price_col":source("daily_close")},
    last_daily_close_4={"steps": value(4),"price_col":source("daily_close")},
    last_daily_close_5={"steps": value(5),"price_col":source("daily_close")},
    last_daily_close_6={"steps": value(6),"price_col":source("daily_close")},
    last_daily_close_7={"steps": value(7),"price_col":source("daily_close")},
    last_daily_close_8={"steps": value(8),"price_col":source("daily_close")},
    last_daily_close_9={"steps": value(9),"price_col":source("daily_close")},
    last_daily_close_10={"steps": value(10),"price_col":source("daily_close")},
    last_daily_close_11={"steps": value(11),"price_col":source("daily_close")},
    last_daily_close_12={"steps": value(12),"price_col":source("daily_close")},
    last_daily_close_13={"steps": value(13),"price_col":source("daily_close")},
    last_daily_close_14={"steps": value(14),"price_col":source("daily_close")},
    last_daily_close_15={"steps": value(15),"price_col":source("daily_close")},
    last_daily_close_16={"steps": value(16),"price_col":source("daily_close")},
    last_daily_close_17={"steps": value(17),"price_col":source("daily_close")},
    last_daily_close_18={"steps": value(18),"price_col":source("daily_close")},
    last_daily_close_19={"steps": value(19),"price_col":source("daily_close")},

    last_weekly_close_0={"steps": value(0),"price_col":source("weekly_close")},
    last_weekly_close_1={"steps": value(1),"price_col":source("weekly_close")},
    last_weekly_close_2={"steps": value(2),"price_col":source("weekly_close")},
    last_weekly_close_3={"steps": value(3),"price_col":source("weekly_close")},
    last_weekly_close_4={"steps": value(4),"price_col":source("weekly_close")},
    last_weekly_close_5={"steps": value(5),"price_col":source("weekly_close")},
    last_weekly_close_6={"steps": value(6),"price_col":source("weekly_close")},
    last_weekly_close_7={"steps": value(7),"price_col":source("weekly_close")},
    last_weekly_close_8={"steps": value(8),"price_col":source("weekly_close")},
    last_weekly_close_9={"steps": value(9),"price_col":source("weekly_close")},
    last_weekly_close_10={"steps": value(10),"price_col":source("weekly_close")},
    last_weekly_close_11={"steps": value(11),"price_col":source("weekly_close")},
    last_weekly_close_12={"steps": value(12),"price_col":source("weekly_close")},
    last_weekly_close_13={"steps": value(13),"price_col":source("weekly_close")},
    last_weekly_close_14={"steps": value(14),"price_col":source("weekly_close")},
    last_weekly_close_15={"steps": value(15),"price_col":source("weekly_close")},
    last_weekly_close_16={"steps": value(16),"price_col":source("weekly_close")},
    last_weekly_close_17={"steps": value(17),"price_col":source("weekly_close")},
    last_weekly_close_18={"steps": value(18),"price_col":source("weekly_close")},
    last_weekly_close_19={"steps": value(19),"price_col":source("weekly_close")},

    last_monthly_close_0={"steps": value(0),"price_col":source("monthly_close")},
    last_monthly_close_1={"steps": value(1),"price_col":source("monthly_close")},
    last_monthly_close_2={"steps": value(2),"price_col":source("monthly_close")},
    last_monthly_close_3={"steps": value(3),"price_col":source("monthly_close")},
    last_monthly_close_4={"steps": value(4),"price_col":source("monthly_close")},
    last_monthly_close_5={"steps": value(5),"price_col":source("monthly_close")},
    last_monthly_close_6={"steps": value(6),"price_col":source("monthly_close")},
    last_monthly_close_7={"steps": value(7),"price_col":source("monthly_close")},
    last_monthly_close_8={"steps": value(8),"price_col":source("monthly_close")},
    last_monthly_close_9={"steps": value(9),"price_col":source("monthly_close")},
    last_monthly_close_10={"steps": value(10),"price_col":source("monthly_close")},
    last_monthly_close_11={"steps": value(11),"price_col":source("monthly_close")},
    last_monthly_close_12={"steps": value(12),"price_col":source("monthly_close")},
    last_monthly_close_13={"steps": value(13),"price_col":source("monthly_close")},
    last_monthly_close_14={"steps": value(14),"price_col":source("monthly_close")},
    last_monthly_close_15={"steps": value(15),"price_col":source("monthly_close")},
    last_monthly_close_16={"steps": value(16),"price_col":source("monthly_close")},
    last_monthly_close_17={"steps": value(17),"price_col":source("monthly_close")},
    last_monthly_close_18={"steps": value(18),"price_col":source("monthly_close")},
    last_monthly_close_19={"steps": value(19),"price_col":source("monthly_close")},
)
def shift_price_by_step_tmpl(price_col:pd.Series, steps:int) -> pd.Series:
    series = price_col.groupby(by='ticker').transform(lambda x:x.shift(steps))
    return series

def joined_last_daily_close_0(close:pd.Series, last_daily_close_0:pd.Series) -> pd.Series:
    df = pd.concat([close, last_daily_close_0], axis=1)
    df.columns = ["close", "last_daily_close_0"]
    logging.error(f"df:{df.iloc[50:100]}")
    series = df["last_daily_close_0"]
    series = series.ffill()
    logging.error(f"series:{series}")
    return series

    
def daily_close_df(last_daily_close_0:pd.Series, last_daily_close_1:pd.Series,
                   last_daily_close_2:pd.Series, last_daily_close_3:pd.Series,
                   last_daily_close_4:pd.Series, last_daily_close_5:pd.Series,
                   last_daily_close_6:pd.Series, last_daily_close_7:pd.Series,
                   last_daily_close_8:pd.Series, last_daily_close_9:pd.Series,
                   last_daily_close_10:pd.Series, last_daily_close_11:pd.Series,
                   last_daily_close_12:pd.Series, last_daily_close_13:pd.Series,
                   last_daily_close_14:pd.Series, last_daily_close_15:pd.Series,
                   last_daily_close_16:pd.Series, last_daily_close_17:pd.Series,
                   last_daily_close_18:pd.Series, last_daily_close_19:pd.Series,) -> pd.DataFrame:
    df = pd.concat([last_daily_close_0, last_daily_close_1, last_daily_close_2, last_daily_close_3, last_daily_close_4,
                    last_daily_close_5, last_daily_close_6, last_daily_close_7, last_daily_close_8, last_daily_close_9,
                    last_daily_close_10, last_daily_close_11, last_daily_close_12, last_daily_close_13, last_daily_close_14,
                    last_daily_close_15, last_daily_close_16, last_daily_close_17, last_daily_close_18, last_daily_close_19], axis=1)
    df.columns=["daily_close_0", "daily_close_1", "daily_close_2", "daily_close_3", "daily_close_4",
                "daily_close_5", "daily_close_6", "daily_close_7", "daily_close_8", "daily_close_9",
                "daily_close_10", "daily_close_11", "daily_close_12", "daily_close_13", "daily_close_14",
                "daily_close_15", "daily_close_16", "daily_close_17", "daily_close_18", "daily_close_19"]
    return df

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
    #timestamp = timestamp.reset_index()
    #logging.error(f"shift_price_tmpl full timestamp before reindex, col_name:{col_name}, timestamp:{timestamp}")
    #logging.error(f"shift_price_tmpl before reindex, col_name:{col_name}, timestamp:{timestamp.iloc[50:100]}")
    #timestamp = timestamp.reindex(["time","ticker"])
    #logging.error(f"shift_price_tmpl, col_name:{col_name}, shift_col:{shift_col.iloc[50:100]}")
    #logging.error(f"shift_price_tmpl, col_name:{col_name}, timestamp:{timestamp.iloc[50:100]}")
    #logging.error(f"shift_price_tmpl, col_name:{col_name}, ticker:{ticker}")
    df = pd.concat([timestamp, shift_col], axis=1)
    #logging.error(f"shift_price_tmpl_shift_col, col_name:{col_name}, df:{df.iloc[50:100]}")
    series = df.groupby(by='ticker', group_keys=True).transform(lambda x:x.shift(-steps*interval_per_day)).reset_index()
    #logging.error(f"shift_price_tmpl series:{series.iloc[50:100]}")
    #series = series.set_index(["ticker","time"])
    #logging.error(f"shift_price_tmpl_col_name:{col_name}, series:{series.iloc[50:100]}, interval_per_day:{interval_per_day}, steps:{steps}")
    return series


