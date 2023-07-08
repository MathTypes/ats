import datetime
import logging
import os

import numpy as np
import pandas as pd
from pyarrow import csv
import pytz
import ray

from mom_trans.classical_strategies import (
    MACDStrategy,
    calc_returns,
    calc_daily_vol,
    calc_vol_scaled_returns,
)
from scipy.signal import argrelmax, argrelmin, argrelextrema, find_peaks
from util import time_util

VOL_THRESHOLD = 5  # multiple to winsorise by
HALFLIFE_WINSORISE = 252

def utc_to_nyse_time(utc_time, interval_minutes):
    utc_time = time_util.round_up(utc_time, interval_minutes)
    nyc_time = pytz.timezone('America/New_York').localize(
        datetime.datetime(utc_time.year, utc_time.month, utc_time.day,
                          utc_time.hour, utc_time.minute))
    # Do not use following.
    # See https://stackoverflow.com/questions/18541051/datetime-and-timezone-conversion-with-pytz-mind-blowing-behaviour
    # why datetime(..., tzinfo) does not work.
    #nyc_time = datetime.datetime(utc_time.year, utc_time.month, utc_time.day,
    #                             utc_time.hour, utc_time.minute,
    #                             tzinfo=pytz.timezone('America/New_York'))
    return nyc_time
    

def compute_minutes_after_daily_close(x):
    # logging.info(f"x:{x}, {type(x)}")
    return x


def compute_minutes_to_daily_close(x):
    return x


def compute_days_to_weekly_close(x):
    return x


def compute_days_to_monthly_close(x):
    return x


def compute_days_to_quarterly_close(x):
    return x


def compute_days_to_option_exipiration(x):
    return x


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
            pass
    # logging.info(f"reading files:{input_dirs}")
    return input_dirs


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
    ds = ds[(ds.hour_of_day > 4) & (ds.hour_of_day < 17)]
    # logging.info(f"ds after filter:{ds.head()}")
    # Need to recompute close_back after filtering
    ds = ds.drop(columns=["close_back", "volume_back", "dv_back"])
    # winsorize using rolling 5X standard deviations to remove outliers
    ewm = ds["close"].ewm(halflife=HALFLIFE_WINSORISE)
    means = ewm.mean()
    stds = ewm.std()
    ds["close"] = np.minimum(ds["close"], means + VOL_THRESHOLD * stds)
    ds["close"] = np.maximum(ds["close"], means - VOL_THRESHOLD * stds)
    ds["daily_returns"] = calc_returns(ds["close"])
    ds["daily_vol"] = calc_daily_vol(ds["daily_returns"])
    ds["week_of_year"] = ds.index.isocalendar().week
    ds["month_of_year"] = ds.index.month
    ds["minutes_after_daily_close"] = ds.time.apply(
        lambda x: compute_minutes_after_daily_close(x)
    )
    ds["minutes_to_daily_close"] = ds.time.apply(
        lambda x: compute_minutes_to_daily_close(x)
    )
    ds["days_to_weekly_close"] = ds.time.apply(
        lambda x: compute_days_to_weekly_close(x)
    )
    ds["days_to_monthly_close"] = ds.time.apply(
        lambda x: compute_days_to_monthly_close(x)
    )
    ds["days_to_quarterly_close"] = ds.time.apply(
        lambda x: compute_days_to_quarterly_close(x)
    )
    ds["days_to_option_expiration"] = ds.time.apply(
        lambda x: compute_days_to_option_exipiration(x)
    )

    trend_combinations = [(8, 24), (16, 48), (32, 96)]
    for short_window, long_window in trend_combinations:
        ds[f"macd_{short_window}_{long_window}"] = MACDStrategy.calc_signal(
            ds["close"], short_window, long_window
        )

    ds_pct_back = ds[["close", "volume", "dv"]].pct_change(periods=1)
    # df_pct_forward = df[["close", "volume", "dv"]].pct_change(periods=-1)
    ds = ds.join(ds_pct_back, rsuffix="_back")
    ds_dup = ds[ds.index.duplicated()]
    if not ds_dup.empty:
        logging.info(f"ds_dup:{ds_dup}")
        #exit(0)
    # .join(df_pct_forward, rsuffix='_fwd')
    # logging.info(f"ds:{ds.head()}")
    ds = ds.dropna()
    # logging.info(f"ds:{ds.info()}")
    return ds

def add_highs(df_cumsum, df_time, width):
    high_idx, _ = find_peaks(df_cumsum, width=width)
    high = df_cumsum.iloc[high_idx].to_frame(name="close_cumsum_high")
    high_time = df_time.iloc[high_idx].to_frame(name="time_high")
    df_high = df_cumsum.to_frame(name="close_cumsum").join(high).join(high_time)
    df_high["close_cumsum_high_ff"] = df_high["close_cumsum_high"].ffill()
    df_high["close_cumsum_high_bf"] = df_high["close_cumsum_high"].bfill()
    return df_high

def add_lows(df_cumsum, df_time, width):
    low_idx, _ = find_peaks(np.negative(df_cumsum), width=width)
    low = df_cumsum.iloc[low_idx].to_frame(name="close_cumsum_low")
    low_time = df_time.iloc[low_idx].to_frame(name="time_low")
    df_low = df_cumsum.to_frame(name="close_cumsum").join(low).join(low_time)
    df_low["close_cumsum_low_ff"] = df_low["close_cumsum_low"].ffill()
    df_low["close_cumsum_low_bf"] = df_low["close_cumsum_low"].bfill()
    return df_low

def ticker_transform(raw_data):
    #logging.info(f"raw_data:{raw_data.iloc[:2]}")
    #raw_data = raw_data.drop(columns=["time_dix"])
    #raw_data = raw_data.insert(0, 'time_idx', range(0, len(raw_data)))
    raw_data["time_idx"] = range(0, len(raw_data))
    raw_data["cum_volume"] = raw_data.volume.cumsum()
    raw_data["cum_dv"] = raw_data.dv.cumsum()
    df_pct_back = raw_data[["close", "volume", "dv"]].pct_change(periods=1)
    df_pct_forward = raw_data[["close", "volume", "dv"]].pct_change(periods=-1)
    raw_data = raw_data.join(df_pct_back, rsuffix="_back").join(df_pct_forward, rsuffix="_fwd")

    raw_data['close_back_cumsum'] = raw_data['close_back'].cumsum()
    raw_data['volume_back_cumsum'] = raw_data['volume_back'].cumsum()

    close_back_cumsum = raw_data['close_back_cumsum']
    timestamp = raw_data['timestamp']
    df = add_highs(close_back_cumsum, timestamp, width=21)
    raw_data["close_high_21_ff"] = df["close_cumsum_high_ff"]
    raw_data["close_high_21_bf"] = df["close_cumsum_high_bf"]
    df = add_lows(close_back_cumsum, timestamp, width=21)
    raw_data["close_low_21_ff"] = df["close_cumsum_low_ff"]
    raw_data["close_low_21_bf"] = df["close_cumsum_low_bf"]
    df = add_highs(close_back_cumsum, timestamp, width=51)
    raw_data["close_high_51_ff"] = df["close_cumsum_high_ff"]
    raw_data["close_high_51_bf"] = df["close_cumsum_high_bf"]
    df = add_lows(close_back_cumsum, timestamp, width=51)
    raw_data["close_low_51_ff"] = df["close_cumsum_low_ff"]
    raw_data["close_low_51_bf"] = df["close_cumsum_low_bf"]
    df = add_highs(close_back_cumsum, timestamp, width=201)
    raw_data["close_high_201_ff"] = df["close_cumsum_high_ff"]
    raw_data["close_high_201_bf"] = df["close_cumsum_high_bf"]
    df = add_lows(close_back_cumsum, timestamp, width=201)
    raw_data["close_low_201_ff"] = df["close_cumsum_low_ff"]
    raw_data["close_low_201_bf"] = df["close_cumsum_low_bf"]
    return raw_data

def add_derived_features(raw_data : pd.DataFrame, interval_minutes):
    raw_data = raw_data.reset_index()
    # TODO: the original time comes from aggregated time is UTC, but actually
    # has New York time in it.
    raw_data["time"] = raw_data.time.apply(utc_to_nyse_time, interval_minutes=interval_minutes)
    raw_data["timestamp"] = raw_data.time.apply(lambda x: int(x.timestamp()))
    raw_data["month"] = raw_data.time.dt.month  # categories have be strings
    raw_data["year"] = raw_data.time.dt.year  # categories have be strings
    raw_data["hour_of_day"] = raw_data.time.apply(lambda x: x.hour)
    raw_data["day_of_week"] = raw_data.time.apply(lambda x: x.dayofweek)
    raw_data["day_of_month"] = raw_data.time.apply(lambda x: x.day)
    raw_data["new_idx"] = raw_data.apply(
        lambda x: x.ticker + "_" + str(x.timestamp), axis=1)
    raw_data = raw_data.set_index("new_idx")
    raw_data = raw_data.sort_values(["ticker", "time"])
    #logging.info(f"raw_data: {raw_data.iloc[:3]}")
    new_features = raw_data.groupby(["ticker"])['volume','dv','close','timestamp'].apply(ticker_transform)
    raw_data = raw_data.join(new_features, rsuffix="_back")
    # Drop duplicae columns
    raw_data = raw_data.loc[:,~raw_data.columns.duplicated()]
    #logging.info(f"raw_data: {raw_data.iloc[-5:]}")
    #raw_data = raw_data.dropna()
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
        df_pct_forward = df[["close", "volume", "dv"]].pct_change(periods=-1)
        df = df.join(df_pct_back, rsuffix="_back").join(df_pct_forward, rsuffix="_fwd")
        df["month"] = df.time.dt.month  # categories have be strings
        df["year"] = df.time.dt.year  # categories have be strings
        df["hour_of_day"] = df.time.apply(lambda x: x.hour)
        df["day_of_week"] = df.time.apply(lambda x: x.dayofweek)
        df["day_of_month"] = df.time.apply(lambda x: x.day)
        df = df.dropna()
        return df
