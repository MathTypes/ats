import datetime
import logging
import math
import os

import numpy as np
import pandas as pd
from pyarrow import csv
import ray
import ta

import pandas_market_calendars as mcal
from scipy.signal import find_peaks

from ats.calendar import market_time
from ats.model.mom_trans.classical_strategies import (
    MACDStrategy,
    calc_returns,
    calc_daily_vol,
)
from ats.util import time_util
from ats.util import profile_util

VOL_THRESHOLD = 5  # multiple to winsorise by
HALFLIFE_WINSORISE = 252


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
    # logging.info(f"reading files:{input_dirs}")
    return input_dirs

@profile_util.profile
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
    # Filter out between london open until new york close
    # ds = ds[(ds.hour_of_day > 1) & (ds.hour_of_day < 17)]
    # logging.info(f"ds after filter:{ds.head()}")
    # Need to recompute close_back after filtering
    ds_dup = ds[ds.index.duplicated()]
    if not ds_dup.empty:
        logging.info(f"ds_dup:{ds_dup}")
        # exit(0)
    # .join(df_pct_forward, rsuffix='_fwd')
    # logging.info(f"ds:{ds.head()}")
    ds = ds.dropna()
    # logging.info(f"ds:{ds.info()}")
    return ds


@profile_util.profile
def add_highs(df_cumsum, df_time, width):
    df_z_scaled = df_cumsum
    df_z_scaled = (df_z_scaled - df_z_scaled.mean()) / df_z_scaled.std()  
    high_idx, _ = find_peaks(df_z_scaled, prominence=width)
    high = df_cumsum.iloc[high_idx].to_frame(name="close_cumsum_high")
    high_time = df_time.iloc[high_idx].to_frame(name="time_high")
    df_high = df_cumsum.to_frame(name="close_cumsum").join(high).join(high_time)
    df_high["close_cumsum_high_ff"] = df_high["close_cumsum_high"].ffill()
    df_high["close_cumsum_high_bf"] = df_high["close_cumsum_high"].bfill()
    df_high["time_high_ff"] = df_high["time_high"].ffill()
    del high
    del high_time
    del high_idx
    return df_high


@profile_util.profile
def add_lows(df_cumsum, df_time, width):
    df_z_scaled = df_cumsum.fillna(0)
    df_z_scaled = -(df_z_scaled - df_z_scaled.mean()) / df_z_scaled.std() * 20
    low_idx, _ = find_peaks(df_z_scaled, prominence=width)
    low = df_cumsum.iloc[low_idx].to_frame(name="close_cumsum_low")
    low_time = df_time.iloc[low_idx].to_frame(name="time_low")
    df_low = df_cumsum.to_frame(name="close_cumsum").join(low).join(low_time)
    df_low["close_cumsum_low_ff"] = df_low["close_cumsum_low"].ffill()
    df_low["close_cumsum_low_bf"] = df_low["close_cumsum_low"].bfill()
    df_low["time_low_ff"] = df_low["time_low"].ffill()
    del low_idx
    del low
    del low_time
    return df_low

def get_time(x, close_col):
    if x[close_col] == x["close_back_cumsum"]:
        return x["timestamp"]
    else:
        return None
    
@profile_util.profile
def ticker_transform(raw_data, interval_minutes, base_price=500):
    ewm = raw_data["close"].ewm(halflife=HALFLIFE_WINSORISE)
    means = ewm.mean()
    stds = ewm.std()
    del ewm
    raw_data["close"] = np.minimum(raw_data["close"], means + VOL_THRESHOLD * stds)
    raw_data["close"] = np.maximum(raw_data["close"], means - VOL_THRESHOLD * stds)
    raw_data["cum_volume"] = raw_data.volume.cumsum()
    raw_data["cum_dv"] = raw_data.dv.cumsum()
    raw_data['close_back'] = np.log(raw_data.close+base_price) - np.log(raw_data.close.shift(1)+base_price)
    # Avoid inf
    raw_data['volume_back'] = np.log(raw_data.volume+2) - np.log(raw_data.volume.shift(1)+2)
    raw_data['dv_back'] = np.log(raw_data.dv) - np.log(raw_data.dv.shift(1))
    
    raw_data["close_back_cumsum"] = raw_data["close_back"].cumsum()
    raw_data["volume_back_cumsum"] = raw_data["volume_back"].cumsum()

    close_back_cumsum = raw_data["close_back_cumsum"]
    timestamp = raw_data["timestamp"]
    interval_per_day = int(23 * 60 / interval_minutes)
    # find_peaks only with prominance which needs to be set to half of the width.
    # in case of high among 5 days, the high needs to be higher than 4 points around
    # it, 2 to the left and 2 to the right.
    raw_data['close_rolling_5d_max'] = raw_data.close_back_cumsum.rolling(5*interval_per_day).max()
    raw_data["close_high_5_ff"] = raw_data["close_rolling_5d_max"].ffill()
    raw_data["close_high_5_bf"] = raw_data["close_rolling_5d_max"].bfill()
    raw_data["time_high_5_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_high_5_ff"), axis=1)
    raw_data["time_high_5_ff"]  = raw_data["time_high_5_ff"].ffill()

    raw_data['close_rolling_5d_min'] = raw_data.close_back_cumsum.rolling(5*interval_per_day).min()
    raw_data["close_low_5_ff"] = raw_data["close_rolling_5d_min"].ffill()
    raw_data["close_low_5_bf"] = raw_data["close_rolling_5d_min"].bfill()
    raw_data["time_low_5_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_low_5_ff"), axis=1)
    raw_data["time_low_5_ff"]  = raw_data["time_low_5_ff"].ffill()

    raw_data['close_rolling_11d_max'] = raw_data.close_back_cumsum.rolling(11*interval_per_day).max()
    raw_data["close_high_11_ff"] = raw_data["close_rolling_11d_max"].ffill()
    raw_data["close_high_11_bf"] = raw_data["close_rolling_11d_max"].bfill()
    raw_data["time_high_11_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_high_11_ff"), axis=1)
    raw_data["time_high_11_ff"]  = raw_data["time_high_11_ff"].ffill()

    raw_data['close_rolling_11d_min'] = raw_data.close_back_cumsum.rolling(11*interval_per_day).min()
    raw_data["close_low_11_ff"] = raw_data["close_rolling_11d_min"].ffill()
    raw_data["close_low_11_bf"] = raw_data["close_rolling_11d_min"].bfill()
    raw_data["time_low_11_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_low_11_ff"), axis=1)
    raw_data["time_low_11_ff"]  = raw_data["time_low_11_ff"].ffill()
 
    raw_data['close_rolling_21d_max'] = raw_data.close_back_cumsum.rolling(21*interval_per_day).max()
    raw_data["close_high_21_ff"] = raw_data["close_rolling_21d_max"].ffill()
    raw_data["close_high_21_bf"] = raw_data["close_rolling_21d_max"].bfill()
    raw_data["time_high_21_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_high_21_ff"), axis=1)
    raw_data["time_high_21_ff"]  = raw_data["time_high_21_ff"].ffill()

    raw_data['close_rolling_21d_min'] = raw_data.close_back_cumsum.rolling(21*interval_per_day).min()
    raw_data["close_low_21_ff"] = raw_data["close_rolling_21d_min"].ffill()
    raw_data["close_low_21_bf"] = raw_data["close_rolling_21d_min"].bfill()
    raw_data["time_low_21_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_low_21_ff"), axis=1)
    raw_data["time_low_21_ff"]  = raw_data["time_low_21_ff"].ffill()

    raw_data['close_rolling_51d_max'] = raw_data.close_back_cumsum.rolling(51*interval_per_day).max()
    raw_data["close_high_51_ff"] = raw_data["close_rolling_51d_max"].ffill()
    raw_data["close_high_51_bf"] = raw_data["close_rolling_51d_max"].bfill()
    raw_data["time_high_51_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_high_51_ff"), axis=1)
    raw_data["time_high_51_ff"]  = raw_data["time_high_51_ff"].ffill()

    raw_data['close_rolling_51d_min'] = raw_data.close_back_cumsum.rolling(51*interval_per_day).min()
    raw_data["close_low_51_ff"] = raw_data["close_rolling_51d_min"].ffill()
    raw_data["close_low_51_bf"] = raw_data["close_rolling_51d_min"].bfill()
    raw_data["time_low_51_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_low_51_ff"), axis=1)
    raw_data["time_low_51_ff"]  = raw_data["time_low_51_ff"].ffill()

    raw_data['close_rolling_201d_max'] = raw_data.close_back_cumsum.rolling(201*interval_per_day).max()
    raw_data["close_high_201_ff"] = raw_data["close_rolling_201d_max"].ffill()
    raw_data["close_high_201_bf"] = raw_data["close_rolling_201d_max"].bfill()
    raw_data["time_high_201_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_high_201_ff"), axis=1)
    raw_data["time_high_201_ff"]  = raw_data["time_high_201_ff"].ffill()

    raw_data['close_rolling_201d_min'] = raw_data.close_back_cumsum.rolling(51*interval_per_day).min()
    raw_data["close_low_201_ff"] = raw_data["close_rolling_201d_min"].ffill()
    raw_data["close_low_201_bf"] = raw_data["close_rolling_201d_min"].bfill()
    raw_data["time_low_201_ff"] = raw_data.apply(lambda x: get_time(x, close_col="close_low_201_ff"), axis=1)
    raw_data["time_low_201_ff"]  = raw_data["time_low_201_ff"].ffill()
    del close_back_cumsum

    raw_data['close_rolling_5d_max'] = raw_data.close_back_cumsum.rolling(5*interval_per_day).max()
    
    # Compute RSI
    raw_data["rsi"] = ta.momentum.RSIIndicator(close=raw_data["close"]).rsi()

    # Compute MACD
    macd = ta.trend.MACD(close=raw_data["close"])
    raw_data["macd"] = macd.macd()
    raw_data["macd_signal"] = macd.macd_signal()

    # Compute Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=raw_data["close"])
    raw_data["bb_high"] = bollinger.bollinger_hband()
    raw_data["bb_low"] = bollinger.bollinger_lband()

    # Compute Moving Averages
    raw_data["sma_50"] = ta.trend.SMAIndicator(
        close=raw_data["close"], window=50
    ).sma_indicator()
    raw_data["sma_200"] = ta.trend.SMAIndicator(
        close=raw_data["close"], window=200
    ).sma_indicator()

    return raw_data

def time_diff(row, base_col, diff_col):
    if pd.isna(row[diff_col]):
        return np.nan
    else:
        return (row[diff_col] - row[base_col])

def compute_vwap(row, dv_col, volume_col, base_price=500):
    logging.error(f"volume_col:{volume_col}")
    if pd.isna(row[dv_col]) or pd.isna(row[volume_col]):
        return np.nan
    else:
        logging.error(f"row['cum_volume']:{row['cum_volume']}")
        logging.error(f"row['cum_dv']:{row['cum_dv']}")
        logging.error(f"row[{volume_col}]:{row[volume_col]}")
        logging.error(f"row[{dv_col}]:{row[dv_col]}")
        logging.error(f"row[close]:{row['close']}")
        if row["cum_volume"]>row[volume_col]:
            if row["cum_dv"] == row[dv_col]:
                logging.error(f"no cum_dv change")
                return 0
            else:
                vwap_price = (row["cum_dv"] - row[dv_col])/(row["cum_volume"]-row[volume_col])
                logging.error(f"vwap_price:{vwap_price}, close:{row['close']}")
                return np.log(row["close"]+base_price) - np.log(vwap_price+base_price)
        else:
            logging.error(f"cum volume does not change")
            return 0

def fill_dv(row, time_col, interval_minutes=30):
    if pd.isna(row[time_col]):
        return np.nan
    else:
        time_diff_threshold = interval_minutes*30+1
        if ((row[time_col] > row["timestamp"]-time_diff_threshold) and
            (row[time_col] < row["timestamp"]+time_diff_threshold)):
            return row["cum_dv"]
        else:
            return np.nan

def fill_volume(row, time_col, interval_minutes=30):
    if pd.isna(row[time_col]):
        return np.nan
    else:
        time_diff_threshold = interval_minutes*30+1
        if ((row[time_col] > row["timestamp"]-time_diff_threshold) and
            (row[time_col] < row["timestamp"]+time_diff_threshold)):
            return row["cum_volume"]
        else:
            return np.nan

@profile_util.profile
def add_group_features(raw_data: pd.DataFrame, interval_minutes, resort=True):
    for column in [
        "close_back",
        "volume_back",
        "dv_back",
        #"close_fwd",
        #"volume_fwd",
        #"dv_fwd",
        "cum_volume",
        "cum_dv",
        "close_back_cumsum",
        "volume_back_cumsum",
        "close_high_5_ff",
        "time_high_5_ff",
        "close_low_5_ff",
        "time_low_5_ff",
        "close_high_11_ff",
        "time_high_11_ff",
        "close_low_11_ff",
        "time_low_11_ff",
        "close_high_21_ff",
        "time_high_21_ff",
        "close_low_21_ff",
        "time_low_21_ff",
        "close_high_51_ff",
        "time_high_51_ff",
        "close_low_51_ff",
        "time_low_51_ff",
        "close_high_201_ff",
        "time_high_201_ff",
        "close_low_201_ff",
        "time_low_201_ff",
        "rsi",
        "macd",
        "macd_signal",
        "bb_high",
        "bb_low",
        "sma_50",
        "sma_200",
        "close_high_5_bf",
        "close_low_5_bf",
        "close_high_11_bf",
        "close_low_11_bf",
        "close_high_21_bf",
        "close_low_21_bf",
        "close_high_51_bf",
        "close_low_51_bf",
        "close_high_201_bf",
        "close_low_201_bf",
        'close_rolling_5d_max', 'close_rolling_5d_min', 'close_rolling_11d_max',
        'close_rolling_11d_min', 'close_rolling_21d_max',
        'close_rolling_21d_min', 'close_rolling_51d_max',
        'close_rolling_51d_min', 'close_rolling_201d_max',
        'close_rolling_201d_min'
    ]:
        if column in raw_data.columns:
            raw_data = raw_data.drop(columns=[column])
    #logging.info(f"raw_data:{raw_data.describe()}")
    new_features = raw_data.groupby(["ticker"], group_keys=False)[
        ["volume", "dv", "close", "timestamp"]
    ].apply(ticker_transform, interval_minutes=interval_minutes)
    new_features = new_features.drop(columns=["volume", "dv", "close", "timestamp"])
    raw_data = raw_data.join(new_features)

    raw_data["daily_returns"] = calc_returns(raw_data["close"])
    raw_data["daily_vol"] = calc_daily_vol(raw_data["daily_returns"])
    trend_combinations = [(8, 24), (16, 48), (32, 96)]
    for short_window, long_window in trend_combinations:
        raw_data[f"macd_{short_window}_{long_window}"] = MACDStrategy.calc_signal(
            raw_data["close"], short_window, long_window
        )

    return raw_data


@profile_util.profile
def add_example_level_features(raw_data: pd.DataFrame, cal, macro_data_builder, add_vwap=False):
    new_york_cal = mcal.get_calendar("NYSE")
    lse_cal = mcal.get_calendar("LSE")
    raw_data["week_of_year"] = raw_data["time"].apply(lambda x: x.isocalendar()[1])
    raw_data["month_of_year"] = raw_data["time"].apply(lambda x: x.month)

    lse_cal = mcal.get_calendar("LSE")
    raw_data["weekly_close_time"] = raw_data.timestamp.apply(
        market_time.compute_weekly_close_time, cal=cal
    )
    raw_data["monthly_close_time"] = raw_data.timestamp.apply(
        market_time.compute_monthly_close_time, cal=cal
    )
    raw_data["option_expiration_time"] = raw_data.timestamp.apply(
        market_time.compute_option_expiration_time, cal=cal
    )
    if macro_data_builder.add_macro_event:
        raw_data["last_macro_event_time"] = raw_data.timestamp.apply(
            market_time.compute_last_macro_event_time, cal=cal, mdb=macro_data_builder
        )
        raw_data["last_macro_event_cum_dv"] = raw_data.apply(fill_dv, time_col="last_macro_event_time", axis=1)
        raw_data["last_macro_event_cum_volume"] = raw_data.apply(fill_volume, time_col="last_macro_event_time", axis=1)
        raw_data["last_macro_event_cum_dv"] = raw_data.last_macro_event_cum_dv.ffill()
        raw_data["last_macro_event_cum_volume"] = raw_data.last_macro_event_cum_volume.ffill()
        raw_data["vwap_since_last_macro_event"] = raw_data.apply(compute_vwap, dv_col="last_macro_event_cum_dv",
                                                                 volume_col="last_macro_event_cum_volume", axis=1)
        raw_data["next_macro_event_time"] = raw_data.timestamp.apply(
            market_time.compute_next_macro_event_time, cal=cal, mdb=macro_data_builder
        )

    raw_data["new_york_open_time"] = raw_data.timestamp.apply(
        market_time.compute_open_time, cal=new_york_cal
    )
    raw_data["new_york_open_cum_dv"] = raw_data.apply(fill_dv, time_col="new_york_open_time", axis=1)
    raw_data["new_york_open_cum_volume"] = raw_data.apply(fill_volume, time_col="new_york_open_time", axis=1)
    raw_data["new_york_open_cum_dv"] = raw_data.new_york_open_cum_dv.ffill()
    raw_data["new_york_open_cum_volume"] = raw_data.new_york_open_cum_volume.ffill()
    raw_data["vwap_since_new_york_open"] = raw_data.apply(compute_vwap, dv_col="new_york_open_cum_dv",
                                                          volume_col="new_york_open_cum_volume", axis=1)
    raw_data["new_york_close_time"] = raw_data.timestamp.apply(
        market_time.compute_close_time, cal=new_york_cal
    )
    raw_data["london_open_time"] = raw_data.timestamp.apply(
        market_time.compute_open_time, cal=lse_cal
    )
    raw_data["london_open_cum_dv"] = raw_data.apply(fill_dv, time_col="london_open_time", axis=1)
    raw_data["london_open_cum_volume"] = raw_data.apply(fill_volume, time_col="london_open_time", axis=1)
    raw_data["london_open_cum_dv"] = raw_data.london_open_cum_dv.ffill()
    raw_data["london_open_cum_volume"] = raw_data.london_open_cum_volume.ffill()
    raw_data["vwap_since_london_open"] = raw_data.apply(compute_vwap, dv_col="london_open_cum_dv",
                                                        volume_col="london_open_cum_volume", axis=1)    
    raw_data["london_close_time"] = raw_data.timestamp.apply(
        market_time.compute_close_time, cal=lse_cal
    )
    raw_data["time_to_new_york_open"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="new_york_open_time"
    )
    raw_data["time_to_new_york_close"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="new_york_close_time"
    )
    raw_data["time_to_london_open"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="london_open_time"
    )
    raw_data["time_to_london_close"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="london_close_time"
    )
    raw_data["month"] = raw_data.time.dt.month  # categories have be strings
    raw_data["year"] = raw_data.time.dt.year  # categories have be strings
    raw_data["hour_of_day"] = raw_data.time.apply(lambda x: x.hour)
    raw_data["day_of_week"] = raw_data.time.apply(lambda x: x.dayofweek)
    raw_data["day_of_month"] = raw_data.time.apply(lambda x: x.day)
    # TODO: use business time instead of calendar time. this changes a lot
    # during new year.
    raw_data["time_to_weekly_close"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="weekly_close_time"
    )
    raw_data["time_to_monthly_close"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="monthly_close_time"
    )
    raw_data["time_to_option_expiration"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="option_expiration_time"
    )
    if macro_data_builder.add_macro_event:
        raw_data["time_to_last_macro_event"] = raw_data.apply(
            time_diff, axis=1, base_col="timestamp", diff_col="last_macro_event_time"
        )
        raw_data["time_to_next_macro_event"] = raw_data.apply(
            time_diff, axis=1, base_col="timestamp", diff_col="next_macro_event_time"
        )
    raw_data["time_to_high_5_ff"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="time_high_5_ff"
    )
    raw_data["time_to_low_5_ff"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="time_low_5_ff"
    )
    raw_data["time_to_high_11_ff"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="time_high_11_ff"
    )
    raw_data["time_to_low_11_ff"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="time_low_11_ff"
    )
    raw_data["time_to_high_21_ff"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="time_high_21_ff"
    )
    raw_data["time_to_low_21_ff"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="time_low_21_ff"
    )
    raw_data["time_to_high_51_ff"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="time_high_51_ff"
    )
    raw_data["time_to_low_51_ff"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="time_low_51_ff"
    )
    raw_data["time_to_high_201_ff"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="time_high_201_ff"
    )
    raw_data["time_to_low_201_ff"] = raw_data.apply(
        time_diff, axis=1, base_col="timestamp", diff_col="time_low_201_ff"
    )
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
        #df_pct_forward = df[["close", "volume", "dv"]].pct_change(periods=-1)
        #df = df.join(df_pct_back, rsuffix="_back").join(df_pct_forward, rsuffix="_fwd")
        df = df.join(df_pct_back, rsuffix="_back")
        df["month"] = df.time.dt.month  # categories have be strings
        df["year"] = df.time.dt.year  # categories have be strings
        df["hour_of_day"] = df.time.apply(lambda x: x.hour)
        df["day_of_week"] = df.time.apply(lambda x: x.dayofweek)
        df["day_of_month"] = df.time.apply(lambda x: x.day)
        df = df.dropna()
        return df
