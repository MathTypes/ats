import os
from datetime import datetime, timezone, timedelta
from typing import List

import pandas as pd
import logging
import ray
from pyarrow import csv
import yfinance as yf
import numpy as np
import pytz

from settings.default import PINNACLE_DATA_CUT, PINNACLE_DATA_FOLDER

def pull_sample_data(ticker: str, intraday: bool) -> pd.DataFrame:
    return pull_futures_sample_data(ticker)

def hloc(c):
    if c.name == "open":
        return c.iloc[0]
    if c.name == "close":
        return c.iloc[-1]
    if c.name == "high":
        return c.max()
    if c.name == "low":
        return c.min()
    if c.name == "volume":
        return c.sum()
    if c.name == "date":
        return c.iloc[0]
    return c    

def ceil_dt(tm):
    tm = tm - timedelta(minutes=tm.minute % 30,
                        seconds=tm.second,
                        microseconds=tm.microsecond)
    return tm

# UDF as a function on Pandas DataFrame batches.
def pandas_transform(df: pd.DataFrame) -> pd.DataFrame:
    eastern = pytz.timezone('US/Eastern')
    custom_date_parser = lambda x: x.tz_localize(eastern)
    df.Time = df.Time.apply(custom_date_parser)
    df["Date"] = df.Time.apply(lambda x : ceil_dt(x))
    return df

def pull_futures_sample_data(ticker: str, asset_type: str) -> pd.DataFrame:
    ticker = ticker.replace("CME_","")
    names = ["Time", "Open", "High", "Low", "Close", "Volume"]
    if asset_type in ["FUT"]:
        file_path = os.path.join("/Users/alex/git_repo/ats/src/model/futures",
                                f"{ticker}_1min_continuous_adjusted.txt")
    else:
        file_path = os.path.join("/Users/alex/git_repo/ats/src/model/data/stock",
                                f"{ticker}_full_1min_adjsplitdiv.txt")
    read_options = csv.ReadOptions(
               column_names=names,
               skip_rows=1)
    parse_options = csv.ParseOptions(delimiter=",")
    #convert_options = csv.ConvertOptions(timestamp_parsers=["%Y-%m-%d %H:%M:%S"])
    ds = ray.data.read_csv(file_path,
                           #convert_options=convert_options,
                           parse_options=parse_options, read_options=read_options)
    ds = ds.repartition(100).map_batches(pandas_transform, batch_size=1000)
    ds = ds.sort("Time")
    ds = ds.groupby("Date")
    ds = ds.map_groups(lambda g: g.apply(lambda c: [hloc(c)]))
    #ds = ds.add_column("ticker", lambda r: ticker)
    return ds

import sys
import ray
from ray.util.dask import enable_dask_on_ray
import pyarrow as pa
import pickle

ticker = sys.argv[1]
asset_type = sys.argv[2]
ray.init()  # Start or connect to Ray.
enable_dask_on_ray()  # Enable the Ray scheduler backend for Dask.
ds = pull_futures_sample_data(ticker, asset_type)
df = ds.to_dask()
df = df.drop(columns=["Time"])
file_path = os.path.join("/Users/alex/git_repo/ats/src/model/data", f"{ticker}_30min.pkl")
schema = pa.schema([pa.field('Date', pa.list_(pa.timestamp('ms'))),
                    pa.field('Open', pa.list_(pa.float64())),
                    pa.field('Close', pa.list_(pa.float64())),
                    pa.field('High', pa.list_(pa.float64())),
                    pa.field('Low', pa.list_(pa.float64())),
                    pa.field('Volume', pa.list_(pa.float64())),
                    ])
df.to_parquet(file_path, schema=schema)
#with open(file_path, 'wb') as handle:
#    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
#df.to_pickle()