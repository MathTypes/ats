# Usage:
#   PYTHONPATH=.. python3 write_parquet.py ES FUT
#
from math import log
from datetime import datetime, timezone, timedelta
import logging
import os
from typing import List
import sys

import numpy as np
import pandas as pd
from pyarrow import csv
import pytz
import pyarrow as pa
import ray
from ray.util.dask import enable_dask_on_ray

from settings.default import PINNACLE_DATA_CUT, PINNACLE_DATA_FOLDER

def pull_sample_data(ticker: str, intraday: bool) -> pd.DataFrame:
    return pull_futures_sample_data(ticker)

def pull_futures_sample_data(ticker: str, asset_type: str) -> pd.DataFrame:
    #ticker = ticker.replace("CME_","")
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
    ds = ray.data.read_csv(file_path,
                           parse_options=parse_options, read_options=read_options)
    ds = ds.sort("Time")
    return ds


asset_type = sys.argv[1]
ticker = sys.argv[2]
file_path = os.path.join("/Users/alex/git_repo/ats/src/model/data/token", asset_type, "30min", ticker)
if os.path.exists(file_path):
    exit(0)

ray.shutdown()
ray.init(_temp_dir=f"/tmp/results/xg_tree", ignore_reinit_error=True)
enable_dask_on_ray()  # Enable the Ray scheduler backend for Dask.

ds = pull_futures_sample_data(ticker, asset_type)
df = ds.to_dask()
df = df.set_index('Time')
df = df.resample('5Min').agg({'Open': 'first', 
                             'High': 'max', 
                             'Low': 'min', 
                             'Close': 'last',
                             'Volume': 'sum'})
df = df[df.Volume>0]
logging.info(f"df:{df.head()}")
orig_df = pd.DataFrame()
orig_df["Volume"] = df["Volume"]
orig_df["Time"] = df.index
orig_df = orig_df.set_index("Time")
df["tic"] = ticker
df = df.groupby('tic').apply(lambda x: x.assign(OpenPct=(x.Open.diff(1)/x.Open)*10000))
df = df.groupby('tic').apply(lambda x: x.assign(ClosePct=(x.Close.diff(1)/x.Close)*10000))
df = df.groupby('tic').apply(lambda x: x.assign(HighPct=(x.High.diff(1)/x.High)*10000))
df = df.groupby('tic').apply(lambda x: x.assign(LowPct=(x.Low.diff(1)/x.Low)*10000))
df = df.groupby('tic').apply(lambda x: x.assign(VolumePct=(x.Volume.diff(1)/x.Volume)))
logging.info(f"df after groupby:{df.head()}")
df = df.drop(columns="tic")
#df  = df.compute().droplevel('tic')
df = df.dropna()
logging.info(f"df after dropna:{df.head()}")
logging.info(f"df:{df.head()}")
for name in ["OpenPct", "HighPct", "LowPct", "ClosePct"]:
    df[name] = df[name].apply(lambda x : int(x))
logging.info(f"df after apply:{df.head()}")
df["VolumePct"] = df["VolumePct"].apply(lambda x : int(log(1+max(x,-0.9))*10))
logging.info(f"df before merge:{df.head()}")
df["Time"]=df.index
logging.info(f"orig_df:{orig_df.head()}")
#df = df.merge(orig_df, how="left", left_index=True, right_index=True)
logging.info(f"df_merge:{df.head()}")
#df.index = df.Time
logging.info(f"df:{df.head()}")
df = df.compute()
df = df.resample('30Min').agg(dict(
    OpenPct=lambda x: list(x),
    HighPct=lambda x: list(x),
    LowPct=lambda x: list(x),
    ClosePct=lambda x: list(x),
    VolumePct=lambda x: list(x),
    Open='first',
    High='max',
    Low='min',
    Close='last',
    Volume='sum'))
logging.info(f"df:{df.head()}")
#df = df[len(df.Volume)>0]
df.to_parquet(file_path, engine='fastparquet')
ray.shutdown()