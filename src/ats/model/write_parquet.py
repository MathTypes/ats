# Usage:
#   PYTHONPATH=.. python3 write_parquet.py ES FUT
#
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
    # ticker = ticker.replace("CME_","")
    names = ["Time", "Open", "High", "Low", "Close", "Volume"]
    if asset_type in ["FUT"]:
        file_path = os.path.join(
            "/Users/alex/git_repo/ats/src/model/futures",
            f"{ticker}_1min_continuous_adjusted.txt",
        )
    else:
        file_path = os.path.join(
            "/Users/alex/git_repo/ats/src/model/data/stock",
            f"{ticker}_full_1min_adjsplitdiv.txt",
        )
    read_options = csv.ReadOptions(column_names=names, skip_rows=1)
    parse_options = csv.ParseOptions(delimiter=",")
    ds = ray.data.read_csv(
        file_path, parse_options=parse_options, read_options=read_options
    )
    ds = ds.sort("Time")
    return ds


ticker = sys.argv[1]
asset_type = sys.argv[2]
ray.init()  # Start or connect to Ray.
enable_dask_on_ray()  # Enable the Ray scheduler backend for Dask.

ds = pull_futures_sample_data(ticker, asset_type)
df = ds.to_dask()
df = df.set_index("Time")
df = df.resample("30Min").agg(
    {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
)
logging.info(f"df_schema:{df.info()}")
logging.info(f"df_schema:{df.info()}")
file_path = os.path.join(
    "/Users/alex/git_repo/ats/src/model/data", asset_type, "30min", ticker
)
df.to_parquet(file_path, engine="fastparquet")
