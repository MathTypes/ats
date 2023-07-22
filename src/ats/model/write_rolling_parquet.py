# Usage:
#   PYTHONPATH=.. python3 write_parquet.py ES FUT
#
import pytz
import logging
import os
import sys

import pandas as pd
from pyarrow import csv
import pytz
import ray
from ray.util.dask import enable_dask_on_ray

from tsfresh.utilities.dataframe_functions import (
    roll_time_series,
)
from util import logging_utils


def pull_sample_data(ticker: str, intraday: bool) -> pd.DataFrame:
    return pull_futures_sample_data(ticker)


def pull_futures_sample_data(
    raw_dir: str, ticker: str, asset_type: str
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


logging_utils.init_logging()
logging.info(f"sys:{sys.argv}")
raw_dir = sys.argv[1]
asset_type = sys.argv[2]
ticker = sys.argv[3]
logging.info(f"raw_dir:{raw_dir}")
logging.info(f"asset_type:{asset_type}")
logging.info(f"ticker:{ticker}")
ray.init()  # Start or connect to Ray.
enable_dask_on_ray()  # Enable the Ray scheduler backend for Dask.

ds = pull_futures_sample_data(raw_dir, ticker, asset_type)
df = ds.to_dask()
df = df.rename(
    columns={
        "Close": "close",
        "Volume": "volume",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Time": "Time",
    }
)
df["dv"] = df["close"] * df["volume"]
newYorkTz = pytz.timezone("America/New_York")
logging.info(f"df:{df.head()}")
logging.info(f"df:{df.info()}")
logging.info(f"df:{df.describe()}")
df["Time"] = df["Time"].apply(
    lambda x: x.to_pydatetime()
    .replace(tzinfo=newYorkTz)
    .astimezone(pytz.timezone("UTC"))
)
df = df.set_index("Time")
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
df["time"] = df.index
logging.info(f"df:{df.head()}")
logging.info(f"df:{df.info()}")
logging.info(f"df:{df.describe()}")
df["cum_volume"] = df.volume.cumsum()
df["cum_dv"] = df.dv.cumsum()
df["ticker"] = ticker
id_column = "ticker"
sort_column = "time"
rolled_backward = roll_time_series(
    df,
    column_id=id_column,
    column_sort=sort_column,
    column_kind=None,
    rolling_direction=1,
    max_timeshift=13 * (50 * 4 + 3),
    min_timeshift=13 * (50 * 4 + 3),
)
# logging.info(f"df_schema:{df.info()}")
file_path = os.path.join(raw_dir, asset_type, "30min_rolling", ticker)
rolled_backward.to_parquet(file_path, engine="fastparquet")
