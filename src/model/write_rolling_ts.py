# Usage:
#   PYTHONPATH=.. python3 write_rolling_ts.py --ticker=ES --asset_type=FUT --start_date=2008-01-01 --end_date=2009-01-01
#
import datetime
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
from util import config_utils
from util import logging_utils
from util import time_util

from tsfresh.utilities.dataframe_functions import roll_time_series
from settings.default import PINNACLE_DATA_CUT, PINNACLE_DATA_FOLDER

def pull_sample_data(ticker: str, intraday: bool) -> pd.DataFrame:
    return pull_futures_sample_data(ticker)

def pull_futures_sample_data(ticker: str, asset_type: str, start_date, end_date, raw_dir) -> pd.DataFrame:
    #ticker = ticker.replace("CME_","")
    names = ["Time", "Open", "High", "Low", "Close", "Volume"]
    if asset_type in ["FUT"]:
        file_path = os.path.join(f"{raw_dir}/futures", f"{ticker}_1min_continuous_adjusted.txt")
    else:
        file_path = os.path.join(f"{raw_dir}/stock", f"{ticker}_full_1min_adjsplitdiv.txt")
    read_options = csv.ReadOptions(
               column_names=names,
               skip_rows=1)
    parse_options = csv.ParseOptions(delimiter=",")
    ds = ray.data.read_csv(file_path,
                           parse_options=parse_options, read_options=read_options)
    ds = ds.sort("Time")
    return ds

if __name__ == '__main__':
    parser = config_utils.get_arg_parser("Scrape tweet by id")
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--ticker", type=str)
    parser.add_argument("--asset_type", type=str)
    parser.add_argument(
        "--start_date",
        type=lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").date(),
        required=False,
        help="Set a start date",
    )
    parser.add_argument(
        "--end_date",
        type=lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").date(),
        required=False,
        help="Set a end date",
    )
    args = parser.parse_args()
    config_utils.set_args(args)
    logging_utils.init_logging()
    ticker = args.ticker
    asset_type = args.asset_type
    ray.init()  # Start or connect to Ray.
    enable_dask_on_ray()  # Enable the Ray scheduler backend for Dask.

    for cur_date in time_util.monthlist(args.start_date, args.end_date):
        for_date = cur_date[0]
        since = cur_date[0] + datetime.timedelta(days=-50)
        until = cur_date[1] + datetime.timedelta(days=5)
            
        ds = pull_futures_sample_data(ticker, asset_type, since, until, args.input_dir)
        df = ds.to_dask()
        df = df.set_index('Time')
        df = df[since:until]
        df = df.rename(columns = {"Volume":"volume", "Open":"open",
                                  "High":"high", "Low":"low", "Close":"close"})                      
        df["dv"] = df["close"]*df["volume"]
        df = df.resample('30Min').agg({'open': 'first', 
                                        'high': 'max', 
                                        'low': 'min', 
                                        'close': 'last',
                                        'volume': 'sum',
                                        "dv": 'sum'})
        df = df.compute()
        df = df.dropna()
        df["ticker"] = ticker
        df["time"] = df.index
        df["cum_volume"]  = df.volume.cumsum()
        df["cum_dv"]  = df.dv.cumsum()
        logging.info(f"df:{df.head()}")
        df_pct_back = df[["close", "volume", "dv"]].pct_change(periods=1)
        df_pct_forward = df[["close", "volume", "dv"]].pct_change(periods=-1)
        df = df.join(df_pct_back).join(df_pct_forward)
        logging.info(f"df:{df.head()}")
        #df = roll_time_series(df, column_id="ticker", column_sort="time")
        id_column = "ticker"
        sort_column = "time"
        df = roll_time_series(df,
                column_id=id_column,
                column_sort=sort_column,
                column_kind=None,
                rolling_direction=1,
                max_timeshift=13*(50*4+3),
                min_timeshift=13*(50*4+3))
        df["id"] = df["id"].astype(str)
        df["month"] = df.time.dt.month  # categories have be strings
        df["year"] = df.time.dt.year  # categories have be strings
        df["hour_of_day"] = df.time.apply(lambda x:x.hour)
        df["day_of_week"] = df.time.apply(lambda x:x.dayofweek)
        df["day_of_month"] = df.time.apply(lambda x:x.day)
        #df["week_of_month"] = df.time.apply(lambda x:x.isocalendar().week_of_month)
        df["week_of_year"] = df.time.apply(lambda x:x.isocalendar().week)
        #df["date"] = df.est_time
        df["time_idx"] = df.index
        df = df.dropna()
        logging.info(f"df:{df.describe()}")
        df = df[df.hour_of_day.isin([3]) & df.day_of_month.isin(0)]
        logging.info(f"df_schema:{df.info()}")
        logging.info(f"df:{df.head()}")
        file_path = os.path.join(args.output_dir, asset_type, "30min_rsp", ticker)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        ds = ray.data.from_pandas(df).repartition(100)
        ds.write_parquet(file_path + "/" + for_date.strftime("%Y%m%d"))
    ray.shutdown()