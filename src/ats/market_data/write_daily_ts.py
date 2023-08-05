# Usage:
#   PYTHONPATH=.. python3 write_daily_ts.py --ticker=ES --asset_type=FUT --start_date=2008-01-01 --end_date=2009-01-01 --input_dir=. --output_dir=data
#
import datetime
import glob
import logging
import os

#import modin.pandas as pd
import pandas as pd
from pyarrow import csv
import ray
from ray.util.dask import enable_dask_on_ray
from ray.data import ActorPoolStrategy
import re

from ats.util import config_utils
from ats.util import logging_utils
from ats.util import time_util


def get_id_time(id: str):
    result = re.match(r".*(\d\d\d\d\-\d\d\-\d\d\s+\d\d:\d\d:\d\d).*", str(id))
    if result:
        return datetime.datetime.strptime(result.group(1), "%Y-%m-%d %H:%M:%S")
    logging.info(f"can not extract id from {id}")
    return None


def pull_sample_data(ticker: str, intraday: bool) -> pd.DataFrame:
    return pull_futures_sample_data(ticker)


def pull_futures_sample_data(
    ticker: str, asset_type: str, start_date, end_date, raw_dir
) -> pd.DataFrame:
    # ticker = ticker.replace("CME_","")
    names = ["Time", "Open", "High", "Low", "Close", "Volume"]
    if asset_type == "FUT":
        file_path = os.path.join(
            f"{raw_dir}/{asset_type}", f"{ticker}_*1min_continuous_*.txt"
        )
    elif asset_type == "ETF":
        file_path = os.path.join(
            f"{raw_dir}/{asset_type}", f"{ticker}_*full_1min_*.txt"
        )
    logging.info(f"glob {file_path}")
    files = glob.glob(file_path)
    logging.info(f"read from {files}")
    read_options = csv.ReadOptions(column_names=names, skip_rows=1)
    parse_options = csv.ParseOptions(delimiter=",")
    ds = ray.data.read_csv(
        files, parse_options=parse_options, read_options=read_options
    )
    ds = ds.sort("Time")
    return ds


class Preprocessor:
    def __init__(self, ticker, orig_since, orig_until, since, until, freq):
        self.orig_since = orig_since
        self.orig_until = orig_until
        self.since = since
        self.until = until
        self.ticker = ticker
        self.freq = freq

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"df:{df}")
        df = df.set_index("Time")
        logging.info(f"since:{self.since}, until:{self.until}")
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
        df = df.resample(self.freq).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "dv": "sum",
            }
        )
        # df = df.compute()
        df["ticker"] = self.ticker
        df["time"] = df.index
        df["cum_volume"] = df.volume.cumsum()
        df["cum_dv"] = df.dv.cumsum()
        df = df.sort_index()
        # It is import to dropna before pct_change. The reason is that
        # we do not have open/high/low/close for Sat, but volume is 0.
        # If we do not drop them, they would cause Sun late trading to
        # have nan volume pct change and causes Sun to be dropped.
        df = df.dropna()
        logging.info(f"df:{df.head()}")
        df_pct_back = df[["close", "volume", "dv"]].pct_change(periods=1)
        df_pct_forward = df[["close", "volume", "dv"]].pct_change(periods=-1)
        df = df.join(df_pct_back, rsuffix="_back").join(df_pct_forward, rsuffix="_fwd")
        # df = roll_time_series(df, column_id="ticker", column_sort="time")
        # df = df.replace([np.inf, -np.inf], np.nan)
        df["month"] = df.time.dt.month  # categories have be strings
        df["year"] = df.time.dt.year  # categories have be strings
        df["hour_of_day"] = df.time.apply(lambda x: x.hour)
        df["day_of_week"] = df.time.apply(lambda x: x.dayofweek)
        df["day_of_month"] = df.time.apply(lambda x: x.day)
        df["time_idx"] = df.index
        logging.info(f"df:{df.head()}")
        logging.info(f"df:{df.describe()}")
        # df = df.dropna()
        df["series_idx"] = df.index
        return df


def process_daily(ds, cur_date, freq, force):
    for_date = cur_date[0]
    orig_since = cur_date
    orig_until = cur_date
    since = orig_since
    until = orig_until

    ds = ds.map_batches(
        Preprocessor,
        batch_size=40096000,
        compute=ActorPoolStrategy(min_size=1, max_size=1),
        fn_constructor_kwargs={
            "ticker": ticker,
            "orig_since": orig_since,
            "orig_until": orig_until,
            "since": since,
            "until": until,
            "freq": freq,
        },
    )
    # df = ds.to_dask()
    file_path = (
        os.path.join(args.output_dir, asset_type, freq, ticker)
        + "/"
        + for_date.strftime("%Y%m%d")
    )
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    else:
        if not force:
            logging.error(f"Directory {file_path} already exists, exiting!")
            return
    ds.write_parquet(file_path)


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    parser = config_utils.get_arg_parser("Scrape tweet by id")
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--force", type=bool)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--ticker", type=str)
    parser.add_argument("--asset_type", type=str)
    parser.add_argument("--freq", type=str)
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
    if ray.is_initialized():
        ray.shutdown()
    ray.init()  # Start or connect to Ray.
    enable_dask_on_ray()  # Enable the Ray scheduler backend for Dask.

    since = args.start_date
    until = args.end_date
    if not since:
        since = datetime.datetime.now().date() - datetime.timedelta(days=60)
    if not until:
        until = datetime.datetime.now().date()
    ds = pull_futures_sample_data(ticker, asset_type, since, until, args.input_dir)
    for cur_date in pd.date_range(start=since, end=until):
        process_daily(ds, cur_date, args.freq, args.force)
    ray.shutdown()
