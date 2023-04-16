import argparse
import datetime
import functools
import logging
import os
import time
import traceback

import pandas as pd

from util import config_utils
from util import logging_utils

from dateutil.parser import parse, ParserError


class DateParser(argparse.Action):
    def __call__(self, parser, namespace, values, option_strings=None):
        setattr(namespace, self.dest, parse(values).date())


def get_time_series_by_instr_date(instr_name, asof_date, time_period):
    df_vec = []
    post_fix = ""
    if asof_date < datetime.date(2023, 3, 15):
        post_fix = "H3"
    elif asof_date < datetime.date(2023, 6, 15):
        post_fix = "M3"
    assetCode = instr_name + post_fix
    file_path = os.path.join(
        config_utils.get_data_root(), "FUT", time_period, assetCode, asof_date.strftime("%Y%m%d") + ".csv"
    )
    try:
        market_df = pd.read_csv(file_path)
        logging.info(f"openning:{file_path}")
        market_df["assetName"] = instr_name
        market_df["assetCode"] = instr_name
        market_df["time"] = pd.to_datetime(
            market_df["date"], format="%Y%m%d  %H:%M:%S"
        ).dt.tz_localize("UTC")
        market_df["idx_time"] = market_df["time"]
        market_df = market_df.rename(
            columns={
                "date": "Date",
                "close": "Adj Close",
                "high": "High",
                "low": "Low",
                "volume": "Volume",
                "open": "Open",
            }
        )
        return market_df
    except Exception as e:
        logging.warn(f"can not open {instr_name} for {asof_date} at {file_path}, e:{e}")
    return None


# @functools.lru_cache(maxsize=64)
def get_time_series_from_monthly(instr_name, from_date, end_date, time_period):
    df_vec = []
    for month in pd.period_range(from_date, end_date, freq="M"):
        logging.info(f"month:{month}")
        path_dir = os.path.join(config_utils.get_ts_root(), "monthly", time_period, instr_name)
        month_file = os.path.join(path_dir, month.strftime("%Y%m") + ".parquet")
        logging.info(f"reading:{month_file}")
        df_vec.append(pd.read_parquet(month_file))
    df = pd.concat(df_vec)
    logging.info(f"reading_df:{df}")
    logging.info(f"df_columns:{df.columns}")
    # df = df.set_index(["idx_time"])
    df = df.sort_index()
    logging.info(f"duplicate index:{df[df.index.duplicated()]}")
    df = df[from_date:end_date]
    df["time"] = df.index
    # df.index = df.index.astype('str')
    df.drop_duplicates(subset=None, keep="first", inplace=True)
    # TODO(jeremy): Replace following line with proper duplicate. We should
    # keep one instead of dropping them all
    df = df[~df.index.duplicated()]
    return df


# @functools.lru_cache(maxsize=64)
def get_time_series_by_range(instr_name, from_date, end_date, time_period):
    df_vec = []
    # logging.info(f'instr_name:{instr_name}, from_date:{from_date}, end_date:{end_date}')
    for cur_date in pd.date_range(
        start=from_date - datetime.timedelta(days=2),
        end=end_date + datetime.timedelta(days=2),
        freq="D",
    ):
        date_df = get_time_series_by_instr_date(instr_name, cur_date.date(), time_period)
        df_vec.append(date_df)
    market_df = pd.concat(df_vec)
    market_df = market_df.set_index(["idx_time"])
    market_df = market_df.sort_index()
    market_df = market_df[["High", "Low", "Open", "Volume", "Adj Close"]].apply(
        pd.to_numeric
    )
    # logging.info(f'market_time:{market_df}')
    # logging.info(f'total_market_shape:{market_df.shape}')
    # logging.info(f'duplicate index:{market_df[market_df.index.duplicated()]}')
    # logging.info(f'duplicate_index_shape:{market_df[market_df.index.duplicated()].shape}')
    market_df = market_df.drop_duplicates(keep="last")
    # traceback.print_stack()
    # logging.info(f'deduped_total_market_shape:{market_df.shape}')
    return market_df[from_date:end_date]


if __name__ == "__main__":
    parser = config_utils.get_arg_parser("Preprocess tweet")
    parser.add_argument("--instr", type=str)
    parser.add_argument(
        "--start_date",
        type=lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").date(),
        required=True,
        help="Set a start date",
    )
    parser.add_argument(
        "--end_date",
        type=lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").date(),
        required=True,
        help="Set a end date",
    )

    args = parser.parse_args()
    config_utils.set_args(args)
    logging_utils.init_logging()
    price_df = get_time_series_by_range(args.instr, args.start_date, args.end_date)
    logging.info(f"price_df_time:{price_df.index}")
