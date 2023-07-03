import datetime
import logging
import os
import pandas as pd
from pyarrow import csv
import ray
from util import time_util

def get_tick_data_with_ray(ticker: str, asset_type: str, start_date, end_date, raw_dir) -> pd.DataFrame:
    #ticker = ticker.replace("CME_","")
    names = ["Time", "Open", "High", "Low", "Close", "Volume"]
    if asset_type in ["FUT"]:
        file_path = os.path.join(f"{raw_dir}/futures", f"{ticker}_1min_continuous_adjusted.txt")
    else:
        file_path = os.path.join(f"{raw_dir}/stock", f"{ticker}_full_1min_adjsplitdiv.txt")
    read_options = csv.ReadOptions(column_names=names, skip_rows=1)
    parse_options = csv.ParseOptions(delimiter=",")
    ds = ray.data.read_csv(file_path,
                           parse_options=parse_options, read_options=read_options)
    ds = ds.sort("Time")
    return ds

def get_tick_data(ticker: str, asset_type: str, start_date, end_date, raw_dir) -> pd.DataFrame:
    #ticker = ticker.replace("CME_","")
    names = ["Time", "Open", "High", "Low", "Close", "Volume"]
    if asset_type in ["FUT"]:
        file_path = os.path.join(f"{raw_dir}/futures", f"{ticker}_1min_continuous_adjusted.txt")
    else:
        file_path = os.path.join(f"{raw_dir}/stock", f"{ticker}_full_1min_adjsplitdiv.txt")
    ds = pd.read_csv(file_path, delimiter=",", header=0, names=names)
    ds["Time"] = ds.Time.apply(lambda x:datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    ds.set_index("Time")
    logging.info(f"ds:{ds.head()}")
    logging.info(f"ds:{ds.info()}")
    return ds

def get_input_dirs(config, ticker, asset_type):
    base_dir = f"{config.dataset.base_dir}/{ticker}"
    input_dirs = []
    start_date = datetime.datetime.strptime(config.job.train_start_date,"%Y-%m-%d") + datetime.timedelta(days=-60)
    start_date = start_date.replace(day=1)
    end_date = datetime.datetime.strptime(config.job.test_end_date,"%Y-%m-%d")
    for cur_date in time_util.monthlist(start_date, end_date):
        for_date = cur_date[0]
        date_dir = os.path.join(base_dir, for_date.strftime("%Y%m%d"))
        files = os.listdir(date_dir)
        files = [date_dir+'/'+f for f in files if os.path.isfile(date_dir+'/'+f)] #Filtering only the files.
        input_dirs.extend(files)
    logging.info(f"reading files:{input_dirs}")
    return input_dirs

def get_processed_data(config, ticker: str, asset_type: str) -> pd.DataFrame:
    input_dirs = get_input_dirs(config, ticker, asset_type)
    ds = ray.data.read_parquet(input_dirs, parallelism=100)
    ds = ds.to_pandas(10000000)
    ds = ds.sort_index()
    ds = ds[~ds.index.duplicated(keep='first')]
    ds = ds[(ds.hour_of_day>4) & ds.hour_of_day<17]
    # Need to recompute close_back after filtering
    ds = ds.drop(columns=["close_back", "volume_back", "dv_back"])
    ds_pct_back = ds[["close", "volume", "dv"]].pct_change(periods=1)
    #df_pct_forward = df[["close", "volume", "dv"]].pct_change(periods=-1)
    ds = ds.join(ds_pct_back, rsuffix='_back')
    #.join(df_pct_forward, rsuffix='_fwd')
    logging.info(f"ds:{ds.head()}")
    exit(0)
    #logging.info(f"ds:{ds.info()}")
    return ds


class Preprocessor:
    def __init__(self, ticker, since, until):
        self.since = since
        self.until = until
        self.ticker = ticker

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index('Time')
        df = df[self.since:self.until]
        df = df.rename(columns = {"Volume":"volume", "Open":"open",
                                  "High":"high", "Low":"low", "Close":"close"})                      
        df["dv"] = df["close"]*df["volume"]
        df = df.resample('30Min').agg({'open': 'first', 
                                        'high': 'max', 
                                        'low': 'min', 
                                        'close': 'last',
                                        'volume': 'sum',
                                        "dv": 'sum'})
        df["ticker"] = self.ticker
        df["time"] = df.index
        df["cum_volume"]  = df.volume.cumsum()
        df["cum_dv"]  = df.dv.cumsum()
        logging.info(f"df:{df.head()}")
        df_pct_back = df[["close", "volume", "dv"]].pct_change(periods=1)
        df_pct_forward = df[["close", "volume", "dv"]].pct_change(periods=-1)
        df = df.join(df_pct_back, rsuffix='_back').join(df_pct_forward, rsuffix='_fwd')
        #df = roll_time_series(df, column_id="ticker", column_sort="time")
        df["month"] = df.time.dt.month  # categories have be strings
        df["year"] = df.time.dt.year  # categories have be strings
        df["hour_of_day"] = df.time.apply(lambda x:x.hour)
        df["day_of_week"] = df.time.apply(lambda x:x.dayofweek)
        df["day_of_month"] = df.time.apply(lambda x:x.day)
        #df["week_of_month"] = df.time.apply(lambda x:x.isocalendar().week_of_month)
        #df["week_of_year"] = df.time.apply(lambda x:x.isocalendar().week)
        #df["date"] = df.est_time
        logging.info(f"df:{df.head()}")
        logging.info(f"df:{df.describe()}")
        df = df.dropna()
        return df
