import logging
import os
import modin.pandas as pd
from pyarrow import csv
import ray

def pull_futures_sample_data(ticker: str, asset_type: str, start_date, end_date, raw_dir) -> pd.DataFrame:
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
