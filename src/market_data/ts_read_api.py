import logging
import os
import time
import functools
import time

import traceback
import datetime
import pandas as pd

FIVE_SECS_ROOT_DIR = '../../data/FUT/5_secs'
ONE_MIN_ROOT_DIR = '../../data/FUT/1_min'


def get_time_series_by_instr_date(instr_name, asof_date):
    df_vec = []
    post_fix = ""
    if asof_date < datetime.date(2023, 3, 15):
        post_fix = 'H3'
    elif asof_date < datetime.date(2023, 6, 15):
        post_fix = 'M3'
    assetCode = instr_name + post_fix
    file_path = os.path.join(ONE_MIN_ROOT_DIR, assetCode,
                             asof_date.strftime("%Y%m%d") + '.csv')
    try:
        market_df = pd.read_csv(file_path)
        market_df['assetName'] = instr_name
        market_df['assetCode'] = instr_name
        market_df["time"] = pd.to_datetime(
            market_df["date"], format="%Y%m%d  %H:%M:%S").dt.tz_localize("UTC")
        market_df['idx_time'] = market_df["time"]
        market_df = market_df.set_index(["idx_time", "assetName"])
        market_df = market_df.rename(columns={'date':'Date', 'close': 'Adj Close', 'high': 'High',
                                              'low': 'Low', 'volume': 'Volume', 'open': 'Open'})
        market_df = market_df[['High', 'Low', 'Open',
                               'Volume', 'Adj Close']].apply(pd.to_numeric)
        market_df = market_df.sort_index()
        #market_df = market_df.set_index('Date')
        return market_df
    except Exception as e:
        logging.warn(
            f"can not open {instr_name} for {asof_date} at {file_path}, e:{e}")
    return None

@functools.lru_cache
def get_time_series_by_range(instr_name, from_date, end_date):
    df_vec = []
    #logging.info(f'instr_name:{instr_name}, from_date:{from_date}, end_date:{end_date}')
    for cur_date in pd.date_range(start=from_date, end=end_date, freq="D"):
        date_df = get_time_series_by_instr_date(instr_name, cur_date.date())
        df_vec.append(date_df)
    market_df = pd.concat(df_vec)
    #logging.info(f'market_time:{market_df}')
    #logging.info(f'total_market_shape:{market_df.shape}') 
    #logging.info(f'duplicate index:{market_df[market_df.index.duplicated()]}')
    #logging.info(f'duplicate_index_shape:{market_df[market_df.index.duplicated()].shape}')
    market_df = market_df.drop_duplicates(keep='last')
    #traceback.print_stack()
    #logging.info(f'deduped_total_market_shape:{market_df.shape}') 
    return market_df
