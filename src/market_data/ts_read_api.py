import logging
import os
import time
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
        market_df['assetCode'] = assetCode
        market_df = market_df.rename(columns={"date": "time"})
        market_df["time"] = pd.to_datetime(
            market_df["time"], infer_datetime_format=True)
        market_df = market_df.set_index("time")
        market_df = market_df.sort_index()
        return market_df
    except:
        logging.warn(
            f"can not open {instr_name} for {asof_date} at {file_path}")
    return None


def get_time_series(instr_name, from_date, end_date):
    df_vec = []
    logging.info(f'from_date:{from_date}, end_date:{end_date}')
    for cur_date in pd.date_range(start=from_date, end=end_date, freq="D"):
        file_path = os.path.join(FIVE_SECS_ROOT_DIR, instr_name + 'H3',
                                 cur_date.strftime("%Y%m%d") + '.csv')
        try:
            daily_market_df = pd.read_csv(file_path)
            df_vec.append(daily_market_df)
        except:
            logging.warn(
                f"can not open {instr_name} for {cur_date} at {file_path}")
    market_df = pd.concat(df_vec)
    market_df['assetName'] = instr_name
    market_df['assetCode'] = instr_name
    market_df = market_df.rename(columns={"date": "time"})
    market_df["time"] = pd.to_datetime(
        market_df["time"], infer_datetime_format=True)
    market_df = market_df.set_index("time")
    market_df = market_df.sort_index()
    return market_df
