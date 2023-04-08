import datetime
from functools import lru_cache
#from app_dir.update_market_data import update_market_data
import json
import logging
import requests
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import gc

from market_data import ts_read_api

pd.set_option("display.precision", 8)

def data_update_():
    #df_crypto = pd.read_csv('market_data/binance.txt')
    df_crypto = pd.read_csv('binance_us.txt')
    df_stocks = pd.read_csv('stocks.txt')
    df_indexes = pd.read_csv('indexes.txt')
    df_futures = pd.read_csv('futures.txt')
    df_forex = pd.read_csv('forex.txt')

    day_limit = 15

    if (((dt.datetime.now() - pd.to_datetime(df_crypto['Last Update'][0])).days >= day_limit) or 
        ((dt.datetime.now() - pd.to_datetime(df_stocks['Last Update'][0])).days >= day_limit) or 
        ((dt.datetime.now() - pd.to_datetime(df_indexes['Last Update'][0])).days >= day_limit) or 
        ((dt.datetime.now() - pd.to_datetime(df_futures['Last Update'][0])).days >= day_limit) or 
        ((dt.datetime.now() - pd.to_datetime(df_forex['Last Update'][0])).days >= day_limit)):
        update_market_data()

    gc.collect()
        
def date_utc(date_):
    date_ = pd.to_datetime(date_, utc = True)
    date_ = date_.dt.tz_localize(None)
    return date_
        
class Data_Sourcing:
    def __init__(self):
        pass

    def intervals(self, selected_interval):
        self.selected_interval = selected_interval
        self.period = None        
            
    def apis(self, asset):
        self.asset = asset
        #limit = 600
        limit = 30
        
        to_date = datetime.date.today()
        from_date = to_date - datetime.timedelta(days=limit)
        self.df = ts_read_api.get_time_series_from_monthly(asset, from_date, to_date)
        logging.info(f'duplicate data source index:{self.df[self.df.index.duplicated()]}')
        #logging.info(f'self.asset:{self.asset}, self.df.column: {self.df.columns}')
        #logging.info(f'self.asset:{self.asset}, market_data_df:{self.df}')
        #logging.info(f'duplicate index:{self.df.index.duplicated()}')
        #logging.info(f'duplicate index cnt:{self.df.index.duplicated().size}')
