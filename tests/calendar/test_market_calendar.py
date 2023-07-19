import datetime
import logging
import unittest.mock as mock

import numpy as np
import pandas_market_calendars as mcal
import pytz
import torch

from ats.calendar import market_time
from ats.util import logging_utils

def test_next_trading_times_at_open():
    market_cal = mcal.get_calendar("NYSE")
    utc_time = datetime.datetime(2012, 1, 4, 14, 30).astimezone(pytz.timezone("America/New_York"))
    logging.info(f"utc_time:{utc_time}")
    trading_times = market_time.get_next_trading_times(market_cal, "30M", utc_time, 2)
    logging.error(f"trading_times:{trading_times}")
    assert trading_times[0] == 1325689200.0
    assert trading_times[1] == 1325691000.0

def test_next_trading_times_at_open_plus_30m():
    market_cal = mcal.get_calendar("NYSE")
    utc_time = datetime.datetime(2012, 1, 4, 15, 30).astimezone(pytz.timezone("America/New_York"))
    logging.info(f"utc_time:{utc_time}")
    trading_times = market_time.get_next_trading_times(market_cal, "30M", utc_time, 2)
    logging.error(f"trading_times:{trading_times}")
    assert trading_times[0] == 1325691000.0
    assert trading_times[1] == 1325692800.0

def test_next_trading_times_at_close_minus_30m():
    market_cal = mcal.get_calendar("NYSE")
    utc_time = datetime.datetime(2012, 1, 4, 21, 30).astimezone(pytz.timezone("America/New_York"))
    logging.info(f"utc_time:{utc_time}")
    trading_times = market_time.get_next_trading_times(market_cal, "30M", utc_time, 2)
    logging.error(f"trading_times:{trading_times}")
    assert trading_times[0] == 1325775600.0
    assert trading_times[1] == 1325777400.0

if __name__ == "__main__":
    logging_utils.init_logging()
