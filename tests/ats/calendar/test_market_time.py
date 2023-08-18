import datetime
import logging

import pandas_market_calendars as mcal
import pytz

from ats.calendar import market_time
from ats.util import logging_utils

def test_compute_last_open_time_nyse_on_good_friday():
    market_cal = mcal.get_calendar("NYSE")
    open_time = market_time.compute_last_open_time(datetime.datetime(2009, 4, 10, 12, 0, 0).timestamp(), market_cal)
    assert open_time == 1239283800
    open_time = market_time.compute_last_open_time(datetime.datetime(2009, 4, 10, 14, 0, 0).timestamp(), market_cal)
    assert open_time == 1239283800

def test_compute_open_time_nyse_on_good_friday():
    market_cal = mcal.get_calendar("NYSE")
    open_time = market_time.compute_open_time(datetime.datetime(2009, 4, 10, 12, 0, 0).timestamp(), market_cal)
    assert open_time == 1239629400
    open_time = market_time.compute_open_time(datetime.datetime(2009, 4, 10, 14, 0, 0).timestamp(), market_cal)
    assert open_time == 1239629400

# Test that caches are working properly
def test_compute_last_open_time_nyse_consecutive_call():
    market_cal = mcal.get_calendar("NYSE")
    open_time = market_time.compute_last_open_time(datetime.datetime(2023, 8, 3, 12, 0, 0).timestamp(), market_cal)
    assert open_time == 1690983000
    open_time = market_time.compute_last_open_time(datetime.datetime(2023, 8, 3, 14, 0, 0).timestamp(), market_cal)
    assert open_time == 1691069400

def test_compute_open_time_nyse_consecutive_call():
    market_cal = mcal.get_calendar("NYSE")
    open_time = market_time.compute_open_time(datetime.datetime(2023, 8, 3, 12, 0, 0).timestamp(), market_cal)
    assert open_time == 1691069400
    open_time = market_time.compute_open_time(datetime.datetime(2023, 8, 3, 14, 0, 0).timestamp(), market_cal)
    assert open_time == 1691069400
    open_time = market_time.compute_open_time(datetime.datetime(2023, 8, 3, 17, 0, 0).timestamp(), market_cal)
    assert open_time == 1691069400
    open_time = market_time.compute_open_time(datetime.datetime(2023, 8, 3, 20, 0, 0).timestamp(), market_cal)
    assert open_time == 1691069400
    open_time = market_time.compute_open_time(datetime.datetime(2023, 8, 3, 22, 0, 0).timestamp(), market_cal)
    assert open_time == 1691069400

def test_compute_open_time_cme_consecutive_call():
    market_cal = mcal.get_calendar("CME_Equity")
    open_time = market_time.compute_open_time(datetime.datetime(2023, 8, 3, 12, 0, 0).timestamp(), market_cal)
    assert open_time == 1691013600
    open_time = market_time.compute_open_time(datetime.datetime(2023, 8, 3, 14, 0, 0).timestamp(), market_cal)
    assert open_time == 1691013600
    open_time = market_time.compute_open_time(datetime.datetime(2023, 8, 3, 17, 0, 0).timestamp(), market_cal)
    assert open_time == 1691013600
    open_time = market_time.compute_open_time(datetime.datetime(2023, 8, 3, 20, 0, 0).timestamp(), market_cal)
    assert open_time == 1691013600
    open_time = market_time.compute_open_time(datetime.datetime(2023, 8, 3, 22, 0, 0).timestamp(), market_cal)
    assert open_time == 1691013600

def test_compute_last_open_time_nyse():
    market_cal = mcal.get_calendar("NYSE")
    open_time = market_time.compute_last_open_time(datetime.datetime(2009, 6, 1, 20, 0, 0).timestamp(), market_cal)
    # Mon Jun 01 2009 13:30:00
    assert open_time == 1243863000

def test_compute_last_open_time_nyse_before_open():
    market_cal = mcal.get_calendar("NYSE")
    open_time = market_time.compute_last_open_time(datetime.datetime(2009, 6, 1, 10, 0, 0).timestamp(), market_cal)
    # Fri May 29 2009 13:30:00
    assert open_time == 1243603800

def test_compute_open_time_cme():
    market_cal = mcal.get_calendar("CME_Equity")
    # 2009-06-01 is Monday
    # 2009-06-01 00:00:00+00:00
    open_time = market_time.get_open_time(market_cal, datetime.date(2009, 6, 1))
    # Sun May 31 2009 15:00:00
    assert open_time == 1243807200

def test_compute_weekly_close_time_mon_cme():
    market_cal = mcal.get_calendar("CME_Equity")
    # 2009-06-01 is Monday
    # 2009-06-01 00:00:00+00:00
    close_time = market_time.get_weekly_close_time(market_cal, datetime.date(2009, 6, 1))
    # Fri Jun 05 2009 14:00:00
    assert close_time == 1244235600.0

def test_compute_weekly_close_time_fri_before_close_cme():
    market_cal = mcal.get_calendar("CME_Equity")
    # 2009-06-01 is Monday
    # 2009-06-01 00:00:00+00:00
    close_time = market_time.get_weekly_close_time(market_cal, datetime.datetime(2009, 6, 5, 20, 0, 0))
    # Fri Jun 05 2009 14:00:00
    assert close_time == 1244235600.0

def test_compute_weekly_close_time_fri_after_close_cme():
    market_cal = mcal.get_calendar("CME_Equity")
    # 2009-06-01 is Monday
    # 2009-06-01 00:00:00+00:00
    # Note that we still use this week's Friday close instead of next week friday close.
    close_time = market_time.get_weekly_close_time(market_cal, datetime.datetime(2009, 6, 5, 21, 10, 0))
    # Fri Jun 05 2009 14:00:00
    assert close_time == 1244235600.0

def test_compute_weekly_close_time_sat_cme():
    market_cal = mcal.get_calendar("CME_Equity")
    # 2009-06-01 is Monday
    # 2009-06-01 00:00:00+00:00
    close_time = market_time.get_weekly_close_time(market_cal, datetime.date(2009, 6, 6))
    # Fri Jun 12 2009 14:00:00
    assert close_time == 1244840400

def test_compute_weekly_close_time_sun_cme():
    market_cal = mcal.get_calendar("CME_Equity")
    # 2009-06-01 is Monday
    # 2009-06-01 00:00:00+00:00
    close_time = market_time.get_weekly_close_time(market_cal, datetime.date(2009, 6, 7))
    # Fri Jun 12 2009 14:00:00
    assert close_time == 1244840400

def test_compute_close_time_cme():
    market_cal = mcal.get_calendar("CME_Equity")
    # 2009-06-01 is Monday
    # 2009-06-01 00:00:00+00:00
    close_time = market_time.get_close_time(market_cal, datetime.date(2009, 6, 1))
    # Mon Jun 01 2009 14:00:00
    assert close_time == 1243890000.0

def test_compute_open_time_lse():
    market_cal = mcal.get_calendar("LSE")
    open_time = market_time.get_open_time(market_cal, datetime.date(2023, 7, 21))
    # Fri Jul 21 2023 00:00:00
    assert open_time == 1689922800.0

def test_compute_close_time_lse():
    market_cal = mcal.get_calendar("LSE")
    close_time = market_time.get_close_time(market_cal,
                                            datetime.date(2023, 7, 21))
    # Fri Jul 21 2023 08:30:00
    assert close_time == 1689953400.0

def test_compute_open_time_nyse():
    market_cal = mcal.get_calendar("NYSE")
    open_time = market_time.get_open_time(market_cal, datetime.date(2023, 7, 21))
    # Fri Jul 21 2023 06:30:00
    assert open_time == 1689946200.0

def test_compute_close_time_nyse():
    market_cal = mcal.get_calendar("NYSE")
    close_time = market_time.get_close_time(market_cal,
                                            datetime.date(2023, 7, 21))
    # Fri Jul 21 2023 13:00:00
    assert close_time == 1689969600.0


def test_compute_monthly_close_time_start_nyse():
    market_cal = mcal.get_calendar("NYSE")
    # 2009-06-01 is Monday
    # 2009-06-01 00:00:00+00:00
    close_time = market_time.get_monthly_close_time(market_cal, datetime.date(2009, 6, 1))
    # Tue Jun 30 2009 13:00:00
    assert close_time == 1246392000.0

def test_compute_monthly_close_time_end_nyse():
    market_cal = mcal.get_calendar("NYSE")
    # 2009-06-01 is Monday
    # 2009-06-01 00:00:00+00:00
    close_time = market_time.get_monthly_close_time(market_cal, datetime.date(2009, 6, 30))
    # Tue Jun 30 2009 13:00:00
    assert close_time == 1246392000.0

def test_compute_option_expiration_time_before():
    market_cal = mcal.get_calendar("NYSE")
    close_time = market_time.get_option_expiration_time(market_cal, datetime.date(2023, 7, 1))
    # Fri Jul 21 2023 13:00:00
    assert close_time == 1689969600.0

def test_compute_option_expiration_time():
    market_cal = mcal.get_calendar("NYSE")
    close_time = market_time.get_option_expiration_time(market_cal, datetime.date(2023, 7, 21))
    # Fri Jul 21 2023 13:00:00
    assert close_time == 1689969600.0

def test_compute_option_expiration_time_after():
    market_cal = mcal.get_calendar("NYSE")
    close_time = market_time.get_option_expiration_time(market_cal, datetime.date(2023, 7, 23))
    # Fri Aug 18 2023 13:00:00
    assert close_time == 1692388800.0

