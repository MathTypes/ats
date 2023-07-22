import datetime
import logging

import pandas_market_calendars as mcal
import pytz

from ats.calendar import market_time
from ats.util import logging_utils


def test_open_time_with_date():
    market_cal = mcal.get_calendar("CME_Equity")
    # 2009-06-01 is Monday
    # 2009-06-01 00:00:00+00:00
    train_start_date = datetime.datetime.strptime("2009-06-01", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    open_time = market_time.get_open_time(market_cal, train_start_date)
    # Sun May 31 2009 15:00:00
    assert open_time == 1243807200
    close_time = market_time.get_close_time(market_cal, train_start_date)
    # Mon Jun 01 2009 14:00:00
    assert close_time == 1243890000.0


def test_next_trading_times_intraday_cme_equity_30m():
    market_cal = mcal.get_calendar("CME_Equity")
    # Mon Jul 24 2023 05:30:00 PST
    utc_time = datetime.datetime(2023, 7, 24, 12, 30).astimezone(
        pytz.timezone("America/New_York")
    )
    logging.info(f"utc_time:{utc_time}")

    trading_times = market_time.get_next_trading_times(market_cal, "30M", utc_time, 2)
    assert trading_times == [1690201800.0, 1690203600.0]


def test_next_trading_times_near_friday_close_cme_equity_30m():
    market_cal = mcal.get_calendar("CME_Equity")
    # Fri Jul 21 2023 12:30:00
    utc_time = datetime.datetime(2023, 7, 21, 19, 30).astimezone(
        pytz.timezone("America/New_York")
    )
    logging.info(f"utc_time:{utc_time}")

    trading_times = market_time.get_next_trading_times(market_cal, "30M", utc_time, 10)
    # Fri Jul 21 2023 12:30, 13:00:00, 13:15, 14:00
    # Sunday 15:30, 16:00
    assert trading_times == [
        1689967800.0,
        1689969600.0,
        1689970500.0,
        1689973200.0,
        1690151400.0,
        1690153200.0,
        1690155000.0,
        1690156800.0,
        1690158600.0,
        1690160400.0,
    ]


def test_next_trading_times_near_monday_close_cme_equity_30m():
    market_cal = mcal.get_calendar("CME_Equity")
    # Fri Jul 21 2023 12:30:00
    utc_time = datetime.datetime(2023, 7, 24, 19, 30).astimezone(
        pytz.timezone("America/New_York")
    )
    logging.info(f"utc_time:{utc_time}")

    trading_times = market_time.get_next_trading_times(market_cal, "30M", utc_time, 10)
    # Mon Jul 24 2023 12:30:00, 13, 13:15
    # Monday 14:00, 15:30, 16:00
    assert trading_times == [
        1690227000.0,
        1690228800.0,
        1690229700.0,
        1690232400.0,
        1690237800.0,
        1690239600.0,
        1690241400.0,
        1690243200.0,
        1690245000.0,
        1690246800.0,
    ]


def test_next_trading_times_at_open_cme_equity_30m():
    market_cal = mcal.get_calendar("CME_Equity")
    # Sat
    utc_time = datetime.datetime(2023, 7, 22, 14, 30).astimezone(
        pytz.timezone("America/New_York")
    )
    logging.info(f"utc_time:{utc_time}")

    trading_times = market_time.get_next_trading_times(market_cal, "30M", utc_time, 2)
    # Sun Jul 23 2023 15:30:00
    assert trading_times[0] == 1690151400.0
    # Sun Jul 23 2023 16:00:00
    assert trading_times[1] == 1690153200.0


def test_next_trading_times_at_open_cme_equity_5m():
    market_cal = mcal.get_calendar("CME_Equity")
    # Sat
    utc_time = datetime.datetime(2023, 7, 22, 14, 30).astimezone(
        pytz.timezone("America/New_York")
    )
    logging.info(f"utc_time:{utc_time}")

    trading_times = market_time.get_next_trading_times(market_cal, "5M", utc_time, 2)
    # Sun Jul 23 2023 15:05:00, 15:10
    assert trading_times == [1690149900.0, 1690150200.0]


def test_next_trading_times_at_open():
    market_cal = mcal.get_calendar("NYSE")
    utc_time = datetime.datetime(2012, 1, 4, 14, 30).astimezone(
        pytz.timezone("America/New_York")
    )
    logging.info(f"utc_time:{utc_time}")
    trading_times = market_time.get_next_trading_times(market_cal, "30M", utc_time, 2)
    assert trading_times[0] == 1325689200.0
    assert trading_times[1] == 1325691000.0


def test_next_trading_times_at_open_plus_30m():
    market_cal = mcal.get_calendar("NYSE")
    utc_time = datetime.datetime(2012, 1, 4, 15, 30).astimezone(
        pytz.timezone("America/New_York")
    )
    logging.info(f"utc_time:{utc_time}")
    trading_times = market_time.get_next_trading_times(market_cal, "30M", utc_time, 2)
    assert trading_times[0] == 1325691000.0
    assert trading_times[1] == 1325692800.0


def test_next_trading_times_at_close_minus_30m():
    market_cal = mcal.get_calendar("NYSE")
    utc_time = datetime.datetime(2012, 1, 4, 21, 30).astimezone(
        pytz.timezone("America/New_York")
    )
    logging.info(f"utc_time:{utc_time}")
    trading_times = market_time.get_next_trading_times(market_cal, "30M", utc_time, 2)
    assert trading_times[0] == 1325775600.0
    assert trading_times[1] == 1325777400.0


if __name__ == "__main__":
    logging_utils.init_logging()
