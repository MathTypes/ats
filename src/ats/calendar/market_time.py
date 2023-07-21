import datetime
import functools
import logging

import pandas_market_calendars as mcal
import pytz

from ats.util import time_util


@functools.lru_cache(maxsize=None)
def get_macro_event_time(cal, x_date, mdb):
    schedule = cal.schedule(
        start_date=x_date, end_date=x_date + datetime.timedelta(days=35)
    )
    open_time = None
    for idx in range(len(schedule.market_open)):
        if mdb.has_event(schedule.market_open[idx]):
            open_time = schedule.market_open[idx]
            break
    return open_time.timestamp()


@functools.lru_cache(maxsize=None)
def get_open_time(cal, x_date):
    schedule = cal.schedule(
        start_date=x_date, end_date=x_date + datetime.timedelta(days=5)
    )
    return schedule.market_open[0].timestamp()


@functools.lru_cache(maxsize=None)
def get_close_time(cal, x_date):
    schedule = cal.schedule(
        start_date=x_date, end_date=x_date + datetime.timedelta(days=5)
    )
    return schedule.market_close[0].timestamp()


@functools.lru_cache(maxsize=None)
def get_weekly_close_time(cal, x_date):
    schedule = cal.schedule(
        start_date=x_date, end_date=x_date + datetime.timedelta(days=8)
    )
    curr_week_of_year = schedule.market_close[0].week_of_year
    curr_close_time = schedule.market_close[0].timestamp()
    for idx in range(len(schedule.market_close)):
        if schedule.market_close[idx].week_of_year != curr_week_of_year:
            break
        curr_close_time = schedule.market_close[idx].timestamp()
    return curr_close_time


@functools.lru_cache(maxsize=None)
def get_monthly_close_time(cal, x_date):
    schedule = cal.schedule(
        start_date=x_date, end_date=x_date + datetime.timedelta(days=38)
    )
    curr_month_of_year = schedule.market_close[0].month_of_year
    curr_close_time = schedule.market_close[0].timestamp()
    for idx in range(len(schedule.market_close)):
        if schedule.market_close[idx].month_of_year != curr_month_of_year:
            break
        curr_close_time = schedule.market_close[idx].timestamp()
    return curr_close_time


@functools.lru_cache(maxsize=None)
def get_option_expiration_time(cal, x_date):
    schedule = cal.schedule(
        start_date=x_date, end_date=x_date + datetime.timedelta(days=38)
    )
    curr_week_of_month = date_util.get_week_of_month(schedule.market_close[0])
    curr_close_time = schedule.market_close[0].timestamp()
    for idx in range(1, len(schedule.market_close), 3):
        if date_util.get_week_of_month(schedule.market_close[idx]) == 2:
            break
        curr_close_time = schedule.market_close[idx].timestamp()
    return get_weekly_close_time(cal, x_date)


def utc_to_nyse_time(utc_time, interval_minutes):
    utc_time = time_util.round_up(utc_time, interval_minutes)
    nyc_time = pytz.timezone("America/New_York").localize(
        datetime.datetime(
            utc_time.year, utc_time.month, utc_time.day, utc_time.hour, utc_time.minute
        )
    )
    # Do not use following.
    # See https://stackoverflow.com/questions/18541051/datetime-and-timezone-conversion-with-pytz-mind-blowing-behaviour
    # why datetime(..., tzinfo) does not work.
    # nyc_time = datetime.datetime(utc_time.year, utc_time.month, utc_time.day,
    #                             utc_time.hour, utc_time.minute,
    #                             tzinfo=pytz.timezone('America/New_York'))
    return nyc_time


def compute_macro_event_time(x, cal, mdb):
    # logging.info(f"x:{x}, {type(x)}")
    try:
        x = datetime.datetime.fromtimestamp(x)
        return int(get_macro_event_time(cal, x.date(), mdb))
    except Exception as e:
        logging.info(f"can not compute macro event for {x}, {e}")
        return x


def compute_open_time(x, cal):
    # logging.info(f"x:{x}, {type(x)}")
    try:
        x = datetime.datetime.fromtimestamp(x)
        return int(get_open_time(cal, x.date()))
    except Exception as e:
        logging.info(f"can not compute open for {x}, {e}")
        return x


def compute_weekly_close_time(x, cal):
    try:
        x = datetime.datetime.fromtimestamp(x)
        return int(get_close_time(cal, x.date()))
    except Exception as e:
        logging.info(f"can not compute open for {x}, {e}")
        return x


def compute_monthly_close_time(x, cal):
    try:
        x = datetime.datetime.fromtimestamp(x)
        return int(get_close_time(cal, x.date()))
    except Exception as e:
        logging.info(f"can not compute open for {x}, {e}")
        return x


def compute_option_expiration_time(x, cal):
    try:
        x = datetime.datetime.fromtimestamp(x)
        return int(get_close_time(cal, x.date()))
    except Exception as e:
        logging.info(f"can not compute open for {x}, {e}")
        return x


def compute_close_time(x, cal):
    # logging.info(f"x:{x}, {type(x)}")
    try:
        x = datetime.datetime.fromtimestamp(x)
        return int(get_close_time(cal, x.date()))
    except Exception as e:
        logging.info(f"can not compute open for {x}, {e}")
        return x


def compute_next_open_time(x, cal):
    try:
        x = datetime.datetime.fromtimestamp(x)
        x_date = x.date() + datetime.timedelta(days=1)
        return int(get_close_time(cal, x_date))
    except Exception as e:
        logging.info(f"can not compute open for {x}, {e}")
        return x


def get_next_trading_times(cal, interval, now, k):
    start_date = now.date()
    schedule = cal.schedule(
        start_date=start_date, end_date=start_date + datetime.timedelta(days=5)
    )
    time_range = mcal.date_range(schedule, frequency=interval)
    results = []
    for utc_time in time_range:
        nyc_time = utc_time.astimezone(pytz.timezone("America/New_York"))
        if nyc_time < now:
            continue
        results.append(nyc_time.timestamp())
        if len(results) >= k:
            break
    return results
