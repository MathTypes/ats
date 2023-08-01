import datetime
import functools
import logging

import pandas_market_calendars as mcal
import pytz

from ats.calendar import date_util
from ats.util import time_util
from ats.util import profile_util

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
    curr_close_time = datetime.datetime.utcfromtimestamp(schedule.market_close[0].timestamp())
    curr_week_of_year = curr_close_time.isocalendar()[1]
    for idx in range(len(schedule.market_close)):
        close_time = datetime.datetime.utcfromtimestamp(schedule.market_close[idx].timestamp())
        if close_time.isocalendar()[1] != curr_week_of_year:
            break
        curr_close_time = close_time
    return curr_close_time.timestamp()


@functools.lru_cache(maxsize=None)
def get_monthly_close_time(cal, x_date):
    schedule = cal.schedule(
        start_date=x_date, end_date=x_date + datetime.timedelta(days=38)
    )
    curr_close_time = datetime.datetime.utcfromtimestamp(schedule.market_close[0].timestamp())
    curr_month = curr_close_time.month
    for idx in range(len(schedule.market_close)):
        close_time = datetime.datetime.utcfromtimestamp(schedule.market_close[idx].timestamp())
        if close_time.month != curr_month:
            break
        curr_close_time = close_time
    return curr_close_time.timestamp()


@functools.lru_cache(maxsize=None)
def get_option_expiration_time(cal, x_date):
    x_date = date_util.get_option_expiration_day(x_date)
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


# TODO: currently we take date from cal. That means if cal is weekly close
# date, it will use that date instead of jumping to next week.

def compute_weekly_close_time(x, cal):
    try:
        x = datetime.datetime.fromtimestamp(x)
        return int(get_weekly_close_time(cal, x.date()))
    except Exception as e:
        logging.error(f"can not compute weekly event for {x}, {e}")
        return None

def compute_monthly_close_time(x, cal):
    try:
        x = datetime.datetime.fromtimestamp(x)
        return int(get_monthly_close_time(cal, x.date()))
    except Exception as e:
        logging.error(f"can not compute monthly close for {x}, {e}")
        return None

def compute_macro_event_time(x, cal, mdb):
    # logging.info(f"x:{x}, {type(x)}")
    try:
        x = datetime.datetime.fromtimestamp(x)
        return int(get_macro_event_time(cal, x.date(), mdb))
    except Exception as e:
        logging.error(f"can not compute macro event for {x}, {e}")
        return None

def compute_option_expiration_time(x, cal):
    try:
        x = datetime.datetime.fromtimestamp(x)
        return int(get_option_expiration_time(cal, x.date()))
    except Exception as e:
        logging.error(f"can not compute option expiration for {x}, {e}")
        return None

def compute_open_time(x, cal):
    # logging.info(f"x:{x}, {type(x)}")
    try:
        x = datetime.datetime.fromtimestamp(x)
        return int(get_open_time(cal, x.date()))
    except Exception as e:
        logging.error(f"can not compute open for {x}, {e}")
        return None

def compute_close_time(x, cal):
    # logging.info(f"x:{x}, {type(x)}")
    try:
        x = datetime.datetime.fromtimestamp(x)
        return int(get_close_time(cal, x.date()))
    except Exception as e:
        logging.error(f"can not compute open for {x}, {e}")
        return None

def compute_next_open_time(x, cal):
    try:
        x = datetime.datetime.fromtimestamp(x)
        x_date = x.date() + datetime.timedelta(days=1)
        return int(get_close_time(cal, x_date))
    except Exception as e:
        logging.error(f"can not compute open for {x}, {e}")
        return None

def get_next_trading_times(cal, interval, now, k):
    start_date = now.date()
    schedule = cal.schedule(
        start_date=start_date, end_date=start_date + datetime.timedelta(days=5)
    )
    time_range = mcal.date_range(schedule, frequency=f"{interval}min")
    results = []
    last_time = None
    for utc_time in time_range:
        nyc_time = utc_time.astimezone(pytz.timezone("America/New_York"))
        if nyc_time < now:
            continue
        # There are cases where market stops around 16:15 EST and it will be
        # included in schedule. We would like to skip it.
        if last_time and nyc_time < last_time + datetime.timedelta(minutes=interval):
            continue
        last_time = nyc_time
        results.append(nyc_time.timestamp())
        if len(results) >= k:
            break
    return results
