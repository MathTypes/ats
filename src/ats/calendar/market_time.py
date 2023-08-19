import datetime
import functools
import logging

import pandas_market_calendars as mcal
import pytz

from ats.calendar import date_util
from ats.util import time_util
from ats.util import profile_util

@functools.lru_cache(maxsize=128000)
def get_last_macro_event_time(cal, x_time, mdb, imp):
    events = mdb.get_last_events(x_time, imp)
    for ix, val in events.event_time.iloc[::-1].items():
        if val<=x_time:
            return val
    return None
    
@functools.lru_cache(maxsize=128000)
def get_next_macro_event_time(cal, x_time, mdb, imp):
    events = mdb.get_next_events(x_time, imp)
    for	et in events.event_time:
        if et>=x_time:
            return et
    return None

_OPEN_TIME_DICT = {}
_CLOSE_TIME_DICT = {}

def get_open_time(cal, x_date):
    if not cal in _OPEN_TIME_DICT.keys():
        _OPEN_TIME_DICT[cal] = {}
    open_time_dict = _OPEN_TIME_DICT[cal]
    if x_date in open_time_dict.keys():
        #logging.error(f"returning {x_date}, {open_time_dict[x_date]}")
        return open_time_dict[x_date]    
    #logging.error(f"x_date:{x_date}")
    schedule = cal.schedule(
        start_date=x_date, end_date=x_date + datetime.timedelta(days=30)
    )
    #logging.error(f"schedule:{schedule}")
    for idx in range(len(schedule.market_open)):
        # Use close in case we do not use prior date when open starts in prior
        # date for global futures
        x_date_idx = schedule.market_close[idx].date()
        #logging.error(f"adding {x_date_idx} with {schedule.market_open[idx].timestamp()}, idx:{idx}")
        open_time_dict[x_date_idx] = schedule.market_open[idx].timestamp()
    if x_date in open_time_dict.keys():
        return open_time_dict[x_date]
    else:
        return None

def get_close_time(cal, x_date):
    if not cal in _CLOSE_TIME_DICT.keys():
        _CLOSE_TIME_DICT[cal] = {}
    close_time_dict = _CLOSE_TIME_DICT[cal]
    if x_date in close_time_dict.keys():
        return close_time_dict[x_date]
    
    schedule = cal.schedule(
        start_date=x_date-datetime.timedelta(days=4),
        end_date=x_date + datetime.timedelta(days=30)
    )
    for idx in range(len(schedule.market_close)):
        x_date_idx = schedule.market_close[idx].date()
        close_time_dict[x_date_idx] = schedule.market_close[idx].timestamp()
    if x_date in close_time_dict.keys():
        return close_time_dict[x_date]
    else:
        return None


@functools.lru_cache(maxsize=128000)
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


@functools.lru_cache(maxsize=128000)
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


@functools.lru_cache(maxsize=128000)
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

def compute_last_macro_event_time(x, cal, mdb, imp):
    try:
        event_time = get_last_macro_event_time(cal, x, mdb, imp)
        if event_time is None:
            return None
        last_macro_event = int(event_time)
        return last_macro_event
    except Exception as e:
        logging.error(f"can not compute macro event for {x}, {e}")
        return None

def compute_next_macro_event_time(x, cal, mdb, imp):
    try:
        event_time = get_next_macro_event_time(cal, x, mdb, imp)
        if event_time is None:
            return None
        next_macro_event = int(event_time)
        return next_macro_event
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

def compute_last_open_time(x, cal):
    try:
        #logging.error(f"x:{x}, cal:{cal}")
        x_time = datetime.datetime.fromtimestamp(x)
        open_time = get_open_time(cal, x_time.date())
        while open_time is None or open_time > x:
            x_time = x_time + datetime.timedelta(days=-1)
            open_time = get_open_time(cal, x_time.date())
        return open_time
    except Exception as e:
        logging.error(f"can not compute_last_open for {x}, {e}")
        return None

def compute_last_close_time(x, cal):
    try:
        x_time = datetime.datetime.fromtimestamp(x)
        close_time = get_close_time(cal, x_time.date())
        while close_time is None or close_time > x:
            x_time = x_time + datetime.timedelta(days=-1)
            close_time = get_close_time(cal, x_time.date())
        return close_time
    except Exception as e:
        logging.error(f"can not compute_last_close for {x}, {e}")
        return None

def compute_next_open_time(x, cal):
    try:
        #logging.error(f"x:{x}, cal:{cal}")
        x_time = datetime.datetime.fromtimestamp(x)
        open_time = get_open_time(cal, x_time.date())
        while open_time is None or open_time < x:
            x_time = x_time + datetime.timedelta(days=1)
            open_time = get_open_time(cal, x_time.date())
        return open_time
    except Exception as e:
        logging.error(f"can not compute_next_open for {x}, {e}")
        return None

def compute_next_close_time(x, cal):
    try:
        x_time = datetime.datetime.fromtimestamp(x)
        x_date = x_time.date()
        close_time = get_close_time(cal, x_time.date())
        while close_time is None or close_time < x:
            x_time = x_time + datetime.timedelta(days=1)
            close_time = get_close_time(cal, x_time.date())
        return close_time
    except Exception as e:
        logging.error(f"can not compute_next_close for {x}, {e}")
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
