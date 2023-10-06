import datetime
import functools
import logging
from datetime import timezone
import pandas_market_calendars as mcal
import pytz #its deprecated tho --- however datetime doesnt seem to be doing anything to it.

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
    for	ix, val in events.event_time.items():
        if val>=x_time:
            return val
    return None

_OPEN_TIME_DICT = {}
_CLOSE_TIME_DICT = {}

@functools.lru_cache(maxsize=128000)
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
    #logging.error(f"{open_time_dict}")
    #logging.error(f"x_date:{x_date}")
    if x_date in open_time_dict.keys():
        return open_time_dict[x_date]
    else:
        return None

@functools.lru_cache(maxsize=128000)
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

def compute_next_weekly_close_time(x, cal):
    try:
        x_time = datetime.datetime.fromtimestamp(x, tz=timezone.utc)
        close_time = get_weekly_close_time(cal, x_time.date())
        if close_time < x:
            close_time = get_weekly_close_time(cal, x_time.date() + datetime.timedelta(days=1))            
        return int(close_time)
    except Exception as e:
        logging.error(f"can not compute weekly event for {x}, {e}")
        return None

def compute_next_monthly_close_time(x, cal):
    try:
        x_time = datetime.datetime.fromtimestamp(x, tz=timezone.utc)
        close_time = get_monthly_close_time(cal, x_time.date())
        if close_time < x:
            close_time = get_monthly_close_time(cal, x_time.date() + datetime.timedelta(days=1))            
        return int(close_time)
    except Exception as e:
        logging.error(f"can not compute monthly event for {x}, {e}")
        return None
    
def compute_last_weekly_close_time(x, cal, k=0):
    try:
        x_time = datetime.datetime.fromtimestamp(x, tz=timezone.utc)
        close_time = get_weekly_close_time(cal, x_time.date())
        while close_time is None or close_time > x:
            x_time = x_time + datetime.timedelta(days=-1)
            close_time = get_weekly_close_time(cal, x_time.date())
        if k==0:
            return int(close_time)
        else:
            return compute_last_weekly_close_time(close_time-10, cal, k-1)
    except Exception as e:
        logging.error(f"can not compute weekly event for {x}, {e}")
        return None

def compute_last_monthly_close_time(x, cal, k=0):
    try:
        x_time = datetime.datetime.fromtimestamp(x, tz=timezone.utc)
        close_time = get_monthly_close_time(cal, x_time.date())
        while close_time is None or close_time > x:
            x_time = x_time + datetime.timedelta(days=-1)
            close_time = get_monthly_close_time(cal, x_time.date())
        #logging.error(f"x:{x}, cal:{cal}, last_monthly_close:{close_time}")
        if k==0:
            return int(close_time)
        else:
            return compute_last_monthly_close_time(close_time-10, cal, k-1)
    except Exception as e:
        logging.error(f"can not compute last monthly event for {x}, {e}")
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
        return int(last_macro_event)
    except Exception as e:
        logging.error(f"can not compute macro event for {x}, {e}")
        return None

def compute_next_macro_event_time(x, cal, mdb, imp):
    try:
        event_time = get_next_macro_event_time(cal, x, mdb, imp)
        if event_time is None:
            return None
        next_macro_event = int(event_time)
        return int(next_macro_event)
    except Exception as e:
        logging.error(f"can not compute macro event for {x}, {e}")
        return None

def compute_last_option_expiration_time(x, cal):
    try:
        x_time = datetime.datetime.fromtimestamp(x, tz=timezone.utc)
        close_time = get_option_expiration_time(cal, x_time.date())
        while close_time is None or close_time > x:
            x_time = x_time + datetime.timedelta(days=-1)
            close_time = get_option_expiration_time(cal, x_time.date())
        return int(close_time)
    except Exception as e:
        logging.error(f"can not compute option expiration for {x}, {e}")
        return None

def compute_option_expiration_time(x, cal):
    try:
        x = datetime.datetime.fromtimestamp(x)
        return int(get_option_expiration_time(cal, x.date()))
    except Exception as e:
        logging.error(f"can not compute option expiration for {x}, {e}")
        return None

def compute_last_open_time(x, cal, k=0):
    try:
        #logging.error(f"x:{x}, cal:{cal}")
        x_time = datetime.datetime.fromtimestamp(x, tz=timezone.utc)
        open_time = get_open_time(cal, x_time.date())
        while open_time is None or open_time > x:
            x_time = x_time + datetime.timedelta(days=-1)
            open_time = get_open_time(cal, x_time.date())
        if k==0:
            return int(open_time)
        else:
            return compute_last_open_time(open_time-10, cal, k-1)
    except Exception as e:
        logging.error(f"can not compute_last_open for {x}, {e}")
        return None

def compute_last_close_time(x, cal, k=0):
    try:
        x_time = datetime.datetime.fromtimestamp(x, tz=timezone.utc)
        close_time = get_close_time(cal, x_time.date())
        while close_time is None or close_time > x:
            x_time = x_time + datetime.timedelta(days=-1)
            close_time = get_close_time(cal, x_time.date())
            #logging.error(f"close_time:{close_time}, x:{x}")
        if k==0:
            return int(close_time)
        else:
            return compute_last_close_time(close_time-10, cal, k-1)
    except Exception as e:
        logging.error(f"can not compute_last_close for {x}, {e}")
        return None

def compute_next_open_time(x, cal, k=0):
    try:
        #logging.error(f"x:{x}, cal:{cal}")
        x_time = datetime.datetime.fromtimestamp(x)
        open_time = get_open_time(cal, x_time.date())
        while open_time is None or open_time < x:
            x_time = x_time + datetime.timedelta(days=1)
            open_time = get_open_time(cal, x_time.date())
        if k==0:
            return int(open_time)
        else:
            return compute_next_open_time(open_time+10, cal, k-1)
    except Exception as e:
        logging.error(f"can not compute_next_open for {x}, {e}")
        return None

def compute_next_close_time(x, cal, k=0):
    try:
        x_time = datetime.datetime.fromtimestamp(x)
        x_date = x_time.date()
        close_time = get_close_time(cal, x_time.date())
        while close_time is None or close_time < x:
            x_time = x_time + datetime.timedelta(days=1)
            close_time = get_close_time(cal, x_time.date())
        if k==0:
            return int(close_time)
        else:
            return compute_next_close_time(close_time+10, cal, k-1)
    except Exception as e:
        logging.error(f"can not compute_next_close for {x}, {e}")
        return None


def get_next_trading_times(cal, interval, nynow, k): #get next k trading times based of calendar
    #now is in nytime
    #all tests pass in interval --- which is kind of useless, but I guess I'll keep it.
    #all tests pass in ny time, which I conver to utc.
    now=nynow.astimezone(pytz.timezone('UTC')) #utc now
    

    ######start of code#########
    start_date=now.date()
    schedule = cal.schedule(
        start_date=start_date, end_date=start_date + datetime.timedelta(days=k*7)
    ) #dataframe type

    # print(type(schedule)) #dataframe. Iterate down market_open.

    # print("schedule begins here:")
    res=[]
    count=0
    for row in schedule.values: #column by column of dataframe
        start_time=row[0]

        if start_time < now:
            continue
        
        res.append(row[0].timestamp())
        count+=1

        if count == k:
            break
    
    return res

'''
error: failed to push some refs to 'https://github.com/MathTypes/ats.git'
hint: Updates were rejected because the remote contains work that you do
hint: not have locally. This is usually caused by another repository pushing
hint: to the same ref. You may want to first integrate the remote changes
hint: (e.g., 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details
'''



# def get_next_trading_times(cal, interval, now, k): 
#     #task: get next k trading times from "now"
#     #( calendar object, 30, datetime object IN NY TIME, how many opening times to return)
#     start_date = now.date() #NY time starting date (date object)


#     schedule = cal.schedule(
#         start_date=start_date, end_date=start_date + datetime.timedelta(days=5)
#     ) #get the nyse/cme's trading hours for these days (dateframe object)
    
#     #PROB: days = 4*k to ensure all k days are covered.
#     #PROB: cal.schedule should be taking in utc time dates --- not ny time dates????
    


#     time_range = mcal.date_range(schedule, frequency=f"{interval}min")  #return a DatetimeIndex object with all timestamps at the frequency given, within the valid range of date and time 
    
#     #(in this case, valid range is the first timestamp in calendar to last timestamp in schedule. For example, my calendar starts at 2023/7/19 22:00 and ends at 2023/7/24 15:00, it'll form list of time 30minutes at a time starting with 22:30 on 7/19 til the end.)

#     #PROB: so trading hours for these days, except you increment at 30 minutes at a time. Sometimes it goes 15 minutes at a time tho --- THIS IS DESPITE THE MARKET NOT CLOSING AT :15. 



#     #when searching: pick keywords common to other ppl.

#     results = []
#     last_time = None
    
#     for utc_time in time_range:
        
#         nyc_time = utc_time.astimezone(pytz.timezone("America/New_York")) #this is a pandas.timestamp, equivalent to datetime.datetime

#         if nyc_time < now:
#             continue
#         # There are cases where market stops around 16:15 EST and it will be
#         # included in schedule. We would like to skip it.
#         if last_time and nyc_time < last_time + datetime.timedelta(minutes=interval):
#             continue
#         last_time = nyc_time
#         results.append(nyc_time.timestamp()) #PROB: !!!!As long as its a time period in between, it'll be accepted??? 
#         if len(results) >= k:
#             break 
#     return results 

'''
PROB's ex:
        market_open               break_start                 break_end              market_close
2023-07-20 2023-07-19 22:00:00+00:00 2023-07-20 20:15:00+00:00 2023-07-20 20:30:00+00:00 2023-07-20 21:00:00+00:00
2023-07-21 2023-07-20 22:00:00+00:00 2023-07-21 20:15:00+00:00 2023-07-21 20:30:00+00:00 2023-07-21 21:00:00+00:00
2023-07-24 2023-07-23 22:00:00+00:00 2023-07-24 20:15:00+00:00 2023-07-24 20:30:00+00:00 2023-07-24 21:00:00+00:00
2023-07-25 2023-07-24 22:00:00+00:00 2023-07-25 20:15:00+00:00 2023-07-25 20:30:00+00:00 2023-07-25 21:00:00+00:00
2023-07-26 2023-07-25 22:00:00+00:00 2023-07-26 20:15:00+00:00 2023-07-26 20:30:00+00:00 2023-07-26 21:00:00+00:00

resulting time intervals by going 30 minutes:

DatetimeIndex(['2023-07-19 22:30:00+00:00', '2023-07-19 23:00:00+00:00',
            '2023-07-19 23:30:00+00:00', '2023-07-20 00:00:00+00:00',
            '2023-07-20 00:30:00+00:00', '2023-07-20 01:00:00+00:00',
            '2023-07-20 01:30:00+00:00', '2023-07-20 02:00:00+00:00',
            '2023-07-20 02:30:00+00:00', '2023-07-20 03:00:00+00:00',
            ...
            '2023-07-26 16:30:00+00:00', '2023-07-26 17:00:00+00:00',
            '2023-07-26 17:30:00+00:00', '2023-07-26 18:00:00+00:00',
            '2023-07-26 18:30:00+00:00', '2023-07-26 19:00:00+00:00',
            '2023-07-26 19:30:00+00:00', '2023-07-26 20:00:00+00:00',
            '2023-07-26 20:15:00+00:00', '2023-07-26 21:00:00+00:00'],
            dtype='datetime64[ns, UTC]', length=230, freq=None)

'''

    




