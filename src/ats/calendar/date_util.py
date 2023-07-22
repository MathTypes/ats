import datetime
import logging
import math

def get_option_expiration_week(date):
    first_day = date.replace(day=1)
    day_of_month = date.day
    if first_day.weekday() == 5:
        return 4
    else:
        return 3

def get_option_expiration_day(date):
    if date.weekday()==6:
        date = date + datetime.timedelta(days=1)
    option_expiration_week = get_option_expiration_week(date)
    cur_week = get_week_of_month(date)
    while cur_week != option_expiration_week:
        date = date + datetime.timedelta(days=7)
        cur_week = get_week_of_month(date)
        logging.error(f"cur_week:{cur_week}, date:{date}, option_expiration_week:{option_expiration_week}")
        # Needs to recompute since we might move to next month
        option_expiration_week = get_option_expiration_week(date)
    while date.weekday() != 4:
        date = date + datetime.timedelta(days=1)
    return date

def get_week_of_month(date):
    logging.error(f"date:{date}")
    first_day = date.replace(day=1)
    day_of_month = date.day
    if first_day.weekday() == 6:
        adjusted_dom = day_of_month - 1
    else:
        adjusted_dom = day_of_month + first_day.weekday()
    week_of_month = int(math.floor(adjusted_dom / 7.0)) + 1
    return week_of_month

def monthlist(begin, end):
    result = []
    while True:
        if begin.month == 12:
            next_month = begin.replace(year=begin.year + 1, month=1, day=1)
        else:
            next_month = begin.replace(month=begin.month + 1, day=1)
        if next_month > end:
            break
        result.append([begin, last_day_of_month(begin)])
        begin = next_month
    result.append([begin, end])
    return result


# borrowed from https://stackoverflow.com/a/13565185
# as noted there, the calendar module has a function of its own
def last_day_of_month(any_day):
    next_month = any_day.replace(day=28) + datetime.timedelta(
        days=4
    )  # this will never fail
    return next_month - datetime.timedelta(days=next_month.day)
