import datetime
import logging

import pandas_market_calendars as mcal
import pytz

from ats.calendar import date_util
from ats.util import logging_utils


def test_get_week_of_month_first_week():
    # 2009-06-01 is Monday
    # 2009-06-01 00:00:00+00:00
    train_start_date = datetime.datetime.strptime("2009-06-01", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_week_of_month(train_start_date) == 1

def test_get_week_of_month_second_week():
    # 2023-07-05 00:00:00+00:00
    train_start_date = datetime.datetime.strptime("2023-07-05", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_week_of_month(train_start_date) == 2

def test_get_week_of_month_third_week():
    # 2023-07-10 00:00:00+00:00
    train_start_date = datetime.datetime.strptime("2023-07-10", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_week_of_month(train_start_date) == 3

def test_get_week_of_month_fourth_week():
    # 2023-07-17 00:00:00+00:00
    train_start_date = datetime.datetime.strptime("2023-07-17", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_week_of_month(train_start_date) == 4

def test_get_week_of_month_fifth_week():
    # 2023-07-24
    train_start_date = datetime.datetime.strptime("2023-07-24", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_week_of_month(train_start_date) == 5

def test_get_week_of_month_sixth_week():
    # 2023-07-31
    train_start_date = datetime.datetime.strptime("2023-07-31", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_week_of_month(train_start_date) == 6
