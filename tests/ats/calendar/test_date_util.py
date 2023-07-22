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

def test_option_expiration_jan():
    train_start_date = datetime.datetime.strptime("2023-01-02", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_option_expiration_week(train_start_date) == 3

def test_option_expiration_feb():
    train_start_date = datetime.datetime.strptime("2023-02-02", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_option_expiration_week(train_start_date) == 3

def test_option_expiration_mar():
    train_start_date = datetime.datetime.strptime("2023-03-23", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_option_expiration_week(train_start_date) == 3

def test_option_expiration_apr():
    train_start_date = datetime.datetime.strptime("2023-04-13", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_option_expiration_week(train_start_date) == 4

def test_option_expiration_may():
    train_start_date = datetime.datetime.strptime("2023-05-22", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_option_expiration_week(train_start_date) == 3

def test_option_expiration_jun():
    train_start_date = datetime.datetime.strptime("2023-06-30", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_option_expiration_week(train_start_date) == 3

def test_option_expiration_jul():
    train_start_date = datetime.datetime.strptime("2023-07-20", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_option_expiration_week(train_start_date) == 4

def test_option_expiration_aug():
    train_start_date = datetime.datetime.strptime("2023-08-31", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_option_expiration_week(train_start_date) == 3

def test_option_expiration_sep():
    train_start_date = datetime.datetime.strptime("2023-09-30", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_option_expiration_week(train_start_date) == 3

def test_option_expiration_oct():
    train_start_date = datetime.datetime.strptime("2023-10-02", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_option_expiration_week(train_start_date) == 3

def test_option_expiration_nov():
    train_start_date = datetime.datetime.strptime("2023-11-12", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_option_expiration_week(train_start_date) == 3

def test_option_expiration_dec():
    train_start_date = datetime.datetime.strptime("2023-12-10", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_option_expiration_week(train_start_date) == 3

