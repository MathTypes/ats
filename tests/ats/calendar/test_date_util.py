import datetime
import logging

import pandas_market_calendars as mcal
import pytz

from ats.calendar import date_util
from ats.util import logging_utils


def test_get_week_of_month_first_week_first():
    assert date_util.get_week_of_month(datetime.date(2023,1,1)) == 1

def test_get_week_of_month_first_week_last():
    assert date_util.get_week_of_month(datetime.date(2023,1,7)) == 1

def test_get_week_of_month_second_week_first():
    assert date_util.get_week_of_month(datetime.date(2023,1,8)) == 2

def test_get_week_of_month_second_week_last():
    assert date_util.get_week_of_month(datetime.date(2023,1,14)) == 2

def test_get_week_of_month_third_week_first():
    assert date_util.get_week_of_month(datetime.date(2023,1,15)) == 3

def test_get_week_of_month_third_week_last():
    assert date_util.get_week_of_month(datetime.date(2023,1,21)) == 3

def test_get_week_of_month_fourth_week_first():
    assert date_util.get_week_of_month(datetime.date(2023,1,22)) == 4

def test_get_week_of_month_fourth_week_last():
    assert date_util.get_week_of_month(datetime.date(2023,1,28)) == 4

def test_get_week_of_month_fifth_week_first():
    assert date_util.get_week_of_month(datetime.date(2023,1,29)) == 5

def test_get_week_of_month_fifth_week_last():
    assert date_util.get_week_of_month(datetime.date(2023,1,31)) == 5

def test_get_week_of_month_first_week_():
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
    assert date_util.get_option_expiration_day(datetime.date(2023, 1, 1)).strftime("%Y%m%d") == "20230120"
    assert date_util.get_option_expiration_day(datetime.date(2023, 1, 19)).strftime("%Y%m%d") == "20230120"
    assert date_util.get_option_expiration_day(datetime.date(2023, 1, 20)).strftime("%Y%m%d") == "20230120"
    # Now we are past expiration
    assert date_util.get_option_expiration_day(datetime.date(2023, 1, 21)).strftime("%Y%m%d") == "20230217"
    assert date_util.get_option_expiration_day(datetime.date(2023, 1, 23)).strftime("%Y%m%d") == "20230217"

def test_option_expiration_feb():
    train_start_date = datetime.datetime.strptime("2023-02-02", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_option_expiration_week(train_start_date) == 3
    assert date_util.get_option_expiration_day(datetime.date(2023, 2, 1)).strftime("%Y%m%d") == "20230217"
    assert date_util.get_option_expiration_day(datetime.date(2023, 2, 15)).strftime("%Y%m%d") == "20230217"
    assert date_util.get_option_expiration_day(datetime.date(2023, 2, 17)).strftime("%Y%m%d") == "20230217"
    # Now we are past expiration
    assert date_util.get_option_expiration_day(datetime.date(2023, 2, 21)).strftime("%Y%m%d") == "20230317"
    assert date_util.get_option_expiration_day(datetime.date(2023, 2, 23)).strftime("%Y%m%d") == "20230317"

def test_option_expiration_mar():
    train_start_date = datetime.datetime.strptime("2023-03-23", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_option_expiration_week(train_start_date) == 3
    assert date_util.get_option_expiration_day(datetime.date(2023, 3, 1)).strftime("%Y%m%d") == "20230317"
    assert date_util.get_option_expiration_day(datetime.date(2023, 3, 15)).strftime("%Y%m%d") == "20230317"
    assert date_util.get_option_expiration_day(datetime.date(2023, 3, 17)).strftime("%Y%m%d") == "20230317"
    # Now we are past expiration
    assert date_util.get_option_expiration_day(datetime.date(2023, 3, 21)).strftime("%Y%m%d") == "20230421"
    assert date_util.get_option_expiration_day(datetime.date(2023, 3, 23)).strftime("%Y%m%d") == "20230421"

def test_option_expiration_apr():
    train_start_date = datetime.datetime.strptime("2023-04-13", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_option_expiration_week(train_start_date) == 4
    assert date_util.get_option_expiration_day(datetime.date(2023, 4, 1)).strftime("%Y%m%d") == "20230421"
    assert date_util.get_option_expiration_day(datetime.date(2023, 4, 15)).strftime("%Y%m%d") == "20230421"
    assert date_util.get_option_expiration_day(datetime.date(2023, 4, 21)).strftime("%Y%m%d") == "20230421"
    # Now we are past expiration
    assert date_util.get_option_expiration_day(datetime.date(2023, 4, 22)).strftime("%Y%m%d") == "20230519"
    assert date_util.get_option_expiration_day(datetime.date(2023, 4, 23)).strftime("%Y%m%d") == "20230519"

def test_option_expiration_may():
    train_start_date = datetime.datetime.strptime("2023-05-22", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_option_expiration_week(train_start_date) == 3
    assert date_util.get_option_expiration_day(datetime.date(2023, 5, 1)).strftime("%Y%m%d") == "20230519"
    assert date_util.get_option_expiration_day(datetime.date(2023, 5, 15)).strftime("%Y%m%d") == "20230519"
    assert date_util.get_option_expiration_day(datetime.date(2023, 5, 19)).strftime("%Y%m%d") == "20230519"
    # Now we are past expiration
    assert date_util.get_option_expiration_day(datetime.date(2023, 5, 21)).strftime("%Y%m%d") == "20230616"
    assert date_util.get_option_expiration_day(datetime.date(2023, 5, 23)).strftime("%Y%m%d") == "20230616"

def test_option_expiration_jun():
    train_start_date = datetime.datetime.strptime("2023-06-30", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_option_expiration_week(train_start_date) == 3
    assert date_util.get_option_expiration_day(datetime.date(2023, 6, 1)).strftime("%Y%m%d") == "20230616"
    assert date_util.get_option_expiration_day(datetime.date(2023, 6, 15)).strftime("%Y%m%d") == "20230616"
    assert date_util.get_option_expiration_day(datetime.date(2023, 6, 16)).strftime("%Y%m%d") == "20230616"
    # Now we are past expiration
    assert date_util.get_option_expiration_day(datetime.date(2023, 6, 21)).strftime("%Y%m%d") == "20230721"
    assert date_util.get_option_expiration_day(datetime.date(2023, 6, 23)).strftime("%Y%m%d") == "20230721"

def test_option_expiration_jul():
    train_start_date = datetime.datetime.strptime("2023-07-20", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_option_expiration_week(train_start_date) == 4
    assert date_util.get_option_expiration_day(datetime.date(2023, 7, 1)).strftime("%Y%m%d") == "20230721"
    assert date_util.get_option_expiration_day(datetime.date(2023, 7, 15)).strftime("%Y%m%d") == "20230721"
    assert date_util.get_option_expiration_day(datetime.date(2023, 6, 21)).strftime("%Y%m%d") == "20230721"
    # Now we are past expiratio7
    assert date_util.get_option_expiration_day(datetime.date(2023, 7, 23)).strftime("%Y%m%d") == "20230818"
    assert date_util.get_option_expiration_day(datetime.date(2023, 7, 31)).strftime("%Y%m%d") == "20230818"

def test_option_expiration_aug():
    train_start_date = datetime.datetime.strptime("2023-08-31", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_option_expiration_week(train_start_date) == 3
    assert date_util.get_option_expiration_day(datetime.date(2023, 8, 1)).strftime("%Y%m%d") == "20230818"
    assert date_util.get_option_expiration_day(datetime.date(2023, 8, 15)).strftime("%Y%m%d") == "20230818"
    assert date_util.get_option_expiration_day(datetime.date(2023, 8, 18)).strftime("%Y%m%d") == "20230818"
    # Now we are past expiration
    assert date_util.get_option_expiration_day(datetime.date(2023, 8, 21)).strftime("%Y%m%d") == "20230915"
    assert date_util.get_option_expiration_day(datetime.date(2023, 8, 23)).strftime("%Y%m%d") == "20230915"

def test_option_expiration_sep():
    train_start_date = datetime.datetime.strptime("2023-09-30", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_option_expiration_week(train_start_date) == 3
    assert date_util.get_option_expiration_day(datetime.date(2023, 9, 1)).strftime("%Y%m%d") == "20230915"
    assert date_util.get_option_expiration_day(datetime.date(2023, 9, 10)).strftime("%Y%m%d") == "20230915"
    assert date_util.get_option_expiration_day(datetime.date(2023, 9, 15)).strftime("%Y%m%d") == "20230915"
    # Now we are past expiration
    assert date_util.get_option_expiration_day(datetime.date(2023, 9, 21)).strftime("%Y%m%d") == "20231020"
    assert date_util.get_option_expiration_day(datetime.date(2023, 9, 23)).strftime("%Y%m%d") == "20231020"

def test_option_expiration_oct():
    train_start_date = datetime.datetime.strptime("2023-10-02", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_option_expiration_week(train_start_date) == 3
    assert date_util.get_option_expiration_day(datetime.date(2023, 10, 1)).strftime("%Y%m%d") == "20231020"
    assert date_util.get_option_expiration_day(datetime.date(2023, 10, 10)).strftime("%Y%m%d") == "20231020"
    assert date_util.get_option_expiration_day(datetime.date(2023, 10, 20)).strftime("%Y%m%d") == "20231020"
    # Now we are past expiration
    assert date_util.get_option_expiration_day(datetime.date(2023, 10, 21)).strftime("%Y%m%d") == "20231117"
    assert date_util.get_option_expiration_day(datetime.date(2023, 10, 23)).strftime("%Y%m%d") == "20231117"

def test_option_expiration_nov():
    train_start_date = datetime.datetime.strptime("2023-11-12", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_option_expiration_week(train_start_date) == 3
    assert date_util.get_option_expiration_day(datetime.date(2023, 11, 1)).strftime("%Y%m%d") == "20231117"
    assert date_util.get_option_expiration_day(datetime.date(2023, 11, 10)).strftime("%Y%m%d") == "20231117"
    assert date_util.get_option_expiration_day(datetime.date(2023, 11, 17)).strftime("%Y%m%d") == "20231117"
    # Now we are past expiration
    assert date_util.get_option_expiration_day(datetime.date(2023, 11, 21)).strftime("%Y%m%d") == "20231215"
    assert date_util.get_option_expiration_day(datetime.date(2023, 11, 23)).strftime("%Y%m%d") == "20231215"

def test_option_expiration_dec():
    train_start_date = datetime.datetime.strptime("2023-12-10", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    assert date_util.get_option_expiration_week(train_start_date) == 3
    assert date_util.get_option_expiration_day(datetime.date(2023, 12, 1)).strftime("%Y%m%d") == "20231215"
    assert date_util.get_option_expiration_day(datetime.date(2023, 12, 10)).strftime("%Y%m%d") == "20231215"
    assert date_util.get_option_expiration_day(datetime.date(2023, 12, 15)).strftime("%Y%m%d") == "20231215"
    # Now we are past expiration
    assert date_util.get_option_expiration_day(datetime.date(2023, 12, 21)).strftime("%Y%m%d") == "20240119"
    assert date_util.get_option_expiration_day(datetime.date(2023, 12, 23)).strftime("%Y%m%d") == "20240119"

