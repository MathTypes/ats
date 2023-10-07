import datetime
import logging

import pandas_market_calendars as mcal
import pytz

from ats.calendar import market_time
from ats.util import logging_utils

from datetime import timezone


def test_open_time_with_date():
    market_cal = mcal.get_calendar("CME_Equity")
    # 2009-06-01 is Monday
    # 2009-06-01 00:00:00+00:00
    train_start_date = datetime.datetime.strptime("2009-06-01", "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    ).date()
    open_time = market_time.get_open_time(market_cal, train_start_date)
    # Sun May 31 2009 15:00:00
    assert open_time == 1243807200

    # logging.error("\n\n\n\n\nthe type of object is: " + str(type(market_time.get_close_time(market_cal, train_start_date))) + "\n\n\n\n\n")
    close_time = market_time.get_close_time(market_cal, train_start_date)

    # Mon Jun 01 2009 14:00:00
    assert close_time == 1243890000.0


def test_next_trading_times_intraday_cme_equity_30ms():
    market_cal = mcal.get_calendar("CME_Equity") #basic exchange calendar
    # market_cal=mcal.get_calendar('NYSE', open_time=datetime.time(9, 30), close_time=datetime.time(16, 0))
    # Mon Jul 24 2023 05:30:00 PST
    nyc_time = datetime.datetime(2023, 7, 24, 12, 30, tzinfo=timezone.utc).astimezone(
        pytz.timezone("America/New_York")
    ) #8:30 ny time
    trading_times = market_time.get_next_trading_times(market_cal, 30, nyc_time, 2)
    assert trading_times == [1690201800.0, 1690203600.0]

# def test_next_trading_times_intraday_cme_equity_30m_2hrbefore(): 
#     market_cal = mcal.get_calendar("CME_Equity")
#     # Mon Jul 24 2023 05:30:00 PST

#     logging.error("hereeee")
#     # nyc_time = datetime.datetime(2023, 7, 24, 12, 30, tzinfo=timezone.utc).astimezone(
#     #     pytz.timezone("America/New_York")
#     # )
#     nyc_time = datetime.datetime(2023, 7, 24, 10, 30, tzinfo=datetime.timezone(datetime.timedelta(seconds=3600))).astimezone(
#         pytz.timezone("America/New_York")
#     )
#     '''
#     #ny time from london time
#     #input time: 10:30am --- 5:30 nyc time.
#     #next trading time: 2023/7/24, 9:30 nyc time --- translates to 2023/7/24 6:30 local time, or 1690205400000, which is not the opening time being tested.
#     '''  

#     logging.error("\n\n\n\nnyc_time is:" + str(nyc_time))
#     logging.info(f"nyc_time:{nyc_time}")

#     trading_times = market_time.get_next_trading_times(market_cal, 30, nyc_time, 2)
#     #market_time in src.ats.calendar
#     assert trading_times == [1690201800.0, 1690203600.0]
#     #[july 24, 12:30pm utc time, july 24 1pm, utc time]

def test_next_trading_times_near_friday_close_cme_equity_30m():
    market_cal = mcal.get_calendar("CME_Equity") #CME Equity hours: NY time mon->fri, prev day 6pm -> nxt day 5pm.
    #or chicago time, prev day 5pm -> nxt day 4pm.

    logging.error("\n\n\n\n\n\nmarket_cal time zone for CME is " + str(market_cal.tz.zone))

    # Fri Jul 21 2023 12:30:00  
    '''
    wait, isn't this 2023/7/21, 15:30 --- its utc-4 during daylight savings. 
    But no matter, since next trading time is [2023/7/23, 5pm ny time, 7/24 5pm, 7/25 5pm, 7/26 5pm, etc.]
    '''
    nytime = datetime.datetime(2023, 7, 21, 19, 30).astimezone(
        pytz.timezone("America/New_York")
    )
    logging.info(f"nytime:{nytime}")

    trading_times = market_time.get_next_trading_times(market_cal, 30, nytime, 10)
    # Fri Jul 21 2023 12:30, 13:00:00, 13:15, 14:00
    # Sunday 15:30, 16:00
    logging.error(f"trading_times:{trading_times}")
    assert trading_times == [
        1689967800.0, 1689969600.0, 1689973200.0, 1690151400.0, 1690153200.0, 1690155000.0, 1690156800.0, 1690158600.0, 1690160400.0, 1690162200.0
    ]
'''
    so its asserting [Friday, July 21, 2023 7:30:00, Friday, July 21, 2023 8:00:00, etc.]
    
    NY time mon->fri, prev day 6pm -> nxt day 5pm. 
    
    Datetime to posix/epoch: dtobj.timestamp()
    posix/epoch to datetime: datetime.datetime.fromtimestamp(1695203619)


    The reason above seems wrong:
            market_open               break_start                 break_end              market_close
2023-07-20 2023-07-19 22:00:00+00:00 2023-07-20 20:15:00+00:00 2023-07-20 20:30:00+00:00 2023-07-20 21:00:00+00:00
2023-07-21 2023-07-20 22:00:00+00:00 2023-07-21 20:15:00+00:00 2023-07-21 20:30:00+00:00 2023-07-21 21:00:00+00:00
2023-07-24 2023-07-23 22:00:00+00:00 2023-07-24 20:15:00+00:00 2023-07-24 20:30:00+00:00 2023-07-24 21:00:00+00:00
2023-07-25 2023-07-24 22:00:00+00:00 2023-07-25 20:15:00+00:00 2023-07-25 20:30:00+00:00 2023-07-25 21:00:00+00:00
2023-07-26 2023-07-25 22:00:00+00:00 2023-07-26 20:15:00+00:00 2023-07-26 20:30:00+00:00 2023-07-26 21:00:00+00:00

    
    '''



    #Notes when constructing test cases: London is UTC+1 during the summer (march->October), UTC normally.


'''
    CME hours: NY time mon->fri, prev day 6pm -> nxt day 5pm. 
    
    Datetime to posix/epoch: dtobj.timestamp()
    posix/epoch to datetime: datetime.datetime.fromtimestamp(1695203619) --- returns local date!!!
    
'''


def test_next_trading_times_near_monday_close_cme_equity_30m():
    market_cal = mcal.get_calendar("CME_Equity")
    
    nytime = datetime.datetime(2023, 7, 24, 19, 30).astimezone(
        pytz.timezone("America/New_York")
    )

    #nytime, 15:30 on 2023/7/24 --- which is a Monday. Meaning, next trading time would be:
    #[2023/7/24 at 18:00, 2023/7/25 18:00, etc.]

    logging.info(f"utc_time:{nytime}")

    trading_times = market_time.get_next_trading_times(market_cal, 30, nytime, 10)
    
    logging.error(f"trading_times:{trading_times}")
    assert trading_times == [
        1690227000.0, 1690228800.0, 1690232400.0, 1690237800.0, 1690239600.0, 1690241400.0, 1690243200.0, 1690245000.0, 1690246800.0, 1690248600.0
    ]


def test_next_trading_times_at_open_cme_equity_30m():
    market_cal = mcal.get_calendar("CME_Equity")
    # Sat
    utc_time = datetime.datetime(2023, 7, 22, 14, 30).astimezone(
        pytz.timezone("America/New_York")
    )
    logging.info(f"utc_time:{utc_time}")

    trading_times = market_time.get_next_trading_times(market_cal, 30, utc_time, 2)
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

    trading_times = market_time.get_next_trading_times(market_cal, 5, utc_time, 2)
    # Sun Jul 23 2023 15:05:00, 15:10
    assert trading_times == [1690149900.0, 1690150200.0]


def test_next_trading_times_at_open():
    market_cal = mcal.get_calendar("NYSE")
    utc_time = datetime.datetime(2012, 1, 4, 14, 30).astimezone(
        pytz.timezone("America/New_York")
    )
    logging.info(f"utc_time:{utc_time}")
    trading_times = market_time.get_next_trading_times(market_cal, 30, utc_time, 2)
    assert trading_times[0] == 1325689200.0
    assert trading_times[1] == 1325691000.0


def test_next_trading_times_at_open_plus_30m():
    market_cal = mcal.get_calendar("NYSE")
    utc_time = datetime.datetime(2012, 1, 4, 15, 30).astimezone(
        pytz.timezone("America/New_York")
    )
    logging.info(f"utc_time:{utc_time}")
    trading_times = market_time.get_next_trading_times(market_cal, 30, utc_time, 2)
    assert trading_times[0] == 1325691000.0
    assert trading_times[1] == 1325692800.0


def test_next_trading_times_at_close_minus_30m():
    market_cal = mcal.get_calendar("NYSE")
    utc_time = datetime.datetime(2012, 1, 4, 21, 30).astimezone(
        pytz.timezone("America/New_York")
    )
    logging.info(f"utc_time:{utc_time}")
    trading_times = market_time.get_next_trading_times(market_cal, 30, utc_time, 2)
    assert trading_times[0] == 1325775600.0
    assert trading_times[1] == 1325777400.0


if __name__ == "__main__":
    logging_utils.init_logging()
