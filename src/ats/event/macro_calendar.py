import datetime
import logging
import os

from time import sleep
import json
import requests
import pandas as pd

from ats.util import config_utils
from ats.util import logging_utils
from ats.calendar import date_utils


def get_release_dates(date):
    date_str = date.strftime("%Y-%m-%d")
    year_month_str = date.strftime("%Y-%m")
    url = f"https://api.stlouisfed.org/fred/releases/dates?api_key={api_key}&file_type=json&realtime_start={date_str}&realtime_end={date_str}"
    response = requests.get(url)
    try:
        response = response.json()
    except Exception as e:
        logging.error(f"response:{response}")
        return
    df_vec = []
    if not "release_dates" in response:
        return
    result = response["release_dates"]
    columns = ["release_id", "release_name", "date"]
    result_dict = [r for r in result]
    indices = [r["release_id"] for r in result]
    logging.info(f"result_dict:{result_dict}")
    df = pd.DataFrame(result_dict, index=indices)
    dir_path = f"/home/ubuntu/ats/data/event/macro/release_date/{year_month_str}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    df.to_csv(f"{dir_path}/{date_str}.csv", sep="~")

def get_releases(month):
    date_str = date.strftime("%Y-%m-%d")
    year_month_str = date.strftime("%Y-%m")
    url = f"https://api.stlouisfed.org/fred/releases?api_key={api_key}&file_type=json&realtime_start={date_str}&realtime_end={date_str}"
    response = requests.get(url).json()
    df_vec = []
    result = response["releases"]
    columns = ["id", "realtime_start", "realtime_end", "name", "press_release", "link"]
    result_dict = [r for r in result]
    indices = [r["id"] for r in result]
    logging.info(f"result_dict:{result_dict}")
    df = pd.DataFrame(result_dict, index=indices)
    dir_path = f"/home/ubuntu/ats/data/event/macro/{year_month_str}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    df.to_csv(f"{dir_path}/{date_str}.csv", sep="~")

def get_series_observation(start_date, end_date, now, series_ids, start_series):
    #start_date_str = start_date.strftime("%Y-%m-%d")
    #end_date_str = end_date.strftime("%Y-%m-%d")
    year_month_str = now.strftime("%Y-%m")
    df_vec = []
    for series_id in series_ids["series_id"]:
        if start_series and series_id<start_series:
            continue
        logging.info(f"series_id:{series_id}")
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&realtime_start={start_date}&realtime_end={end_date}"
        logging.info(f"url:{url}")
        response = requests.get(url).json()
        logging.info(f"response:{response}")
        if not "observations" in response:
            continue
        result = response["observations"]
        columns = ["date", "realtime_start", "realtime_end", "value"]
        result_dict = [r for r in result]
        indices = [r["date"] for r in result]
        df = pd.DataFrame(result_dict, index=indices)
        df["series_id"] = series_id
        df_vec.append(df)
        sleep(1)
    df = pd.concat(df_vec)
    #logging.info(f"merged_df:{df}")
    dir_path = f"/home/ubuntu/ats/data/event/macro/series_observation/{year_month_str}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    df.to_csv(f"{dir_path}/{start_date}_{end_date}.csv", sep="~")
    
def get_series(date, releases_ids):
    date_str = date.strftime("%Y-%m-%d")
    year_month_str = date.strftime("%Y-%m")
    df_vec = []
    for release_id in release_ids["release_id"]:
        logging.info(f"release_id:{release_id}")
        url = f"https://api.stlouisfed.org/fred/release/series?release_id={release_id}&api_key={api_key}&file_type=json&realtime_start={date_str}&realtime_end={date_str}"
        logging.info(f"url:{url}")
        response = requests.get(url).json()
        #logging.info(f"response:{response}")
        if not "seriess" in response:
            continue
        result = response["seriess"]
        columns = ["id", "realtime_start", "realtime_end", "title", "observation_start", "observation_end",
                   "frequency", "frequency_short", "units", "units_short", "seasonal_adjustment",
                   "seasonal_adjustment_short", "last_updated", "popularity", "group_popularity", "notes"]
        result_dict = [r for r in result]
        indices = [r["id"] for r in result]
        #logging.info(f"result_dict:{result_dict}")
        df = pd.DataFrame(result_dict, index=indices)
        df["release_id"] = release_id
        #logging.info(f"df:{df}")
        df_vec.append(df)
        sleep(1)
    df = pd.concat(df_vec)
    #logging.info(f"merged_df:{df}")
    dir_path = f"/home/ubuntu/ats/data/event/macro/series/{year_month_str}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    df.to_csv(f"{dir_path}/{date_str}.csv", sep="~")


if __name__ == "__main__":
    parser = config_utils.get_arg_parser("Get macro events from FRED")
    parser.add_argument("--mode", type=str)
    parser.add_argument("--start_date", type=str)
    parser.add_argument("--end_date", type=str)
    parser.add_argument("--start_series", type=str)
    now = datetime.datetime.now()
    args = parser.parse_args()

    logging_utils.init_logging()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)
    api_key = "5e7a5f80ba03257b5913e97bec786466"
    start_date = args.start_date
    end_date = args.end_date
    if not start_date:
        start_date = now - datetime.timedelta(days=30)
    if not end_date:
        end_date = now + datetime.timedelta(days=180)

    if args.mode == "series":
        release_ids = pd.read_csv("data/event/macro/fred_release_ids.txt",
                                  header=None, names=["release_id"])
        logging.info(f"release_ids:{release_ids}")
        for date in pd.date_range(start=start_date, end=end_date):
            get_series(date, release_ids)
            sleep(1)

    if args.mode == "release_date":
        for date in pd.date_range(start=start_date, end=end_date):
            get_release_dates(date)

    if args.mode == "series_observation":
        series_ids = pd.read_csv("data/event/macro/series.txt",
                                 header=None, names=["series_id"])
        logging.info(f"series_ids:{series_ids}")
        get_series_observation(start_date, end_date, now, series_ids, args.start_series)
