import logging
import os

from time import sleep
import json
import requests
import pandas as pd

from ats.util import logging_utils

if __name__ == "__main__":
    logging_utils.init_logging()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option('display.max_colwidth', None)
    api_key = "5e7a5f80ba03257b5913e97bec786466"
    for date in pd.date_range(start="2001-01-01", end="2023-07-19"):
        date_str = date.strftime("%Y-%m-%d")
        year_month_str = date.strftime("%Y-%m")
        url = f"https://api.stlouisfed.org/fred/releases?api_key={api_key}&file_type=json&realtime_start={date_str}&realtime_end={date_str}"
        response = requests.get(url).json()
        #data = json.loads(response)
        # Use json_normalize() to convert JSON to DataFrame
        #dict= json.loads(response)
        #df = json_normalize(response['releases'])
        df_vec = []
        result = response['releases']
        columns = ["id", "realtime_start", "realtime_end", "name", "press_release", "link"]
        result_dict = [r for r in result]
        indices = [r["id"] for r in result]
        logging.info(f"result_dict:{result_dict}")
        df = pd.DataFrame(result_dict, index=indices)
        dir_path = f"/home/ubuntu/ats/data/event/macro/{year_month_str}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        df.to_csv(f"{dir_path}/{date_str}.csv", sep="~")
        # Convert JSON to DataFrame Using read_json()
        #df2 = pd.read_json(jsonStr, orient ='index')

        # Use pandas.DataFrame.from_dict() to Convert JSON to DataFrame
        #dict= json.loads(data)
        #df2 = pd.DataFrame.from_dict(dict, orient="index")

        logging.info(f"response:{df}")
        sleep(1)
