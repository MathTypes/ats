# importing libraries and packages
# from absl import logging
# Example of usage:
# PYTHONPATH=. python3 twitter/download_tweet.py --username=eliant_capital --since=2023-03-23 --until=2023-03-24
#
import argparse
import logging
import datetime
import json
from dataclasses import asdict 
import os
import re
import pandas as pd
from urllib3 import Retry
import snscrape.modules.twitter as sntwitter
from util import config_utils
from util import logging_utils
import requests
from requests.adapters import HTTPAdapter
import os
from urllib.parse import urlparse, parse_qsl, parse_qs
import numpy as np

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

DATE_PATTERN = re.compile(r".*(\d\d\d\d-\d\d-\d\d)T.*")
URL_PATTERN = re.compile(r"'fullUrl':\s+'([^']+)'")
    
def get_date_from_file(output_dir):
    matches = DATE_PATTERN.match(output_dir)
    if matches:
        return matches[1]
    return ""

def process_df(df, reply_file):
    done_file = reply_file + ".img.done"
    if os.path.exists(done_file):
        return
    output_dir = os.path.dirname(done_file)
    p = URL_PATTERN
    img_output_dir = os.path.dirname(reply_file)
    date_str = get_date_from_file(reply_file)
    if date_str:
        img_output_dir = img_output_dir + "/" + date_str
    if not os.path.exists(img_output_dir):
        os.makedirs(img_output_dir)
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1, allowed_methods=None, status_forcelist=[429, 500, 502, 503, 504]))
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    for index, row in df.iterrows():
        logging.info(f"row:{row}")
        logging.info(f"row:_media{row['media']}")
        #if isfloat(row["media"]):
        #    continue
        matches = p.findall(row['media'])
        logging.info(f'matches:{matches}')
        proxies = {
            "http": "http://host.docker.internal:8118",
            "https": "http://host.docker.internal:8118",
        }
        for i, image_url in enumerate(matches):
            logging.info(f"url:{image_url}")
            if "com/media" in image_url:
                logging.info(f"Image url:{image_url}")
                parsed = urlparse(image_url)
                format = "na"
                if "format=" in image_url:
                    format = parse_qs(parsed.query)['format'][0]
                    img_data = session.get(image_url, proxies=proxies).content
                    img_path = img_output_dir + "/" + str(row["id"]) + "_" + str(i) + "_" + os.path.basename(parsed.path) + f".{format}"
                    logging.info(f"Writing image:{img_path}")
                    with open(img_path, 'wb') as handler:
                        handler.write(img_data)
    with open(done_file, 'w') as fp:
        pass

if __name__ == "__main__":
    parser = config_utils.get_arg_parser("Scrape tweet images")
    parser.add_argument("--reply_file", type=str)

    args = parser.parse_args()
    config_utils.set_args(args)
    logging_utils.init_logging()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    if args.reply_file:
        existing_tweets = set()
        df = pd.read_csv(args.reply_file, sep='`')
        process_df(df, args.reply_file)
