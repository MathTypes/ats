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

import snscrape.modules.twitter as sntwitter
from util import config_utils
from util import logging_utils
import requests
import os
from urllib.parse import urlparse, parse_qsl, parse_qs
import numpy as np

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def process_df(df, output_dir):
    p = re.compile(r"'fullUrl':\s+'([^']+)'")
    #p = re.compile(r"'fullUrl:")
    for index, row in df.iterrows():
        logging.info(f"row:{row['media']}")
        #if isfloat(row["media"]):
        #    continue
        try:
            matches = p.findall(row['media'])
            logging.info(f'matches:{matches}')
            for i, image_url in enumerate(matches):
                logging.info(f"url:{image_url}")
                if "com/media" in image_url:
                    logging.info(f"Image url:{image_url}")
                    parsed = urlparse(image_url)
                    format = "na"
                    if "format=" in image_url:
                        format = parse_qs(parsed.query)['format'][0]
                    img_data = requests.get(image_url).content
                    img_path = output_dir + "/" + str(row["id"]) + "_" + str(i) + "_" + os.path.basename(parsed.path) + f".{format}"
                    logging.info(f"Writing image:{img_path}")
                    with open(img_path, 'wb') as handler:
                        handler.write(img_data)
        except Exception as e:
            logging.info(f"Exception:{e}")
            pass

if __name__ == "__main__":
    parser = config_utils.get_arg_parser("Scrape tweet by id")
    parser.add_argument("--reply_file", type=str)

    args = parser.parse_args()
    config_utils.set_args(args)
    logging_utils.init_logging()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    if args.reply_file:
        done_file = args.reply_file + ".img.done"
        if not os.path.exists(done_file):
            existing_tweets = set()
            df = pd.read_csv(args.reply_file, sep='`')
            process_df(df, os.path.dirname(done_file))
            with open(done_file, 'w') as fp:
                pass
            exit(0)
