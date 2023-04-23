# importing libraries and packages
# from absl import logging
# Example of usage:
# PYTHONPATH=. python3 twitter/selenium_scrape.py --username=eliant_capital --since=2023-03-23 --until=2023-03-24
#
import argparse
import logging
import datetime
import os

import pandas as pd

from twitter_scraper_selenium import get_profile_details
from twitter_scraper_selenium import scrape_profile, scrape_keyword
from util import config_utils
from util import logging_utils


def scrape_tweets(username, since, until, output_dir):
    empty_tweets = set()
    # Creating list to append tweet data
    tweets_list1 = []
    since = since.date().isoformat()
    until = until.date().isoformat()
    query = f"from:{username} since:{since} until:{until}"
    output_file = os.path.join(output_dir, args.username + f"-to-{since}" + ".csv")
    if not os.path.exists(output_file):
        scrape_keyword(
            keyword = f"to:{username}",
            since=since,
            until=until,
            output_format="csv",
            browser="chrome",
            tweets_count=1000000,
            filename=username + f"-to-{since}",
            directory=output_dir,
            headless=False,
        )
    output_file = os.path.join(output_dir, username + f"-to-{since}" + ".csv")
    if not os.path.exists(output_file):
        scrape_keyword(
            keyword = f"from:{username}",
            since=since,
            until=until,
            output_format="csv",
            browser="chrome",
            tweets_count=1000000,
            filename=username + f"-from-{since}",
            directory=output_dir,
            headless=False,
        )

if __name__ == "__main__":
    parser = config_utils.get_arg_parser("Scape tweet")
    parser.add_argument("--username", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument(
        "--start_date",
        type=lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").date(),
        required=False,
        help="Set a start date",
    )
    parser.add_argument(
        "--end_date",
        type=lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").date(),
        required=False,
        help="Set a end date",
    )

    args = parser.parse_args()
    config_utils.set_args(args)
    logging_utils.init_logging()
    if not args.username or not args.output_dir:
        logging.error("Must specify username and output_dir")
        exit(-1)

    if args.end_date:
        end_date = args.end_date
        if end_date < args.start_date + datetime.timedelta(31):
            end_date = args.start_date + datetime.timedelta(31)
        for cur_date in pd.date_range(args.start_date, end_date, freq="M"):
            cur_start_date = cur_date-datetime.timedelta(2)
            cur_end_date = cur_date + datetime.timedelta(days=32)
            scrape_tweets(args.username, cur_start_date, cur_end_date, args.output_dir)
    else:
        scrape_profile(
            twitter_username=args.username,
            output_format="csv",
            browser="chrome",
            tweets_count=1000000,
            filename=args.username,
            directory=args.output_dir,
        )


    logging.info(f'checking: {output_file}')
