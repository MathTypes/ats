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


#def scrape_tweets(keyword, since, until, output_file):
    #output_file = os.path.join(output_dir, username + f"-from-{since}-{until}" + ".csv")
    #if not os.path.exists(output_file):
    #    scrape_keyword(
    #        keyword = f"from:{username}",
    #        since=since,
    #        until=until,
    #        output_format="csv",
    #        browser="chrome",
    #        tweets_count=1000000,
    #        filename=username + f"-from-{since}",
    #        directory=output_dir,
    #        headless=False,
    #    )

if __name__ == "__main__":
    parser = config_utils.get_arg_parser("Scape tweet")
    parser.add_argument("--username", type=str)
    parser.add_argument("--symbol", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--login", type=bool)
    parser.add_argument("--email", type=str)
    parser.add_argument("--browser_profile", type=str)
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
    if not args.username and not args.symbol:
        logging.error("Must specify username or symbol")
        exit(-1)
    if not args.output_dir:
        logging.error("Must specify output_dir")
        exit(-1)

    end_date = args.end_date
    if end_date < args.start_date + datetime.timedelta(31):
        end_date = args.start_date + datetime.timedelta(31)
    for cur_date in pd.date_range(args.start_date, end_date, freq="D"):
        since = cur_date
        until = cur_date + datetime.timedelta(days=1)
        if args.username:
            output_file_name = args.username + f"-to-{since}-{until}"
            keyword = f"from: {args.username}"
        else:
            output_file_name = args.symbol + f"-to-{since}-{until}"
            keyword = f"${args.symbol}"
        empty_tweets = set()
        # Creating list to append tweet data
        tweets_list1 = []
        since = since.date().isoformat()
        until = until.date().isoformat()
        #query = f"{keyword} since:{since} until:{until}"
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        output_file = args.output_dir + "/" + output_file_name + ".csv"
        logging.info(f"Checking output_file:{output_file}")
        if not os.path.exists(output_file):
            browser_profile = None
            email = None
            if args.email:
                email = args.email
                browser_profile = args.browser_profile + "/selenium_profile_" + email
            scrape_keyword(
                keyword = f"{keyword}",
                since=since,
                until=until,
                output_format="csv",
                browser="chrome",
                tweets_count=1000000,
                filename=output_file_name,
                directory=args.output_dir,
                browser_profile=browser_profile,
                headless=False,
                email = email,
                login = args.login
            )
