# importing libraries and packages
# from absl import logging
# Example of usage:
# YTHONPATH=. python3 twitter/selenium_scrape.py --symbol=qqq --start_date=2022-04-01 --end_date=2023-04-26 --output_dir=/Volumes/Seagate\ Portable\ Drive/data/selenium/user --email=alexsimon788213 --browser_profile=/Users/jianjunchen/repo/ats-1/src 
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
import logging
#from google.cloud import firestore_v1 as firestore
from firebase_admin import firestore
from util import logging_utils

import firebase_admin
from firebase_admin import credentials
from firebase import firebase

def read_collection(db, query):
    doc_ref = db.collection('unique_ids').document(query)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    else:
        return None

if __name__ == "__main__":
    parser = config_utils.get_arg_parser("Scape tweet")
    parser.add_argument("--username", type=str)
    parser.add_argument("--symbol", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--since_id", type=str)
    parser.add_argument("--max_id", type=str)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--login", type=bool)
    parser.add_argument("--headless", type=bool)
    parser.add_argument("--email", type=str)
    parser.add_argument("--firebase_cert", type=str)
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

    cred = credentials.Certificate(args.firebase_cert)
    default_app = firebase_admin.initialize_app(cred)
    db = firestore.client()
    firebase = firebase.FirebaseApplication('https://keen-rhino-386415.firebaseio.com', None)

    until = datetime.datetime.today()
    for symbol in args.symbol.split(","):
        keyword = f"${symbol}"
        id_key = f"selenium_recent_symbol_{symbol}"
        df = read_collection(db, keyword)
        since_id = None
        max_id = None
        if df:
            since_id = df["last_id"]
        if args.since_id:
            since_id = args.since_id
        if args.max_id:
            max_id = args.max_id
        logging.info(f"search:{keyword}, since_id:{since_id}, max_id:{max_id}")
        empty_tweets = set()
        # Creating list to append tweet data
        tweets_list1 = []
        browser_profile = None
        email = None
        if args.email:
            email = args.email
            browser_profile = args.browser_profile + "/selenium_profile_" + email
        since = args.start_date
        until = args.end_date
        logging.info(f"keyword:{keyword}")
        logging.info(f"browser_profile:{browser_profile}")
        logging.info(f"args.login:{args.login}")
        data = scrape_keyword(
            keyword = f"{keyword}",
            output_format="",
            browser="chrome",
            since_id=since_id,
            max_id=max_id,
            tweets_count=args.rows,
            filename="",
            directory="",
            browser_profile=browser_profile,
            headless=args.headless,
            email = email,
            login = args.login
        )
        last_tweet_id = int(since_id) if since_id else None
        if data:
            for key, value in data.items():
                logging.info(f"key:{key}, value:{value}")
                db.collection(u'recent_tweet_symbol').document(key).set(value)
                if not last_tweet_id or int(value["tweet_id"]) > last_tweet_id:
                    last_tweet_id = int(value["tweet_id"])
        db.collection(u'unique_ids').document(id_key).set({"last_id":last_tweet_id})
