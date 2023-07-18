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

from google.cloud import firestore_v1 as firestore
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

if __name__ == "__main__":
    parser = config_utils.get_arg_parser("Scape tweet")
    parser.add_argument("--username", type=str)
    parser.add_argument("--symbol", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--login", type=bool)
    parser.add_argument("--headless", type=bool)
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

    until = datetime.datetime.today()
    if args.username:
        output_file_name = args.username + f"-to-{until.strftime('%y%m%d')}"
        keyword = f"from:{args.username}"
    else:
        output_file_name = args.symbol + f"-to-{until.strftime('%y%m%d')}"
        keyword = f"${args.symbol}"
    empty_tweets = set()
    # Creating list to append tweet data
    tweets_list1 = []
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
        since = args.start_date
        until = args.end_date
        logging.info(f"keyword:{keyword}")
        logging.info(f"browser_profile:{browser_profile}")
        logging.info(f"args.login:{args.login}")
        data = scrape_keyword(
            keyword = f"{keyword}",
            output_format="",
            browser="chrome",
            tweets_count=30,
            filename=output_file_name,
            directory="",
            browser_profile=browser_profile,
            headless=args.headless,
            email = email,
            login = args.login
        )

        cred = credentials.Certificate("/Users/jianjunchen/repo/secrets/keen-rhino-386415-firebase-adminsdk-9jpg7-968c43811b.json")
        default_app = firebase_admin.initialize_app(cred)
        db = firestore.client()
        firebase = firebase.FirebaseApplication('https://keen-rhino-386415.firebaseio.com', None)

        #data = {'1657852295558709248': {'tweet_id': '1657852295558709248', 'username': 'davidcharting', 'name': 'David', 'profile_picture': 'https://pbs.twimg.com/profile_images/1641528445149519895/f5IC8trq_x96.jpg', 'replies': 0, 'retweets': 0, 'likes': 0, 'is_retweet': False, 'posted_time': '2023-05-14T20:56:03+00:00', 'content': 'Relatively light week until Powell speaks on Friday. Hopefully the $SPY does not stay trapped in this 404-417 range.\n\nThere is also a range within that range… first level of support is at 408.75 and first level of resistance is at 414.75.', 'hashtags': [], 'mentions': [], 'images': ['https://pbs.twimg.com/media/FwHf4BXWwAYuF9r?format=jpg&name=medium', 'https://pbs.twimg.com/media/FwHf4BTXoAAcos0?format=jpg&name=medium'], 'videos': [], 'tweet_url': 'https://twitter.com/davidcharting/status/1657852295558709248', 'link': ''}}
        #doc_ref = db.collection(u'recent_tweet')
        #postdata = data
        # Assumes any auth/headers you need are already taken care of.
        #result = firebase.post('/tweet', postdata, {'print': 'pretty'})
        for key, value in data.items():
            logging.info(f"key:{key}, value:{value}")
            db.collection(u'recent_tweet').document(key).set(value)
