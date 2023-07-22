# Usage
#
# /Users/jianjunchen/ats/lib/python3.7/site-packages/nitter_scraper/search.py
import datetime
import logging
import ipaddress

from nitter_scraper import NitterScraper
from util import config_utils
from util import logging_utils
from firebase_admin import firestore
from util import logging_utils

import firebase_admin
from firebase_admin import credentials
from firebase import firebase


def read_collection(db, query):
    doc_ref = db.collection("unique_ids").document(query)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    else:
        return None


# borrowed from https://stackoverflow.com/a/13565185
# as noted there, the calendar module has a function of its own
def last_day_of_month(any_day):
    next_month = any_day.replace(day=28) + datetime.timedelta(
        days=4
    )  # this will never fail
    return next_month - datetime.timedelta(days=next_month.day)


if __name__ == "__main__":
    parser = config_utils.get_arg_parser("Scape tweet with nitter")
    parser.add_argument("--users", type=str)
    parser.add_argument("--host", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--existing", type=bool, default=False)
    parser.add_argument("--since_id", type=str)
    parser.add_argument("--max_id", type=str)
    parser.add_argument("--firebase_cert", type=str)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--port", type=int, default=8008)

    args = parser.parse_args()
    config_utils.set_args(args)
    logging_utils.init_logging()
    cred = credentials.Certificate(args.firebase_cert)
    default_app = firebase_admin.initialize_app(cred)
    db = firestore.client()
    firebase = firebase.FirebaseApplication(
        "https://keen-rhino-386415.firebaseio.com", None
    )

    users = args.users.split(",")
    cur_date = datetime.datetime.today()
    with NitterScraper(
        host=ipaddress.ip_address(args.host),
        port=args.port,
        existing_instance=args.existing,
    ) as nitter:
        for user in users:
            if not user:
                continue
            try:
                id_key = f"nitter_recent_user_{user}"
                df = read_collection(db, id_key)
                since_id = None
                max_id = None
                if df:
                    since_id = df["last_id"]
                if args.since_id:
                    since_id = args.since_id
                if args.max_id:
                    max_id = args.max_id
                logging.info(f"search:{user}, since_id:{since_id}, max_id:{max_id}")
                last_tweet_id = int(since_id)
                logging.info(f"last_tweet_id:{last_tweet_id}")
                tweets = nitter.get_tweets(
                    user,
                    pages=args.rows,
                    break_on_tweet_id=last_tweet_id,
                    address="https://nitter.it",
                )
                for tweet in tweets:
                    # df2 = {'tweet_id': str(tweet.tweet_id), 'Url': tweet.tweet_url, 'Username': tweet.username}
                    db.collection("recent_tweet_by_user").document(
                        str(tweet.tweet_id)
                    ).set(tweet.dict())
                    if not last_tweet_id or tweet.tweet_id > last_tweet_id:
                        last_tweet_id = tweet.tweet_id
                    logging.info(f"Adding:{tweet}")
                db.collection("unique_ids").document(id_key).set(
                    {"last_id": last_tweet_id}
                )
            except Exception as e:
                logging.info(f"Exception:{e}")
