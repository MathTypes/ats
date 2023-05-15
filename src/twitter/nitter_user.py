# Usage
#
# /Users/jianjunchen/ats/lib/python3.7/site-packages/nitter_scraper/search.py
import os
import time
import datetime
import logging
from pprint import pprint
from filelock import Timeout, FileLock

import pandas as pd
from nitter_scraper import NitterScraper
from util import config_utils
from util import logging_utils
from twitter.tweet_util import get_last_tweet_id

# borrowed from https://stackoverflow.com/a/13565185
# as noted there, the calendar module has a function of its own
def last_day_of_month(any_day):
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)  # this will never fail
    return next_month - datetime.timedelta(days=next_month.day)

def monthlist(begin,end):
    begin = datetime.datetime.strptime(begin, "%Y-%m-%d")
    end = datetime.datetime.strptime(end, "%Y-%m-%d")

    result = []
    while True:
        if begin.month == 12:
            next_month = begin.replace(year=begin.year+1,month=1, day=1)
        else:
            next_month = begin.replace(month=begin.month+1, day=1)
        if next_month > end:
            break
        result.append ([begin.strftime("%Y-%m-%d"),last_day_of_month(begin).strftime("%Y-%m-%d")])
        begin = next_month
    result.append ([begin.strftime("%Y-%m-%d"),end.strftime("%Y-%m-%d")])
    return result

if __name__ == "__main__":
    parser = config_utils.get_arg_parser("Scape tweet")
    parser.add_argument("--users", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--existing", type=bool, default=False)
    parser.add_argument("--port", type=int, default=8008)

    args = parser.parse_args()
    config_utils.set_args(args)
    logging_utils.init_logging()

    users = args.users.split(",")
    cur_date = datetime.datetime.today()
    with NitterScraper(host="0.0.0.0", port=args.port, existing_instance=args.existing) as nitter:
        for user in users:
            if not user:
                continue
            try:
                symbol_output_dir = os.path.join(args.output_dir, user)
                if not os.path.exists(symbol_output_dir):
                    os.makedirs(symbol_output_dir)
                last_tweet_id = get_last_tweet_id(symbol_output_dir)
                logging.info(f'last_tweet_id:{last_tweet_id}')
                output_file = symbol_output_dir + "/" + user + "_" + cur_date.strftime("%Y-%m-%d") + ".csv"
                if not os.path.exists(output_file):
                    lock = FileLock(f"{output_file}.lock")
                    with lock:
                        df = pd.DataFrame()
                        tweets = nitter.get_tweets(user, pages=10000, break_on_tweet_id=last_tweet_id, address="https://nitter.it")
                        for tweet in tweets:
                            df2 = {'tweet_id': str(tweet.tweet_id), 'Url': tweet.tweet_url, 'Username': tweet.username}
                            df = df.append(df2, ignore_index = True)
                        logging.info(f"df:{df}")
                        df.to_csv(output_file)
                time.sleep(0.5)
            except Exception as e:
                logging.info(f"Exception:{e}")
