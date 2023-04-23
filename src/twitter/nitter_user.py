import os
import time
import datetime
import logging
from pprint import pprint

import pandas as pd
from nitter_scraper import NitterScraper
from util import config_utils
from util import logging_utils

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
    parser.add_argument("--port", type=int, default=8008)
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

    last_tweet_id = None
    users = args.users.split(",")
    df_vec = []
    with NitterScraper(host="0.0.0.0", port=args.port) as nitter:
        for symbol in users:
            for cur_date in pd.date_range(args.start_date, args.end_date, freq="D"):
                since = cur_date.strftime("%Y-%m-%d")
                until = (cur_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                df = pd.DataFrame()
                query = f"search?f=tweets&q=from%3A{symbol}&until={until}&since={since}"
                try:
                    tweets = nitter.search_tweets(query, pages=100)
                    for tweet in tweets:
                        #df2 = {'Id': str(tweet.tweet_id), 'Url': tweet.tweet_url, 'Username': tweet.username}
                        df2 = tweet.dict()
                        df2["tweet_id"] = str(df2["tweet_id"])
                        df = df.append(df2, ignore_index = True)
                except Exception as e:
                    logging.info(f"e:{e}")
                    pass
                logging.info(f"df:{df}")
                symbol_output_dir = os.path.join(args.output_dir, symbol)
                if not os.path.exists(symbol_output_dir):
                    os.mkdir(symbol_output_dir)
                output_file = symbol_output_dir + "/" + symbol + "_" + since + "_" + until + ".csv"
                if not os.path.exists(output_file):
                    df.to_csv(output_file)
