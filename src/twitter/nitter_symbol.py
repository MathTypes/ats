# Usage
# PYTHONPATH=. python3 twitter/nitter_symbol.py --output_dir=../data/nitter_tweet_ids --symbols=spy --start_date=2010-01-01 --end_date=2023-04-20
#
import os
import time
import datetime
import logging
from pprint import pprint

import pandas as pd
from nitter_scraper import NitterScraper
from util import config_utils
from util import logging_utils

if __name__ == "__main__":
    parser = config_utils.get_arg_parser("Scape tweet")
    parser.add_argument("--symbols", type=str)
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

    last_tweet_id = None
    users = args.symbols.split(",")
    df_vec = []
    with NitterScraper(host="0.0.0.0", port=8008) as nitter:
        for symbol in users:
            for cur_date in pd.date_range(args.start_date, args.end_date, freq="D"):
                since = cur_date.isoformat()
                until = (cur_date + datetime.timedelta(days=1)).isoformat()
                df = pd.DataFrame()
                query = f"search?f=tweets&q=${symbol}&until={until}&since={since}"
                try:
                    tweets = nitter.search_tweets(query, pages=100)
                    for tweet in tweets:
                        #df2 = {'Id': str(tweet.tweet_id), 'Url': tweet.tweet_url, 'Username': tweet.username}
                        df2 = tweet.dict()
                        df2["tweet_id"] = str(df2["tweet_id"])
                        df2["timestamp_ms"] = tweet.time.timestamp()*1000
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
