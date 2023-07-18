# Usage
# PYTHONPATH=. python3 twitter/nitter_symbol.py --output_dir=../data/nitter_tweet_ids --symbols=spy --start_date=2010-01-01 --end_date=2023-04-20
#
import dotenv
import os
import time
import datetime
import logging
from pprint import pprint
import twarc
import pandas as pd
from util import config_utils
from util import logging_utils
from twitter.tweet_util import get_last_tweet_id

if __name__ == "__main__":
    parser = config_utils.get_arg_parser("Scape tweet")
    parser.add_argument("--symbols", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument(
        "--start_date",
        type=lambda d: datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S").date(),
        required=False,
        help="Set a start date",
    )
    parser.add_argument(
        "--end_date",
        type=lambda d: datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S").date(),
        required=False,
        help="Set a end date",
    )

    args = parser.parse_args()
    config_utils.set_args(args)
    logging_utils.init_logging()
    dotenv.load_dotenv()
    consumer_key = os.environ.get("CONSUMER_KEY")
    consumer_secret = os.environ.get("CONSUMER_SECRET")
    bearer_token = os.environ.get("BEARER_TOKEN")
    access_token = os.environ.get("ACCESS_TOKEN")
    access_token_secret = os.environ.get("ACCESS_TOKEN_SECRET")

    start_date = args.start_date
    end_date = args.end_date
    if not end_date:
        end_date = datetime.datetime.now()
    if not start_date:
        start_date = end_date - datetime.timedelta(seconds=3600)
    # Implicitly test the constructor in application auth mode. This ensures that
    # the tests don't depend on test ordering, and allows using the pytest
    # functionality to only run a single test at a time.

    T = twarc.Twarc2(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
    )
    for symbol in args.symbols.split(","):
        symbol_output_dir = os.path.join(args.output_dir, symbol)
        last_tweet_id = get_last_tweet_id(symbol_output_dir)
        since = start_date.strftime("%Y%m%d%H%M%S")
        until = end_date.strftime("%Y%m%d%H%M%S")
        output_file = symbol_output_dir + "/" + symbol + "_" + since + "_" + until + ".csv"
        logging.info(f"processing since:{since} until:{until}, output_file:{output_file}, last_tweet_id:{last_tweet_id}")
        if not os.path.exists(output_file):                    
            df = pd.DataFrame()
            try:
                for response_page in T.search_recent(f"{symbol}", start_time=start_date, end_time=end_date, since_id=last_tweet_id):
                    #logging.info(f"response_page:{response_page}")
                    for tweet in response_page["data"]:
                        logging.info(f"tweet:{tweet}")
                        # convert created_at to datetime with utc timezone
                        #dt = tweet["created_at"].strip("Z")
                        #dt = datetime.datetime.fromisoformat(dt)
                        #dt = dt.replace(tzinfo=datetime.timezone.utc)
                        #df2 = {'Id': str(tweet.tweet_id), 'Url': tweet.tweet_url, 'Username': tweet.username}
                        try:
                            df2 = tweet
                            df2["tweet_id"] = str(df2["id"])
                            df = df.append(df2, ignore_index = True)
                        except Exception as e:
                            logging.info(f"e:{e}")
                            pass
            except Exception as e:
                logging.info(f"e:{e}")
                pass
            logging.info(f"df:{df}")
            if not os.path.exists(symbol_output_dir):
                os.makedirs(symbol_output_dir)
            df.to_csv(output_file)
