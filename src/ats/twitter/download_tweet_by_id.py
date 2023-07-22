# importing libraries and packages
# from absl import logging
# Example of usage:
# PYTHONPATH=. python3 twitter/download_tweet.py --username=eliant_capital --since=2023-03-23 --until=2023-03-24
#
import logging
from dataclasses import asdict
from filelock import FileLock
import os

import pandas as pd

import snscrape.modules.twitter as sntwitter
from util import config_utils
from util import logging_utils


def upload_tweet_to_neo4j_by_ids(ids, existing_tweets):
    logging.info(f"existing_tweets:{len(existing_tweets)},")
    tweet_urls = []
    for id in ids:
        tweet_urls.extend(
            sntwitter.TwitterTweetScraper(
                id, existing_tweets, mode=sntwitter.TwitterTweetScraperMode.RECURSE
            ).get_items()
        )
    df = [asdict(tweet) for tweet in tweet_urls]
    return pd.DataFrame.from_dict(df)


if __name__ == "__main__":
    parser = config_utils.get_arg_parser("Scrape tweet by id")
    parser.add_argument("--id_file", type=str)

    args = parser.parse_args()
    config_utils.set_args(args)
    logging_utils.init_logging()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    if args.id_file:
        logging.info(f"Using idfile:{args.id_file}")
        done_file = args.id_file + ".done"
        output_file = args.id_file + ".reply.csv"
        if not os.path.exists(output_file):
            lock = FileLock(f"{output_file}.lock")
            with lock:
                existing_tweets = set()
                id_df = pd.read_csv(args.id_file)
                if "tweet_id" in id_df.columns:
                    ids = id_df["tweet_id"]
                else:
                    ids = id_df["Id"]
                df = upload_tweet_to_neo4j_by_ids(ids, existing_tweets=existing_tweets)
                # df = df.dropna()
                logging.info(f"Writing: {output_file}")
                logging.info(f"Writing: df:{df}")
                # df["date"] = pd.to_datetime(df["date"], errors="coerce")
                # df['timestamp'] = df['date'].dt.timestamp()
                # df.to_parquet(output_file)
                df.to_csv(output_file, sep="`")
                with open(done_file, "w") as fp:
                    pass
                exit(0)
