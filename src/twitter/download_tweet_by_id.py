# importing libraries and packages
# from absl import logging
# Example of usage:
# PYTHONPATH=. python3 twitter/download_tweet.py --username=eliant_capital --since=2023-03-23 --until=2023-03-24
#
import argparse
import logging
import datetime
from dataclasses import asdict 
import os

import pandas as pd

import snscrape.modules.twitter as sntwitter
from neo4j_util import sentiment_api
from util import config_utils
from util import logging_utils

def upload_tweet_to_neo4j_by_ids(ids, existing_tweets):
    logging.info(f"existing_tweets:{len(existing_tweets)},")
    empty_tweets = set()
    tweet_urls = []
    for id in ids:
        tweet_urls.extend(sntwitter.TwitterTweetScraper(id, existing_tweets,
            mode=sntwitter.TwitterTweetScraperMode.RECURSE).get_items())
    df = [asdict(tweet) for tweet in tweet_urls]
    return pd.DataFrame.from_dict(df)

if __name__ == "__main__":
    parser = config_utils.get_arg_parser("Scrape tweet by id")
    parser.add_argument("--id_file", type=str)

    args = parser.parse_args()
    config_utils.set_args(args)
    logging_utils.init_logging()
    
    if args.id_file:
        done_file = args.id_file + ".done"
        if not os.path.exists(done_file):
            existing_tweets = set()
            ids = pd.read_csv(args.id_file)["tweet_id"]
            df = upload_tweet_to_neo4j_by_ids(ids, existing_tweets = existing_tweets)
            output_file = args.id_file + ".reply.parquet"
            logging.info(f"df:{df}")
            df.to_parquet(output_file)
            #df.to_csv(output_file, sep = '`')
            with open(done_file, 'w') as fp:
                pass
            exit(0)
