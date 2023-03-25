# importing libraries and packages
# from absl import logging
# Example of usage:
# PYTHONPATH=. python3 ../scripts/download_tweet.py --username=eliant_capital --output_dir=../data
#
import argparse
import logging
import datetime
import os

import pandas as pd

import snscrape.modules.twitter as sntwitter
from neo4j_util.neo4j_tweet_util import Neo4j


def upload_tweet_to_neo4j(username, since, until):
    if until == '':
        until = datetime.datetime.strftime(datetime.date.today(), '%Y-%m-%d')
    if since == '':
        since = datetime.datetime.strftime(datetime.datetime.strptime(
            until, '%Y-%m-%d') - datetime.timedelta(days=7), '%Y-%m-%d')

    # Creating list to append tweet data
    tweets_list1 = []
    usernames = args.username.split(",")
    query = f"from:{args.username} since:{since} until:{until}"
    neo4j = Neo4j()
    tweets = sntwitter.TwitterSearchScraper(query).get_items()
    neo4j.bulk_load(tweets)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='A test script for http://stackoverflow.com/q/14097061/78845'
    )
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    parser.add_argument("--username", type=str, required=True)
    parser.add_argument("--since", type=str, required=True)
    parser.add_argument("--until", type=str, required=True)

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    upload_tweet_to_neo4j(args.username, args.since, args.until)
