# importing libraries and packages
# from absl import logging
# Example of usage:
# PYTHONPATH=. python3 twitter/download_tweet.py --username=eliant_capital --since=2023-03-23 --until=2023-03-24
#
import argparse
import logging
import datetime
import os

import pandas as pd

import snscrape.modules.twitter as sntwitter
from neo4j_util.neo4j_tweet_util import Neo4j
from neo4j_util.sentiment_api import get_tweet_id_by_range
from util import logging_utils

def upload_tweet_to_neo4j(username, since, until, existing_tweets):
    if until == '':
        until = datetime.datetime.strftime(datetime.date.today(), '%Y-%m-%d')
    if since == '':
        since = datetime.datetime.strftime(datetime.datetime.strptime(
            until, '%Y-%m-%d') - datetime.timedelta(days=7), '%Y-%m-%d')

    logging.info(f'existing_tweets:{len(existing_tweets)}')
    empty_tweets = set()
    # Creating list to append tweet data
    tweets_list1 = []
    if args.username:
        usernames = args.username.split(",")
        neo4j = Neo4j()
        query = f"from:{args.username} since:{since} until:{until}"
        tweet_urls = sntwitter.TwitterSearchScraper(query, empty_tweets).get_items()
        tweet_urls = [ t for t in tweet_urls if not t.id in existing_tweets]
        neo4j.bulk_load(tweet_urls)
        query = f"to:{args.username} since:{since} until:{until}"
        tweet_urls = sntwitter.TwitterSearchScraper(query, empty_tweets).get_items()
        tweet_urls = [ t for t in tweet_urls if not t.id in existing_tweets]
        neo4j.bulk_load(tweet_urls)

    if args.hash_tag:
        hash_tags = args.hash_tag.split(",")
        for hash_tag in hash_tags:
            query = f"f#{hash_tag} since:{since} until:{until}"
            tweet_urls = sntwitter.TwitterSearchScraper(query, empty_tweets).get_items()
            tweet_urls = [ t for t in tweet_urls if not t.id in existing_tweets]
            neo4j.bulk_load(tweet_urls)

    if args.stock:
        stocks = args.stock.split(",")
        for stock in stocks:
            query = f"f${stock} since:{since} until:{until}"
            tweet_urls = sntwitter.TwitterSearchScraper(query, empty_tweets).get_items()
            tweet_urls = [ t for t in tweet_urls if not t.id in existing_tweets]
            neo4j.bulk_load(tweet_urls)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='A test script for http://stackoverflow.com/q/14097061/78845'
    )
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    parser.add_argument("--username", type=str)
    parser.add_argument("--hash_tag", type=str)
    parser.add_argument("--stock", type=str)
    parser.add_argument("--since", type=str, required=True)
    parser.add_argument("--until", type=str, required=True)

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    logging_utils.init_logging()
    if not args.username and not args.hash_tag and not args.stock:
        logging.error("Must specify at least one of username, hash_Tag or stock")
        exit(-1)
    
    existing_tweets = set(get_tweet_id_by_range(args.since, args.until))
    upload_tweet_to_neo4j(args.username, args.since, args.until, existing_tweets)
