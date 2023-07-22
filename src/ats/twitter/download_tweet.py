# importing libraries and packages
# from absl import logging
# Example of usage:
# PYTHONPATH=. python3 twitter/download_tweet.py --username=eliant_capital --since=2023-03-23 --until=2023-03-24
#
import logging
import datetime
import os

import pandas as pd

import snscrape.modules.twitter as sntwitter
from neo4j_util.neo4j_tweet_util import Neo4j
from neo4j_util import sentiment_api
from util import config_utils
from util import logging_utils


def upload_tweet_to_neo4j(args, since, until, existing_tweets):
    logging.info(f"existing_tweets:{len(existing_tweets)}")
    # Creating list to append tweet data
    # neo4j = Neo4j()
    since = since.date().isoformat()
    until = until.date().isoformat()
    if args.username:
        usernames = args.username.split(",")
        user = usernames[0]
        # query = f"from:{args.username} since:{since} until:{until}"
        symbol_output_dir = os.path.join(args.output_dir, user)
        output_file = (
            symbol_output_dir + "/" + user + "_" + since + "_" + until + ".csv"
        )
        logging.info(
            f"processing since:{since} until:{until}, output_file:{output_file}"
        )
        if not os.path.exists(output_file):
            tweet_urls = sntwitter.TwitterUserScraper(
                args.username, existing_tweets
            ).get_items()
            df = pd.DataFrame()
            df = [asdict(tweet) for tweet in tweet_urls]
            logging.info(f"df:{df}")
            if not os.path.exists(symbol_output_dir):
                os.makedirs(symbol_output_dir)
            df.to_csv(output_file)
        # neo4j.bulk_load(tweet_urls)
        # query = f"to:{args.username} since:{since} until:{until}"
        # tweet_urls = sntwitter.TwitterSearchScraper(query, existing_tweets).get_items()
        # tweet_urls = [t for t in tweet_urls if not t.id in existing_tweets]
        # neo4j.bulk_load(tweet_urls)


def upload_tweet_to_neo4j_by_ids(ids, existing_tweets):
    logging.info(f"existing_tweets:{len(existing_tweets)},")
    # Creating list to append tweet data
    neo4j = Neo4j()
    for id in ids:
        if not id in existing_tweets:
            tweet_urls = []
            tweet_urls.extend(
                sntwitter.TwitterTweetScraper(
                    id, existing_tweets, mode=sntwitter.TwitterTweetScraperMode.RECURSE
                ).get_items()
            )
            tweet_urls = [t for t in tweet_urls if not t.id in existing_tweets]
            logging.info(f"tweet_urls:{tweet_urls}")
            neo4j.bulk_load(tweet_urls)
            # Add new ids to existing_tweets so that we do not reprocess them
            new_tweet_ids = [t.id for t in tweet_urls]
            existing_tweets.update(new_tweet_ids)


if __name__ == "__main__":
    parser = config_utils.get_arg_parser("Scrape tweet")
    parser.add_argument("--username", type=str)
    parser.add_argument("--id_file", type=str)
    parser.add_argument("--hash_tag", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--stock", type=str)

    args = parser.parse_args()
    config_utils.set_args(args)
    logging_utils.init_logging()
    if not args.username and not args.hash_tag and not args.stock and not args.id_file:
        logging.error(
            "Must specify at least one of username, hash_Tag or stock or id_file"
        )
        exit(-1)

    if args.id_file:
        done_file = args.id_file + ".done"
        if not os.path.exists(done_file):
            ids = pd.read_csv(args.id_file)["tweet_id"]
            existing_tweets = set(
                sentiment_api.get_tweet_id_by_user(args.username, ids.min(), ids.max())
            )
            logging.info(f"existing_tweets:{existing_tweets}")
            filted_ids = [id for id in ids if not id in existing_tweets]
            upload_tweet_to_neo4j_by_ids(filted_ids, existing_tweets=existing_tweets)
            with open(done_file, "w") as fp:
                pass
            exit(0)

    end_date = args.end_date
    if end_date < args.start_date + datetime.timedelta(31):
        end_date = args.start_date + datetime.timedelta(31)
    # for cur_date in pd.date_range(args.start_date, end_date, freq="M"):
    # cur_start_date = cur_date-datetime.timedelta(2)
    # cur_end_date = cur_date + datetime.timedelta(days=32)
    cur_end_date = datetime.datetime.today()
    cur_start_date = cur_end_date - datetime.timedelta(days=180)
    # existing_tweets = set(sentiment_api.get_tweet_id_by_range(cur_start_date, cur_end_date))
    upload_tweet_to_neo4j(args, cur_start_date, cur_end_date, existing_tweets=())
