import logging
import os
import sys
import argparse
import pandas as pd

from keyword_util import add_subject_keyword
from neo4j_util.sentiment_api import get_unprocessed_tweets, get_tweet_replies_v2
from neo4j_util.neo4j_tweet_util import Neo4j
from util import logging_utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='A test script for http://stackoverflow.com/q/14097061/78845'
    )
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    logging_utils.init_logging()
    while True:
        data = get_unprocessed_tweets()
        logging.error(f'unprocess_data:{data}')
        if data.empty:
            break
        data = add_subject_keyword(data)
        logging.error(f'process_data:{data}')
        neo4j_util = Neo4j()
        neo4j_util.update_processed_text(data)

    #conv_data = get_tweet_replies_v2()
    #conv_data = add_subject_keyword(conv_data)


