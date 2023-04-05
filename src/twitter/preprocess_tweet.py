import logging
import os
import sys
import argparse
import pandas as pd

from nlp.keyword_util import add_subject_keyword
from neo4j_util.sentiment_api import get_unprocessed_tweets, get_tweet_replies_v2
from neo4j_util.neo4j_tweet_util import Neo4j
from util import config_utils
from util import logging_utils
from data.front_end_utils import (
    subject_analysis
)

if __name__ == "__main__":
    parser = config_utils.get_arg_parser("Preprocess tweet")

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    config_utils.set_args(args)
    logging_utils.init_logging()
    while True:
        data = get_unprocessed_tweets()
        logging.info(f'unprocess_data:{data["tweet_id"]}')
        if data.empty:
            break
        data = add_subject_keyword(data)
        data = subject_analysis(data)
        #logging.error(f'process_data:{data}')
        count = 0
        RETRIES = 5
        while count < RETRIES:
            try:
                neo4j_util = Neo4j()
                neo4j_util.update_processed_text(data)
                break
            except Exception as e:
                logging.error(f"caught exception: {e}")
            count = count + 1
        #break
    #conv_data = get_tweet_replies_v2()
    #conv_data = add_subject_keyword(conv_data)


