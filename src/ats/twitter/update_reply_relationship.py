import datetime
import logging
import os
import sys
import argparse
import pandas as pd

from nlp.keyword_util import add_subject_keyword
from neo4j_util import sentiment_api
from neo4j_util.neo4j_tweet_util import Neo4j
from util import config_utils
from util import logging_utils
from data.front_end_utils import subject_analysis

import re


if __name__ == "__main__":
    parser = config_utils.get_arg_parser("Preprocess tweet")
    parser.add_argument(
        "--start_date",
        type=lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").date(),
        required=True,
        help="Set a start date",
    )
    parser.add_argument(
        "--end_date",
        type=lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").date(),
        required=True,
        help="Set a end date",
    )

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    config_utils.set_args(args)
    logging_utils.init_logging()
    while True:
        data = sentiment_api.update_tweets_unprocessed_for_reply(
            args.start_date, args.end_date
        )
        logging.info(f'replied_ids:{data["reply_tweet_id"]}')
        if data.empty:
            break
        # break
    # conv_data = get_tweet_replies_v2()
    # conv_data = add_subject_keyword(conv_data)
