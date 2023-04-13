import argparse
import datetime
from dateutil.parser import parse, ParserError
import functools
import logging
import os
import time
import traceback

import pandas as pd

from util import config_utils
from util import logging_utils
from neo4j_util import sentiment_api

class DateParser(argparse.Action):
    def __call__(self, parser, namespace, values, option_strings=None):
        setattr(namespace, self.dest, parse(values).date())

if __name__ == "__main__":
    parser = config_utils.get_arg_parser("Preprocess tweet")
    parser.add_argument("--start_date",
        type=lambda d: datetime.datetime.strptime(d, '%Y-%m-%d').date(),
        required=True,
        help='Set a start date')
    parser.add_argument("--end_date",
        type=lambda d: datetime.datetime.strptime(d, '%Y-%m-%d').date(),
        required=True,
        help='Set a end date')

    args = parser.parse_args()
    config_utils.set_args(args)
    logging_utils.init_logging()
    news_df = sentiment_api.get_processed_tweets(args.start_date, args.end_date)
    logging.info(f'price_df:{news_df}')
    for n, g in news_df.groupby(pd.Grouper(freq='M')):
        logging.info(f'n:{n}')
        logging.info(f'g:{g}')
        path_dir = os.path.join(config_utils.get_ts_root(), "monthly", "twitter")
        os.makedirs(path_dir, exist_ok=True)
        month_file = os.path.join(path_dir,
                             n.strftime("%Y%m") + '.parquet')
        logging.info(f'writing:{month_file}')
        g.to_parquet(month_file)
    
