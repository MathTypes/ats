import datetime
import logging
import os
import sys
import argparse
import pandas as pd

from util import config_utils
from util import logging_utils

import re

if __name__ == "__main__":
    parser = config_utils.get_arg_parser("Read file")
    parser.add_argument(
        "--file",
        type=str,
        help="File to read",
    )

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    config_utils.set_args(args)
    logging_utils.init_logging()
    pd.set_option("display.max_columns", 100)
    df = pd.read_parquet(args.file, engine='fastparquet')
    print(df)