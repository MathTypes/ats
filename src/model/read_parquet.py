import sys
import pandas as pd
from util import config_utils
from util import logging_utils

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    parser = config_utils.get_arg_parser("Read parquet files")
    parser.add_argument("--input", type=str)
    parser.add_argument("--min_time", type=str)
    parser.add_argument("--max_time", type=str)
    args = parser.parse_args()
    config_utils.set_args(args)
    logging_utils.init_logging()

    df = pd.read_parquet(args.input)
    if args.min_time and args.max_time:
        df = df.loc[(df.time>args.min_time) & (df.time<args.max_time)]
    print(df)
