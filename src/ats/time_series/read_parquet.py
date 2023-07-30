import pandas as pd

from ats.util import config_utils
from ats.util import logging_utils

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    parser = config_utils.get_arg_parser("Read parquet files")
    parser.add_argument("--input", type=str)
    parser.add_argument("--columns", type=str)
    parser.add_argument("--min_time", type=str)
    parser.add_argument("--max_time", type=str)
    parser.add_argument("--ticker", type=str)
    parser.add_argument("--nrows", type=int)
    args = parser.parse_args()
    config_utils.set_args(args)
    logging_utils.init_logging()

    df = pd.read_parquet(args.input)
    if args.min_time and args.max_time:
        df = df.loc[(df.time > args.min_time) & (df.time < args.max_time)]
    if args.columns:
        df = df[args.columns]
    if args.ticker:
        df = df[df.ticker==args.ticker]
    if args.nrows:
        if args.nrows>0:
            df = df.iloc[:args.nrows]
        else:
            df = df.iloc[args.nrows:]
    print(df)
