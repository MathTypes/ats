import datetime
import logging
import pandas as pd

import firebase_api
from util import config_utils
from util import logging_utils

import re


def get_cash_tags(text):
    pattern = "\$([a-zA-Z.]+)\W"
    result = re.findall(pattern, text)
    # logging.info(f"result:{result}")
    return result


def find_symbol(x):
    x = str(x).lower()
    # logging.info(f"x:{x}")
    m = re.search("\$([a-z]{1,4})\W+", x)
    # logging.info(f"m:{m}")
    if m:
        return m.group(1)
    else:
        return ""


def is_power_user(username):
    return username in [
        "PharmD_KS",
        "pghosh1",
        "DougKass",
        "JimScalpert",
        "eliant_capital",
        "Mayhem4Markets",
        "Tricky_OW",
        "DoubleWideCap",
    ]


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
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
    parser.add_argument("--update", help="reprocess existing ones", action="store_true")

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    config_utils.set_args(args)
    logging_utils.init_logging()
    while True:
        data = firebase_api.get_processed_tweets_by_asset(
            "recent_tweet_by_user", args.start_date, args.end_date, args.update
        )
        if data.empty:
            break
        data["power_user"] = data.username.apply(is_power_user)
        data = data[data.power_user]
        if data.empty:
            break
        # logging.info(f'unprocess_data:{data["tweet_id"]}')
        logging.info(f"{data}")
        logging.info(f"{data.info()}")
        firebase_api.update_asset_sentiment(data)
        break
        # logging.error(f"process_data:{data}")
        # break
        # break
    # conv_data = get_tweet_replies_v2()
    # conv_data = add_subject_keyword(conv_data)
