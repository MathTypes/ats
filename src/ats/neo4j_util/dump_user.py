##
# Usage: PYTHONPATH=. python3 neo4j_util/dump_user.py --output_file=../data/tweet_users.csv --neo4j_host=bolt://34.105.8.79:7687 --neo4j_pass=Tq7ks8zY
#
import logging
import string
import pandas as pd

from neo4j_util import sentiment_api
from util import config_utils
from util import logging_utils


def get_users(prefix):
    users = sentiment_api.read_query(
        "MATCH (t:Tweet) WHERE toLower(left(t.user, 1))=$prefix RETURN t.user, count(*) as cnt",
        {"prefix": prefix},
    )
    return users


if __name__ == "__main__":
    parser = config_utils.get_arg_parser("Scrape tweet users")
    parser.add_argument("--output_file", type=str)

    args = parser.parse_args()
    config_utils.set_args(args)
    logging_utils.init_logging()
    df_vec = []
    for prefix in string.ascii_lowercase:
        df = get_users(prefix)
        logging.info(f"df:{df}")
        df_vec.extend(df)
    users = pd.DataFrame(df_vec, columns=["username"])
    users.to_csv(args.output_file)
