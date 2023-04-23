import time
import datetime
import logging
from pprint import pprint

import pandas as pd
from nitter_scraper import NitterScraper
from util import config_utils
from util import logging_utils

if __name__ == "__main__":
    parser = config_utils.get_arg_parser("Scape tweet")
    parser.add_argument("--username", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument(
        "--start_date",
        type=lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").date(),
        required=False,
        help="Set a start date",
    )
    parser.add_argument(
        "--end_date",
        type=lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").date(),
        required=False,
        help="Set a end date",
    )

    args = parser.parse_args()
    config_utils.set_args(args)
    logging_utils.init_logging()

    last_tweet_id = None
    users = args.username.split(",")
    with NitterScraper(host="0.0.0.0", port=8012) as nitter:
        profile = nitter.get_profile("dgnsrekt")
        print("serialize to json\n")
        print(profile.json(indent=4))
        print("serialize to a dictionary\n")
        pprint(profile.dict())
    exit(0)

    df_vec = []
    with NitterScraper(host="0.0.0.0", port=8008) as nitter:
        for user in users:
            df = pd.DataFrame()
            for cur_date in pd.date_range(args.start_date, args.end_date, freq="D"):
                since = cur_date.isoformat()
                until = (cur_date + datetime.timedelta(days=1)).isoformat()
                query = f"search?f=tweets&q={user}&until={until}&since={since}"
                try:
                    tweets = nitter.search_tweets(query, pages=100)
                    for tweet in tweets:
                        df2 = {'Id': str(tweet.tweet_id), 'Url': tweet.tweet_url, 'Username': tweet.username}
                        df = df.append(df2, ignore_index = True)
                except Exception as e:
                    logging.info(f"e:{e}")
                    pass
            logging.info(f"df:{df}")
            df.to_csv(args.output_dir + "/" + args.username + ".csv")
    exit(0)

    with NitterScraper(host="0.0.0.0", port=8008) as nitter:
        for user in users:
            df = pd.DataFrame()
            tweets = nitter.get_tweets(user, pages=100)
            for tweet in tweets:
                df2 = {'Id': str(tweet.tweet_id), 'Url': tweet.tweet_url, 'Username': tweet.username}
                df = df.append(df2, ignore_index = True)
            logging.info(f"df:{df}")
            df.to_csv(args.output_dir + "/" + args.username + ".csv")
    exit(0)

    print("Scraping with local nitter docker instance.")

    with NitterScraper(host="0.0.0.0", port=8008) as nitter:
        while True:
            #https://nitter.net/search?f=tweets&q=DougKass&until=2012-01-03&since=2012-01-02
            for tweet in nitter.get_tweets(
                "DougKass",
                pages=1, break_on_tweet_id=last_tweet_id):
                if tweet.is_pinned is True:
                    continue
                if tweet.is_retweet is True:
                    continue
                if tweet.tweet_id != last_tweet_id:
                    print(tweet.json(indent=4))
                last_tweet_id = tweet.tweet_id
                break

            time.sleep(0.1)
    exit(0)

    with NitterScraper(host="0.0.0.0", port=8008) as nitter:
        while True:
            #https://nitter.net/search?f=tweets&q=DougKass&until=2012-01-03&since=2012-01-02
            for tweet in nitter.search_tweets(
                "search?f=tweets&q=DougKass&until=2012-01-03&since=2012-01-02",
                pages=1, break_on_tweet_id=last_tweet_id, address="https://nitter.net"):
                if tweet.is_pinned is True:
                    continue
                if tweet.is_retweet is True:
                    continue
                if tweet.tweet_id != last_tweet_id:
                    print(tweet.json(indent=4))
                last_tweet_id = tweet.tweet_id
                break

            time.sleep(0.1)
    exit(0)

