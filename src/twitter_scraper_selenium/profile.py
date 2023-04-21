#!/usr/bin/env python3

from typing import Union
from .driver_initialization import Initializer
from .driver_utils import Utilities
from .element_finder import Finder
import re
import json
import csv
import os
import logging

logger = logging.getLogger(__name__)
format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(format)
logger.addHandler(ch)


class Profile:
    """this class needs to be instantiated in orer to scrape post of some
    twitter profile"""

    def __init__(
        self, twitter_username, browser, proxy, tweets_count, headless, browser_profile
    ):
        self.twitter_username = twitter_username
        self.URL = "https://twitter.com/{}".format(twitter_username.lower())
        self._driver = ""
        self.browser = browser
        self.proxy = proxy
        self.tweets_count = tweets_count
        self.posts_data = {}
        self.retry = 10
        self.headless = headless
        self.browser_profile = browser_profile

    def _start_driver(self):
        """changes the class member _driver value to driver on call"""
        self._driver = Initializer(
            self.browser, self.headless, self.proxy, self.browser_profile
        ).init()
        self._driver.maximize_window()  # For maximizing window
        # gives an implicit wait for 20 second
        self._driver.implicitly_wait(20)

    def _close_driver(self):
        self._driver.close()
        self._driver.quit()

    def _check_tweets_presence(self, tweet_list):
        if len(tweet_list) <= 0:
            self.retry -= 1

    def _check_retry(self):
        return self.retry <= 0

    def _fetch_and_store_data(self):
        try:
            all_ready_fetched_posts = []
            present_tweets = Finder.find_all_tweets(self._driver)
            self._check_tweets_presence(present_tweets)
            all_ready_fetched_posts.extend(present_tweets)

            while len(self.posts_data) < self.tweets_count:
                logging.error("tweets:{len(self.posts_data))}")
                for tweet in present_tweets:
                    logging.error(f"find tweet:{tweet}")
                    status, tweet_url = Finder.find_status(tweet)
                    logging.error(f"find status:{status}, url:{tweet_url}")
                    replies = Finder.find_replies(tweet)
                    retweets = Finder.find_shares(tweet)
                    status = status[-1]
                    username = tweet_url.split("/")[3]
                    is_retweet = (
                        True
                        if self.twitter_username.lower() != username.lower()
                        else False
                    )
                    name = Finder.find_name_from_tweet(tweet, is_retweet)
                    retweet_link = tweet_url if is_retweet is True else ""
                    posted_time = Finder.find_timestamp(tweet)
                    content = Finder.find_content(tweet)
                    likes = Finder.find_like(tweet)
                    images = Finder.find_images(tweet)
                    videos = Finder.find_videos(tweet)
                    hashtags = re.findall(r"#(\w+)", content)
                    mentions = re.findall(r"@(\w+)", content)
                    profile_picture = Finder.find_profile_image_link(tweet)
                    link = Finder.find_external_link(tweet)
                    logging.error(f"find link:{link}")
                    self.posts_data[status] = {
                        "tweet_id": status,
                        "username": username,
                        "name": name,
                        "profile_picture": profile_picture,
                        "replies": replies,
                        "retweets": retweets,
                        "likes": likes,
                        "is_retweet": is_retweet,
                        "retweet_link": retweet_link,
                        "posted_time": posted_time,
                        "content": content,
                        "hashtags": hashtags,
                        "mentions": mentions,
                        "images": images,
                        "videos": videos,
                        "tweet_url": tweet_url,
                        "link": link,
                    }

                Utilities.scroll_down(self._driver)
                logging.error("before scroll down")
                Utilities.wait_until_completion(self._driver)
                logging.error("after scroll down")
                Utilities.wait_until_tweets_appear(self._driver)
                logging.error("after tweet appear")
                present_tweets = Finder.find_all_tweets(self._driver)
                present_tweets = [
                    post
                    for post in present_tweets
                    if post not in all_ready_fetched_posts
                ]
                self._check_tweets_presence(present_tweets)
                all_ready_fetched_posts.extend(present_tweets)
                logging.error(f"check_retry:{self._check_retry()}")
                if self._check_retry() is True:
                    break

        except Exception as ex:
            logger.exception("Error at method fetch_and_store_data : {}".format(ex))

    def scrap(self):
        try:
            logging.error(f"scrap")
            self._start_driver()
            logging.error(f"after start_driver")
            self._driver.get(self.URL)
            logging.error(f"after get url")
            Utilities.wait_until_completion(self._driver)
            Utilities.wait_until_tweets_appear(self._driver)
            logging.error("tweet appear")
            self._fetch_and_store_data()
            logging.error("after fetch and store")
            self._close_driver()
            data = dict(list(self.posts_data.items())[0 : int(self.tweets_count)])
            logging.error("close driver")
            return data
        except Exception as ex:
            logger.exception("Error at method scrap : {} ".format(ex))
            self._close_driver()


def json_to_csv(filename, json_data, directory):
    os.chdir(directory)  # change working directory to given directory
    # headers of the CSV file
    fieldnames = [
        "tweet_id",
        "username",
        "name",
        "profile_picture",
        "replies",
        "retweets",
        "likes",
        "is_retweet",
        "retweet_link",
        "posted_time",
        "content",
        "hashtags",
        "mentions",
        "images",
        "videos",
        "tweet_url",
        "link",
    ]
    mode = "w"
    if os.path.exists("{}.csv".format(filename)):
        mode = "a"
    # open and start writing to CSV files
    with open(
        "{}.csv".format(filename), mode, newline="", encoding="utf-8"
    ) as data_file:
        # instantiate DictWriter for writing CSV fi
        writer = csv.DictWriter(data_file, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()  # write headers to CSV file
        # iterate over entire dictionary, write each posts as a row to CSV file
        for key in json_data:
            # parse post in a dictionary and write it as a single row
            row = {
                "tweet_id": key,
                "username": json_data[key]["username"],
                "name": json_data[key]["name"],
                "profile_picture": json_data[key]["profile_picture"],
                "replies": json_data[key]["replies"],
                "retweets": json_data[key]["retweets"],
                "likes": json_data[key]["likes"],
                "is_retweet": json_data[key]["is_retweet"],
                "retweet_link": json_data[key]["retweet_link"],
                "posted_time": json_data[key]["posted_time"],
                "content": json_data[key]["content"],
                "hashtags": json_data[key]["hashtags"],
                "mentions": json_data[key]["mentions"],
                "images": json_data[key]["images"],
                "videos": json_data[key]["videos"],
                "tweet_url": json_data[key]["tweet_url"],
                "link": json_data[key]["link"],
            }
            writer.writerow(row)  # write row to CSV file
        data_file.close()  # after writing close the file
    logger.setLevel(logging.INFO)
    logger.info("Data Successfully Saved to {}.csv".format(filename))


def scrape_profile(
    twitter_username: str,
    browser: str = "firefox",
    proxy: Union[str, None] = None,
    tweets_count: int = 10,
    output_format: str = "json",
    filename: str = "",
    directory: str = os.getcwd(),
    headless: bool = True,
    browser_profile: Union[str, None] = None,
):
    """Scrap tweets of twitter profile using twitter username.

    Args:
        twitter_username (str): Twitter username of the account.
        browser (str, optional): Which browser to use for scraping?, Only 2 are supported Chrome and Firefox. Defaults to "firefox".
        proxy (Union[str, None], optional): Optional parameter, if user wants to use proxy for scraping. If the proxy is authenticated proxy then the proxy format is username:password@host:port. Defaults to None.
        tweets_count (int, optional): Number of posts to scrap. Defaults to 10.
        output_format (str, optional): The output format, whether JSON or CSV. Defaults to "json".
        filename (str, optional): If output_format parameter is set to CSV, then it is necessary for filename parameter to passed. If not passed then the filename will be same as keyword passed. Defaults to "".
        directory (str, optional): If output_format parameter is set to CSV, then it is valid for directory parameter to be passed. If not passed then CSV file will be saved in current working directory. Defaults to os.getcwd().
        headless (bool, optional): Whether to run browser in headless mode?. Defaults to True.
        browser_profile (Union[str, None], optional): Path of Browser Profile where cookies might be located to scrap data in authenticated way. Defaults to None.

    Returns:
        str: tweets data in CSV or JSON
    """
    profile_bot = Profile(
        twitter_username, browser, proxy, tweets_count, headless, browser_profile
    )
    data = profile_bot.scrap()
    if output_format.lower() == "json":
        if filename == "":
            # if filename was not provided then print the JSON to console
            return json.dumps(data)
        elif filename != "" and len(data) > 0:
            # if filename was provided, save it to that file
            mode = "w"
            json_file_location = os.path.join(directory, filename + ".json")
            if os.path.exists(json_file_location):
                mode = "r"
            with open(json_file_location, mode, encoding="utf-8") as file:
                if mode == "r":
                    try:
                        file_content = file.read()
                        content = json.loads(file_content)
                    except json.decoder.JSONDecodeError:
                        logger.warning("Invalid JSON Detected!")
                        content = {}
                    file.close()
                    data.update(content)
            with open(json_file_location, "w", encoding="utf-8") as file_in_write_mode:
                json.dump(data, file_in_write_mode)
                logger.setLevel(logging.INFO)
                logger.info("Data Successfully Saved to {}".format(json_file_location))
    elif output_format.lower() == "csv":
        if filename == "":
            filename = twitter_username
        logging.info(f"data:{data}")
        json_to_csv(filename=filename, json_data=data, directory=directory)
