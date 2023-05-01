#!/usr/bin/env python3

from typing import Union
import time
from .driver_initialization import Initializer
from .driver_utils import Utilities
from .element_finder import Finder
import re
import json
import os
import csv
from twitter_scraper_selenium.scraping_utilities import Scraping_utilities
import logging
from seleniumwire import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys

logger = logging.getLogger(__name__)
format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(format)
logger.addHandler(ch)

def get_headers(driver, email):
    #email = input('Please enter Twitter email/username: ')
    # email = "JeremyQiu6"
    account_dict = {"DragonFlye68042":"sunshine762",
                    "AlexSimon190658":"qiu7788lz",
                    "alexsimon788213":"sunshine762",
                    "alexsimon757084":"sunshine762"}
    password = account_dict[email]
    #password = input('Please enter Twitter password: ')
    #password = "tjqh754lir"
    #options = CustomChromeOptions()
    #options.add_argument("user-data-dir=/Users/jianjunchen/repo/ats-1/src/selenium_profile/Default")
    #options.add_argument("--disable-blink-features=AutomationControlled") 
    #options.add_experimental_option("excludeSwitches", ["enable-automation"]) 
    #options.add_experimental_option("useAutomationExtension", False) 
    #options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36")
    #driver = webdriver.Chrome(options=options)

    driver.get("https://twitter.com")
    while 'Error' in driver.title:
        time.sleep(1)
        driver.get("https://twitter.com")
        if 'Error' not in driver.title:
            break
    WebDriverWait(driver, 300).until(EC.presence_of_element_located((By.XPATH,"//*[text()='Log in']")))
    login_btn = driver.find_element(By.XPATH, "//*[text()='Log in']")
    login_btn.click()
    WebDriverWait(driver, 300).until(EC.presence_of_element_located((By.CSS_SELECTOR,'[autocomplete="username"]',)))
    username_field = driver.find_element(By.CSS_SELECTOR, '[autocomplete="username"]')
    username_field.send_keys(email)
    time.sleep(2)
    next_btn = driver.find_element(By.XPATH, "//*[text()='Next']")
    next_btn.click()
    WebDriverWait(driver, 300).until(EC.presence_of_element_located((By.CSS_SELECTOR,'[autocomplete="current-password"]',)))
    password_field = driver.find_element(By.CSS_SELECTOR, '[autocomplete="current-password"]')
    password_field.send_keys(password)
    time.sleep(2)
    login_btn = driver.find_element(By.CSS_SELECTOR, '[data-testid="LoginForm_Login_Button"]')
    login_btn.click()
    
    WebDriverWait(driver, 300).until(EC.presence_of_element_located((By.CSS_SELECTOR, '[placeholder="Search Twitter"]')))
    search_field = driver.find_element(By.CSS_SELECTOR, '[placeholder="Search Twitter"]')
    search_field.click()
    search_field.send_keys('whatever')
    search_field.send_keys(Keys.ENTER)
    while True:
        for request in driver.requests:
            logging.info(f'url:{request.url}')
            if 'https://twitter.com/i/api/2/search/adaptive.json?' in request.url:
                headers_dict = vars(request.headers)['_headers']
                headers = {}
                fields = ['authorization', 'User-Agent', 'cookie', 'Accept', 'Accept-Language', 'Referer', 'x-twitter-auth-type', 'x-guest-token', 'x-csrf-token', 'x-twitter-active-user']
                for i in headers_dict:
                    key = i[0]
                    value = i[1]
                    if key in fields:
                        headers[key] = value
                return headers

class Keyword:
    """This class needs to be instantiated in order to find something
    on twitter related to keywords"""

    def __init__(
        self,
        keyword: str,
        browser: str,
        proxy: Union[str, None],
        tweets_count: int,
        url: Union[str, None],
        headless: bool,
        browser_profile: Union[str, None],
    ):
        """Scrape Tweet using keyword.

        Args:
            keyword (str): Keyword to search on twitter.
            browser (str): Which browser to use for scraping?, Only 2 are supported Chrome and Firefox,default is set to Firefox.
            proxy (Union[str, None]): Optional parameter, if user wants to use proxy for scraping. If the proxy is authenticated proxy then the proxy format is username:password@host:port
            tweets_count (int): Number of tweets to scrap
            url (Union[str, None]): URL of the webpage.
            headless (bool): Whether to run browser in headless mode?.
            browser_profile (Union[str, None]): Path of Browser Profile where cookies might be located to scrap data in authenticated way.
        """
        self.keyword = keyword
        self.URL = url
        self.driver = ""
        self.browser = browser
        self.proxy = proxy
        self.tweets_count = tweets_count
        self.posts_data = {}
        self.retry = 10
        self.headless = headless
        self.browser_profile = browser_profile

    def start_driver(self):
        """changes the class member driver value to driver on call"""
        self.driver = Initializer(
            self.browser, self.headless, self.proxy, self.browser_profile
        ).init()

    def close_driver(self):
        self.driver.close()
        self.driver.quit()

    def check_tweets_presence(self, tweet_list):
        if len(tweet_list) <= 0:
            self.retry -= 1

    def check_retry(self):
        return self.retry <= 0

    def fetch_and_store_data(self):
        try:
            all_ready_fetched_posts = []
            present_tweets = Finder.find_all_tweets(self.driver)
            logging.info(f"present_tweets:{present_tweets}")
            self.check_tweets_presence(present_tweets)
            all_ready_fetched_posts.extend(present_tweets)

            while len(self.posts_data) < self.tweets_count:
                for tweet in present_tweets:
                    logging.info(f"tweet:{tweet}")
                    try:
                        name = Finder.find_name_from_tweet(tweet)
                        status, tweet_url = Finder.find_status(tweet)
                        replies = Finder.find_replies(tweet)
                        retweets = Finder.find_shares(tweet)
                        username = tweet_url.split("/")[3]
                        status = status[-1]
                        is_retweet = Finder.is_retweet(tweet)
                        posted_time = Finder.find_timestamp(tweet)
                        content = Finder.find_content(tweet)
                        likes = Finder.find_like(tweet)
                        images = Finder.find_images(tweet)
                        videos = Finder.find_videos(tweet)
                        hashtags = re.findall(r"#(\w+)", content)
                        mentions = re.findall(r"@(\w+)", content)
                        profile_picture = Finder.find_profile_image_link(tweet)
                        link = Finder.find_external_link(tweet)

                        self.posts_data[status] = {
                            "tweet_id": status,
                            "username": username,
                            "name": name,
                            "profile_picture": profile_picture,
                            "replies": replies,
                            "retweets": retweets,
                            "likes": likes,
                            "is_retweet": is_retweet,
                            "posted_time": posted_time,
                            "content": content,
                            "hashtags": hashtags,
                            "mentions": mentions,
                            "images": images,
                            "videos": videos,
                            "tweet_url": tweet_url,
                            "link": link,
                        }
                    except Exception as e:
                        logging.info(f"Exception: {e}")
                Utilities.scroll_down(self.driver)
                Utilities.wait_until_completion(self.driver)
                Utilities.wait_until_tweets_appear(self.driver)
                present_tweets = Finder.find_all_tweets(self.driver)
                present_tweets = [
                    post
                    for post in present_tweets
                    if post not in all_ready_fetched_posts
                ]
                self.check_tweets_presence(present_tweets)
                all_ready_fetched_posts.extend(present_tweets)
                if self.check_retry() is True:
                    break

        except Exception as ex:
            logger.exception("Error at method fetch_and_store_data : {}".format(ex))

    def scrap(self, login, email):
        try:
            self.start_driver()
            if login:
                headers = get_headers(self.driver, email)
                logging.info(f"headers:{headers}")
            logging.info(f"after driver is started")
            #self.driver.get('https://www.olx.com.pk/lahore/apple/q-iphone-6s/?search%5Bfilter_float_price%3Afrom%5D=40000&search%5Bfilter_float_price%3Ato%5D=55000')
            time.sleep(1)
            self.driver.get(self.URL)
            logging.info(f"after url is received")
            Utilities.wait_until_completion(self.driver, 30)
            logging.info(f"after url completion")
            Utilities.wait_until_tweets_appear(self.driver, 30)
            logging.info(f"after url tweet appearance")
            self.fetch_and_store_data()
            logging.info(f"after fetch and store")

            self.close_driver()
            data = dict(list(self.posts_data.items())[0 : int(self.tweets_count)])
            return data

        except Exception as ex:
            self.close_driver()
            logger.exception("Error at method scrap on : {}".format(ex))


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
    # open and start writing to CSV files
    if os.path.exists("{}.csv".format(filename)):
        mode = "a"
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
                "posted_time": json_data[key]["posted_time"],
                "content": json_data[key]["content"],
                "hashtags": json_data[key]["hashtags"],
                "mentions": json_data[key]["mentions"],
                "images": json_data[key]["images"],
                "videos": json_data[key]["videos"],
                "tweet_url": json_data[key]["tweet_url"],
                "link": json_data[key]["link"],
            }
            writer.writerow(row)  # write row to CSV fi
        data_file.close()  # after writing close the file
    logger.setLevel(logging.INFO)
    logger.info("Data Successfully Saved to {}.csv".format(filename))


def scrape_keyword(
    keyword: str,
    browser: str = "firefox",
    until: Union[str, None] = None,
    since: Union[int, None] = None,
    since_id: Union[int, None] = None,
    max_id: Union[int, None] = None,
    within_time: Union[str, None] = None,
    proxy: Union[str, None] = None,
    tweets_count: int = 10,
    output_format: str = "json",
    filename: str = "",
    directory: str = os.getcwd(),
    headless: bool = True,
    browser_profile: Union[str, None] = None,
    email: str = None,
    login: bool = False
):
    """Scrap tweets using keywords.

    Args:
        keyword (str): Keyword to search on twitter.
        browser (str, optional): Which browser to use for scraping?, Only 2 are supported Chrome and Firefox,default is set to Firefox. Defaults to "firefox".
        until (Union[str, None], optional): Optional parameter,Until date for scraping,a end date from where search ends. Format for date is YYYY-MM-DD or unix timestamp in seconds. Defaults to None.
        since (Union[int, None], optional): Optional parameter,Since date for scraping,a past date from where to search from. Format for date is YYYY-MM-DD or unix timestamp in seconds. Defaults to None.
        since_id (Union[int, None], optional): After (NOT inclusive) a specified Snowflake ID. Defaults to None.
        max_id (Union[int, None], optional): At or before (inclusive) a specified Snowflake ID. Defaults to None.
        within_time (Union[str, None], optional): Search within the last number of days, hours, minutes, or seconds. Defaults to None.
        proxy (Union[str, None], optional): Optional parameter, if user wants to use proxy for scraping. If the proxy is authenticated proxy then the proxy format is username:password@host:port. Defaults to None.
        tweets_count (int, optional): Number of posts to scrap. Defaults to 10.
        output_format (str, optional): The output format, whether JSON or CSV. Defaults to "json".
        filename (str, optional): If output parameter is set to CSV, then it is necessary for filename parameter to passed. If not passed then the filename will be same as keyword passed. Defaults to "".
        directory (str, optional): If output parameter is set to CSV, then it is valid for directory parameter to be passed. If not passed then CSV file will be saved in current working directory. Defaults to current work directory.
        headless (bool, optional): Whether to run browser in Headless Mode?. Defaults to True.
        browser_profile (str, optional): Path of Browser Profile where cookies might be located to scrap data in authenticated way. Defaults to None.

    Returns:
        str: tweets data in CSV or JSON
    """
    URL = Scraping_utilities.url_generator(
        keyword,
        since=since,
        until=until,
        since_id=since_id,
        max_id=max_id,
        within_time=within_time,
    )
    logging.info(f'sending URL:{URL}')
    keyword_bot = Keyword(
        keyword,
        browser=browser,
        url=URL,
        proxy=proxy,
        tweets_count=tweets_count,
        headless=headless,
        browser_profile=browser_profile,
    )
    data = keyword_bot.scrap(login, email)
    logging.info(f'done scraping url:{URL}, data:{data}')
    if output_format.lower() == "json":
        if filename == "":
            # if filename was not provided then print the JSON to console
            return json.dumps(data)
        elif filename != "":
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
                with open(
                    json_file_location, "w", encoding="utf-8"
                ) as file_in_write_mode:
                    json.dump(data, file_in_write_mode)
                    logger.setLevel(logging.INFO)
                    logger.info(
                        "Data Successfully Saved to {}".format(json_file_location)
                    )
    elif output_format.lower() == "csv":
        if filename == "":
            filename = keyword
        json_to_csv(filename=filename, json_data=data, directory=directory)
