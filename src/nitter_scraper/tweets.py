"""Module for scraping tweets"""
from bs4 import BeautifulSoup
import re
import logging
from datetime import datetime
import re
from typing import Dict, Optional

from requests_html import HTMLSession

from nitter_scraper.schema import Tweet  # noqa: I100, I202


def link_parser(tweet_link):
    logging.info(f"link:{tweet_link}")
    #links = list(tweet_link.links)
    #tweet_url = links[0]
    #parts = links[0].split("/")
    parts = tweet_link.split("/")

    tweet_id = parts[-1].replace("#m", "")
    username = parts[1]
    return tweet_id, username, tweet_link


def date_parser(tweet_date):
    split_datetime = tweet_date.split(",")

    day, month, year = split_datetime[0].strip().split("/")
    hour, minute, second = split_datetime[1].strip().split(":")

    data = {}

    data["day"] = int(day)
    data["month"] = int(month)
    data["year"] = int(year)

    data["hour"] = int(hour)
    data["minute"] = int(minute)
    data["second"] = int(second)

    return datetime(**data)


def clean_stat(stat):
    return int(stat.replace(",", ""))


def stats_parser(tweet_stats):
    stats = {}
    for ic in tweet_stats.find(".icon-container"):
        key = ic.find("span", first=True).attrs["class"][0].replace("icon", "").replace("-", "")
        value = ic.text
        stats[key] = value
    return stats


def attachment_parser(attachements):
    photos, videos = [], []
    if attachements:
        photos = [i.attrs["src"] for i in attachements.find("img")]
        videos = [i.attrs["src"] for i in attachements.find("source")]
    return photos, videos


def cashtag_parser(text):
    cashtag_regex = re.compile(r"\$[^\d\s]\w*")
    return cashtag_regex.findall(text)


def hashtag_parser(text):
    hashtag_regex = re.compile(r"\#[^\d\s]\w*")
    return hashtag_regex.findall(text)


def url_parser(links):
    return sorted(filter(lambda link: "http://" in link or "https://" in link, links))


def parse_tweet(html) -> Dict:
    data = {}
    id, username, url = link_parser(html)
    data["tweet_id"] = id
    data["tweet_url"] = url
    data["username"] = username
    data["is_retweet"] = False
    data["is_pinned"] = False
    data["time"] = datetime.today()
    data["text"] = ""
    data["retweets"] = 0
    data["likes"] = 0
    data["replies"] = 0
    entries = {}
    entries["photos"] = ""
    entries["videos"] = ""
    data["entries"] = entries
    logging.info(f"data:{data}")
    return data


def timeline_parser(html):
    return html.find(".photo-rail-grid", first=True)


def pagination_parser(timeline, address, username) -> str:
    if not timeline.find(".show-more"):
        return ""
    next_page = list(timeline.find(".show-more")[-1].links)[0]
    return f"{address}/{username}{next_page}"


def get_tweets(
    username: str,
    pages: int = 25,
    break_on_tweet_id: Optional[int] = None,
    address="https://nitter.net",
) -> Tweet:
    """Gets the target users tweets

    Args:
        username: Targeted users username.
        pages: Max number of pages to lookback starting from the latest tweet.
        break_on_tweet_id: Gives the ability to break out of a loop if a tweets id is found.
        address: The address to scrape from. The default is https://nitter.net which should
            be used as a fallback address.

    Yields:
        Tweet Objects

    """
    url = f"{address}/{username}"
    session = HTMLSession()

    def gen_tweets(pages):
        logging.info(f"url:{url}")
        response = session.get(url)
        logging.info(f"response:{response}")

        while pages > 0:
            if response.status_code == 200:
                logging.info(f"response:{response.html.html}")
                #timeline = timeline_parser(response.html)
                #logging.info(f"timeline:{timeline}")

                #next_url = pagination_parser(timeline, address, username)
                #if not next_url:
                #    logging.info("no next_url")
                #    pages = 0
                #    break
                soup = BeautifulSoup(response.html.html,'html.parser') 
                pattern = re.compile(r"\/.*?#m")
                timeline_items = soup.findAll("a", href=pattern)

                for item in timeline_items:
                    logging.info(f"item:{item}")
                    #if "show-more" in item.attrs["class"]:
                    #    continue

                    tweet_data = parse_tweet(item["href"])
                    logging.info(f"tweet_data:{tweet_data}")
                    tweet = Tweet.from_dict(tweet_data)

                    if tweet.tweet_id == break_on_tweet_id:
                        pages = 0
                        break

                    yield tweet
                break
            #if next_url:
            #    response = session.get(next_url)
            #    pages -= 1


    yield from gen_tweets(pages)
