import snscrape.modules.twitter as twitterScraper

scraper = twitterScraper.TwitterUserScraper("DougKass", {})

try:
    if scraper._get_entity():
        print(True)
except ValueError:
    print("Not found")
