from twitter_scraper_selenium import scrape_profile


scrape_profile(
    twitter_username="NoProb_XXX",
    output_format="csv",
    browser="chrome",
    tweets_count=100,
    filename="microsoft",
    directory="Downloads",
)
