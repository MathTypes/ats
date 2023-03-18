# importing libraries and packages
#from absl import logging
import os
import snscrape.modules.twitter as sntwitter
import pandas as pd

import argparse
import logging

parser = argparse.ArgumentParser(
    description='A test script for http://stackoverflow.com/q/14097061/78845'
)
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--username", type=str, required=True)

args = parser.parse_args()
if args.verbose:
    logging.basicConfig(level=logging.DEBUG)


# Creating list to append tweet data 
tweets_list1 = []

# Using TwitterSearchScraper to scrape data and append tweets to list
for i,tweet in enumerate(sntwitter.TwitterUserScraper(args.username).get_items()): #declare a username 
    logging.info(f'tweet:{tweet}')
    if i>1000: #number of tweets you want to scrape
        break
    tweets_list1.append([tweet.date, tweet.id, tweet.content, tweet.user.username]) #declare the attributes to be returned
    
# Creating a dataframe from the tweets list above 
tweets_df1 = pd.DataFrame(tweets_list1, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])

file_path = os.path.join(args.output_dir, args.username)
if not os.path.exists(file_path):
    os.makedirs(file_path, exist_ok=True)
tweets_df1.to_csv(os.path.join(file_path, 'out.csv'))