#
# Example of usage:
# PYTHONPATH=. python3 ../scripts/gpt_annotation.py --input_file=../data/tweet/eliant_capital/out.csv \
#    --output_file=../data/tweet/eliant_capital/gpt_annotation.csv
#
# from absl import logging
import glob
import os
from ratelimiter import RateLimiter
import snscrape.modules.twitter as sntwitter
import pandas as pd
import os
import openai

import argparse
import logging

parser = argparse.ArgumentParser(
    description='A script to add GPT annotation'
)
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--max_rows", help="max number of rows",
                    type=int, default=-1)

args = parser.parse_args()
if args.verbose:
    logging.basicConfig(level=logging.DEBUG)


@RateLimiter(max_calls=25, period=60)
def query(prompt, data, to_print=True):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="{}:\n\n{}".format(prompt, data),
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    payload = response['choices'][0]['text'].strip()
    if to_print:
        print(payload)
    return payload


def get_entity_analysis(x: str):
    res = query(
        "Extract knowledge graph and summary in a tree structure with nodes along with reasoning", x)
    return res


def get_opinion(x: str):
    # res = query(
    # "Extract US market sentiment from the text on a scale of 1 to 10", x)
    res = query(
        "Recognize all opinion terms in the following review with the format ['opinion_1', 'opinion_2', ...]:", x)
    return res


def get_sentiment(x: str):
    # res = query(
    # "Extract US market sentiment from the text on a scale of 1 to 10", x)
    res = query(
        "Recognize the sentiment polarity for aspect term 'US market' in the following review with the format ['aspect', 'sentiment']:", x)
    return res


def get_entity_surprise(x: str):
    res = query(
        "Extract surpise level on a scale from 0 to 10 along with reasoning", x)
    return res


def get_aspect_opinion(x: str):
    res = query(
        "Recognize the opinion term for aspect term 'market' in the following review with the format ['opinion_1', 'opinion_2', ...]:", x)
    return res


def get_aspect(x: str):
    res = query(
        "Recognize all aspect terms in the following review with the format ['aspect_1', 'aspect_2', ...]:", x)
    return res


def get_aspect_sentiment(x: str):
    res = query(
        "Recognize all aspect terms with their corresponding sentiment polarity in the following review with the format ['aspect', 'sentiment_polarity']: ", x)
    return res


openai.api_key = os.environ["OPENAI_API_KEY"]
openai.verify_ssl_certs = False

from neo4j import GraphDatabase

host = 'bolt://neo4j:7687'
user = 'neo4j'
password = 'pleaseletmein'
driver = GraphDatabase.driver(host, auth=(user, password))

def read_query(query, params={}):
    with driver.session() as session:
        result = session.run(query, params)
        response = [r.values()[0] for r in result]
        return response

def get_conversation_ids():
    text = read_query(
        "MATCH (n:Conversation)-[CONTAINS] - (t:Tweet) return n.id, min(t.pubdate) as start_date, max(t.pubdate) as end_date order by start_date")
    return text

def get_tweets(conv_id):
    text = read_query(
        f"MATCH(c:Conversation {id:{conv_id}})-[CONTAINS]->(t:Tweet) return t.raw_content as text order by t.pubdate")
    logging.error(f'text:text')
    return text


def read_dataset():
    """Load spam training dataset without any labels."""
    conversation_ids = get_conversation_ids()
    df = pd.DataFrame(columns = ['conv_id', 'text'])
    for conv_id in conversation_ids:
        tweets = get_tweets(conv_id)
        df = df.append({'conv_id' : conv_id, 'text' : tweets.join("\n")}, ignore_index = True)
    return df


tweets_df1 = read_dataset()
# make sure indexes pair with number of rows
tweets_df1 = tweets_df1.reset_index()

if args.max_rows > 0:
    tweets_df1 = tweets_df1.iloc[1:args.max_rows]
tweets_df1['aspect'] = tweets_df1['text'].map(
    lambda x: get_aspect(x))
tweets_df1['opinion'] = tweets_df1['text'].map(
    lambda x: get_opinion(x))
tweets_df1['aspect_opinion'] = tweets_df1['text'].map(
    lambda x: get_aspect_opinion(x))
tweets_df1['sentiment'] = tweets_df1['text'].map(lambda x: get_sentiment(x))
tweets_df1['aspect_sentiment'] = tweets_df1['text'].map(lambda x: get_aspect_sentiment(x))
tweets_df1['entity_surprise'] = tweets_df1['text'].map(
    lambda x: get_entity_surprise(x))
tweets_df1['entity_analysis'] = tweets_df1['text'].map(
    lambda x: get_entity_analysis(x))

rels_query ="""
UNWIND $data as row
MATCH (s:Entity {id: row.source})
MATCH (t:Entity {id: row.target})
CALL apoc.merge.relationship(s, row.type,
  {},
  {},
  t,
  {}
)
YIELD rel
RETURN distinct 'done';
"""

for i, row in tweets_df1[['entity', 'property', 'value']].iterrows():
    source = row['entity']['name'] if len(row['entity']['allUris']) == 0 else row['entity']['allUris'][0]
    target = row['value']['name'] if len(row['value']['allUris']) == 0 else row['value']['allUris'][0]
    type = row['property']['name'].replace(' ', '_').upper()
    relParams.append({'source':source,'target':target,'type':type})
    run_query(rels_query, {'data': relParams})

