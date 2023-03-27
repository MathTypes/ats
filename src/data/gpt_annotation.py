
import argparse
import logging
import os

from ratelimiter import RateLimiter
from promptify import OpenAI
from promptify import Prompter

from neo4j_util.sentiment_api import get_gpt_unprocessed_replied_tweets
from neo4j_util.neo4j_tweet_util import Neo4j
from util import logging_utils
import openai

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

def get_named_entity(x: str):
    res = query(
        "Extract top three financial named entities from following text in csv format:", x)
    return res

def get_equity_sentiment(x: str):
    res = query(
        "Extract top three financial named entities and sentiment and score from 1 to 5 from following text in csv format:", x)
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='A script to add GPT annotation'
    )
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    api_key = os.environ["OPENAI_API_KEY"]
    model = OpenAI(api_key)  # or `HubModel()` for Huggingface-based inference
    nlp_prompter = Prompter(model)

    logging_utils.init_logging()
    while True:
        data = get_gpt_unprocessed_replied_tweets()
        logging.error(f'unprocess_data:{data}')
        for i, row in data.iterrows():
            logging.info(f'link:{row["perma_link"]}')
            logging.info(f'text:{row["text"]}')
            result = get_named_entity(row["text"])
            # Output
            logging.info(f'result:{result}')
            equity_sentiment_result = get_equity_sentiment(row["text"])
            # Output
            logging.info(f'equity_sentiment_result:{equity_sentiment_result}')
            #logging.error(f'process_data:{data}')
            #neo4j_util = Neo4j()
            #neo4j_util.update_gpt_processed_text(data)        
        break
        
