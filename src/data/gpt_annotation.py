
import os
from promptify import OpenAI
from promptify import Prompter

import argparse
import logging
from neo4j_util.sentiment_api import get_unprocessed_tweets, get_tweet_replies_v2
from neo4j_util.neo4j_tweet_util import Neo4j
from util import logging_utils


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
        data = get_unprocessed_tweets()
        logging.error(f'unprocess_data:{data}')
        for i, row in data.iterrows():
            result = nlp_prompter.fit('ner.jinja',
                                    domain='financial',
                                    text_input=row["text"],
                                    labels=None)
            # Output
            print(result)
            break
            logging.error(f'process_data:{data}')
            neo4j_util = Neo4j()
            neo4j_util.update_processed_text(data)        
        break
        
