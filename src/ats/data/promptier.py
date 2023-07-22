import os
from promptify import OpenAI
from promptify import Prompter

import argparse
import logging

parser = argparse.ArgumentParser(description="A script to add GPT annotation")
parser.add_argument(
    "-v", "--verbose", help="increase output verbosity", action="store_true"
)
parser.add_argument("--input", type=str, required=True)

args = parser.parse_args()
if args.verbose:
    logging.basicConfig(level=logging.DEBUG)

api_key = os.environ["OPENAI_API_KEY"]
model = OpenAI(api_key)  # or `HubModel()` for Huggingface-based inference
nlp_prompter = Prompter(model)

result = nlp_prompter.fit(
    "ner.jinja", domain="financial", text_input=args.input, labels=None
)
# Output
print(result)
