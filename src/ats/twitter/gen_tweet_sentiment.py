import datetime
import logging
import os
import sys
import tweetnlp
import argparse
import modin.pandas as pd
#import pandas as pd

from data.front_end_utils import subject_analysis
import firebase_api
from nlp import keyword_util
from util import config_utils
from util import logging_utils
import re
import ray
from ray.util.dask import enable_dask_on_ray

def get_cash_tags(text):
    pattern = '\$([a-zA-Z.]+)\W'
    result = re.findall(pattern, text)
    #logging.info(f"result:{result}")
    return result


def find_symbol(x):
    x = str(x).lower()
    #logging.info(f"x:{x}")
    m = re.search("\$([a-z]{1,4})\W+", x)
    #logging.info(f"m:{m}")
    if m:
        return m.group(1)
    else:
        return ""


topic_model = None
sentiment_model = None
emoji_model = None
emotion_model = None
irony_model = None
ner_model = None

def get_topic_model():
    global topic_model
    if not topic_model:
        topic_model = tweetnlp.load_model('topic_classification')        
    return topic_model

def get_sentiment_model():
    global sentiment_model
    if not sentiment_model:
        sentiment_model = tweetnlp.load_model('sentiment')        
    return sentiment_model

def get_emoji_model():
    global emoji_model
    if not emoji_model:
        emoji_model = tweetnlp.load_model('emoji')        
    return emoji_model

def get_emotion_model():
    global emotion_model
    if not emotion_model:
        emotion_model = tweetnlp.load_model('emotion')        
    return emotion_model

def get_ner_model():
    global ner_model
    if not ner_model:
        ner_model = tweetnlp.load_model('ner')
    return ner_model

def get_irony_model():
    global irony_model
    if not irony_model:
        irony_model = tweetnlp.load_model('irony')
    return irony_model

def topic_analysis(data):
    if data:
        topic_result = get_topic_model().topic(data, return_probability=True)        
        #logging.info(f"topic_result:{topic_result}")
        topics = ""
        prob = ""
        for topic in topic_result["label"]:
            probability = topic_result["probability"][topic]
            if topics:
                topics = topics + "," + topic
                prob = prob + "," + str(probability)
            else:
                topics = topic
                prob =  str(probability)
        return topics + ":" + prob
    return ""

def sentiment_analysis(data):
    if data:
        topic_result = get_sentiment_model().sentiment(data, return_probability=True)        
        #logging.info(f"topic_result:{topic_result}")
        label = topic_result["label"]
        probability = topic_result["probability"][label]
        return label + ":" + str(probability)
    return ""

def irony_analysis(data):
    if data:
        topic_result = get_irony_model().irony(data, return_probability=True)        
        label = topic_result["label"]
        probability = topic_result["probability"][label]
        return label + ":" + str(probability)
    return ""

def emoji_analysis(data):
    if data:
        topic_result = get_emoji_model().emoji(data, return_probability=True)        
        label = topic_result["label"]
        probability = topic_result["probability"][label]
        return label + ":" + str(probability)
    return ""

def emotion_analysis(data):
    if data:
        topic_result = get_emotion_model().emotion(data, return_probability=True)        
        label = topic_result["label"]
        probability = topic_result["probability"][label]
        return label + ":" + str(probability)
    return ""

def ner_analysis(data):
    if data:
        topic_result = get_ner_model().ner(data, return_probability=True)        
        for val in topic_result:
            if val["type"] == "event":
                return ":".join(val["entity"])
        return ""
    return ""

def get_label(sentiment):
    if ":" in sentiment:
        val = sentiment.split(":")
        return val[0]
    return "NA"

def get_score(sentiment):
    if ":" in sentiment:
        val = sentiment.split(":")
        return float(val[1])
    return -100

def get_topic_score(sentiment):
    if ":" in sentiment:
        val = sentiment.split(":")
        return val[1]
    return ""

def is_power_user(username):
    return username in ['PharmD_KS','pghosh1','DougKass','JimScalpert','eliant_capital','Mayhem4Markets', 'Tricky_OW', 'DoubleWideCap']

def pandas_transform(df: pd.DataFrame) -> pd.DataFrame:
    df["power_user"] = df.username.apply(is_power_user)
    df = df[df.power_user]
    if df.empty:
        return df
    df["symbol"] = df["text"].apply(lambda x: find_symbol(x))
    df["cash_tags"] = df.text.apply(get_cash_tags)
    df["orig_text"] = df["text"]
    df["nlp_sentiment"] = df["orig_text"].apply(sentiment_analysis)
    df["nlp_sentiment_label"] = df["nlp_sentiment"].apply(get_label)
    df["nlp_sentiment_score"] = df["nlp_sentiment"].apply(get_score)
    df["nlp_topic"] = df["orig_text"].apply(topic_analysis)
    df["nlp_topic_label"] = df["nlp_topic"].apply(get_label)
    df["nlp_topic_score"] = df["nlp_topic"].apply(get_topic_score)
    df["nlp_emotion"] = df["orig_text"].apply(emotion_analysis)
    df["nlp_emotion_label"] = df["nlp_emotion"].apply(get_label)
    df["nlp_emotion_score"] = df["nlp_emotion"].apply(get_score)
    df["nlp_emoji"] = df["orig_text"].apply(emoji_analysis)
    df["nlp_emoji_label"] = df["nlp_emoji"].apply(get_label)
    df["nlp_emoji_score"] = df["nlp_emoji"].apply(get_score)
    df["nlp_irony"] = df["orig_text"].apply(irony_analysis)
    df["nlp_irony_label"] = df["nlp_irony"].apply(get_label)
    df["nlp_irony_score"] = df["nlp_irony"].apply(get_score)
    df["nlp_ner_event"] = df["orig_text"].apply(ner_analysis)
    return df

if __name__ == "__main__":
    #ray.shutdown()
    #ray.init(_temp_dir=f"/tmp/results/xg_tree", ignore_reinit_error=True)
    #enable_dask_on_ray()  # Enable the Ray scheduler backend for Dask.

    pd.set_option('display.max_columns', None)
    parser = config_utils.get_arg_parser("Preprocess tweet")
    parser.add_argument(
        "--start_date",
        type=lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").date(),
        required=True,
        help="Set a start date",
    )
    parser.add_argument(
        "--end_date",
        type=lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").date(),
        required=True,
        help="Set a end date",
    )
    parser.add_argument("--update", help="reprocess existing ones", action="store_true")

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    config_utils.set_args(args)
    logging_utils.init_logging()
    while True:
        data = firebase_api.get_unprocessed_tweets('recent_tweet_by_user', args.start_date, args.end_date, args.update)
        if data.empty:
            break
        data = pandas_transform(data)
        #ds = ray.data.from_pandas(data)
        #ds = ds.map_batches(pandas_transform, batch_size=10)
        #data = keyword_util.add_subject_keyword(data)
        #data = subject_analysis(data)
        #data = ds.to_dask()
        logging.info(f"{data.head()}")
        logging.info(f"{data.info()}")
        firebase_api.update_processed_text(data)
        break
    ray.shutdown()
        #
        #logging.error(f"process_data:{data}")
        #break
        # break
    # conv_data = get_tweet_replies_v2()
    # conv_data = add_subject_keyword(conv_data)
