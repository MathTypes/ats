import logging
#from google.cloud import firestore_v1 as firestore
from firebase_admin import firestore
from util import logging_utils
import pandas as pd
import firebase_admin
from firebase_admin import credentials
from firebase import firebase
import re
import datetime
from util import time_util

cred = credentials.Certificate("/Users/jianjunchen/repo/secrets/keen-rhino-386415-firebase-adminsdk-9jpg7-968c43811b.json")
default_app = firebase_admin.initialize_app(cred)

def get_db():
    db = firestore.client()
    return db

def delete_collection(tab_name, batch_size):
    coll_ref = get_db().collection(tab_name)
    docs = coll_ref.list_documents(page_size=batch_size)
    deleted = 0

    for doc in docs:
        if not "realtime" in doc.id and not "nitter" in doc.id and not "selenium_" in doc.id:
            print(f'Deleting doc {doc.id} => {doc.get().to_dict()}')
            doc.delete()
            deleted = deleted + 1
        if "nitter_recent_user_pghosh1" in doc.id:
            print(f'Deleting doc {doc.id} => {doc.get().to_dict()}')
            doc.delete()
            deleted = deleted + 1

    if deleted >= batch_size:
        return delete_collection(coll_ref, batch_size)

def get_unprocessed_tweets(tab_name, start_time, end_time, batch_size):
    tweets = list(get_db().collection(tab_name).stream())
    tweets_dict = list(map(lambda x: x.to_dict(), tweets))
    df = pd.DataFrame(tweets_dict)
    df = df[df["process_time"].isnull()]
    #logging.info(f"df:{df}")
    return df

def normalize_asset(symbol):
    if symbol in ["es", "spx"]:
        return "spy"
    if symbol in ["nq"]:
        return "qqq"
    return symbol


def get_processed_tweets_by_asset(tab_name, start_time, end_time, batch_size):
    tweets = list(get_db().collection(tab_name).where('process_time', '!=', 'null').stream())
    tweets_dict = list(map(lambda x: x.to_dict(), tweets))
    df = pd.DataFrame()
    df = pd.DataFrame(columns=["asset", "sentiment", "polarity",
                               "sentimentClass", "text", "time",
                               'lemma_text', 'keyword_text',
                               'subject_ner_names', 'subject_ner_count'])
    for val in tweets_dict:
        #cash_tags = val["cash_tags"].split()
        assets = set([x.lower() for x in val["cash_tags"]])
        if val["symbol"]:
            assets.add(normalize_asset(val["symbol"]))
        if not assets:
            continue
        logging.info(f"assets:{assets}")
        for tag in assets:
            df = df.append(
                {"asset": tag, "text": val["text"],
                 "sentiment": val["sentiment"],
                 "sentimentClass": val["sentimentClass"],
                 "time": val["ts"],
                 "lemma_text": val["lemma_text"],
                 "keyword_text": val["keyword_text"],
                 "subject_ner_names": val["subject_ner_names"],
                 "username": val["username"],
                }, ignore_index=True
            )
    return df

def update_processed_text(df):
    db = get_db()
    for index, row in df.iterrows():
        db.collection('recent_tweet_by_user').document(str(row["tweet_id"])).update({
            'process_time': datetime.datetime.now().timestamp()*1000,
            'cash_tags' :row['cash_tags'],
            'symbol' :row['symbol'],
            'lemma_text' :row['lemma_text'],
            'keyword_text' :row['keyword_text'],
            'subject_ner_names' :row['subject_ner_names'],
            'subject_ner_count' :row['subject_ner_count'],
            'polarity' :row['polarity'],
            'sentiment' :row['sentiment'],
            'sentimentClass' :row['sentimentClass'],
            })

def update_asset_sentiment(df):
    db = get_db()
    for index, row in df.iterrows():
        rounded_time = int(time_util.round_down(datetime.datetime.fromtimestamp(row["time"]/1000), 10).timestamp())
        doc_key = row["asset"] + "_" + str(rounded_time)
        logging.info(f"adding row: {row}")
        db.collection('recent_sentiment').document(doc_key).set({
            'asof': row["time"],
            "id": doc_key,
            'update_time': datetime.datetime.now().timestamp()*1000,
            'lemma_text' :row['lemma_text'],
            'keyword_text' :row['keyword_text'],
            'subject_ner_names' :row['subject_ner_names'],
            'subject_ner_count' :row['subject_ner_count'],
            'polarity' :row['polarity'],
            'sentiment' :row['sentiment'],
            'sentimentClass' :row['sentimentClass'],
            })
