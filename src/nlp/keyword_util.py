import logging
import pandas as pd
import numpy as np
import tweetnlp
import string
import inflect

import spacy
import nltk

nltk.download('wordnet')
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import requests
import time

# Set up the API call to the Inference API to do sentiment analysis
model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
hf_token = "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"
API_URL = "https://api-inference.huggingface.co/models/" + model
headers = {"Authorization": "Bearer %s" % (hf_token)}
wordnet_lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words("english")
stop_words.extend(
    ["from", "subject", "re", "edu", "use", "say", "said", "would", "also"]
)

sp = spacy.load("en_core_web_sm")
p = inflect.engine()


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
        label = topic_result["label"]
        probability = topic_result["probability"][label]
        return label + ":" + str(probability)
    return ""

def sentiment_analysis(data):
    if data:
        topic_result = get_sentiment_model().sentiment(data, return_probability=True)        
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

def add_hb_sentiment(df):
    df["nlp_sentiment"] = df["orig_text"].apply(sentiment_analysis)
    df["nlp_sentiment_label"] = df["nlp_sentiment"].apply(get_label)
    df["nlp_sentiment_score"] = df["nlp_sentiment"].apply(get_score)
    return df

def add_subject_keyword(df):
    # df = pd.read_csv(data_path, index_col=0)
    #df["subject"] = np.nan
    #for i in range(df.shape[0]):
    #    parag = list(df.iloc[i]["text"].split(", "))
    #    df.iloc[i]["subject"] = parag[0][2:]

    df["text"] = df["text"].apply(lambda x: text_process(x))
    df["subject"] = df["text"]
    df["lemma_text"] = df["text"].apply(lambda x: text_process(x, lemmetization=True))
    df["keyword_text"] = df["text"].apply(lambda x: text_process(x, word_cloud=True))
    df["text_ner_names"] = df["text"].apply(
        lambda x: name_entity_recognition(x, obj="name")
    )
    df["text_ner_count"] = df["text"].apply(
        lambda x: name_entity_recognition(x, obj="count")
    )

    df["lemma_subject"] = df["subject"].apply(
        lambda x: text_process(x, lemmetization=True)
    )
    df["keyword_subject"] = df["subject"].apply(
        lambda x: text_process(x, word_cloud=True)
    )
    df["subject_ner_names"] = df["subject"].apply(
        lambda x: name_entity_recognition(x, obj="name")
    )
    df["subject_ner_count"] = df["subject"].apply(
        lambda x: name_entity_recognition(x, obj="count")
    )

    return df


def name_entity_recognition(text, obj=None):
    l_e = []
    try:
        sen = sp(text)
        for entity in sen.ents:
            l_e.append(entity.label_)
    except:
        pass
    e_count = {}
    for e in l_e:
        e_count[e] = l_e.count(e)
    if obj == "name":
        return list(e_count.keys())
    else:
        return list(e_count.values())
    return


def text_process(text, word_cloud=False, stemming=False, lemmetization=False):
    if not isinstance(text, str):
        text = str(text)
    text = text[:4000]
    #logging.info(f"text:{text}")
    if lemmetization:
        text = "".join([i for i in text if i not in string.punctuation])
        text = text.lower()
        list_text = list(text.split(" "))
        text = [i for i in list_text if i not in stop_words]
        lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
        return lemm_text
    if word_cloud:
        text = "".join([i for i in text if i not in string.punctuation])
        text = text.lower()
        return [c.replace("'", "") for c in text.split(" ") if c not in stop_words]
    # text = convert_number(text)
    parag = ""
    for text in text.split(" ")[1:]:
        if text:
            if text[-1] == "'" or text[-1] == '"':
                parag += " " + text[:-1]
            if text[0] == "'" or text[0] == '"':
                if text[-1] == "'" or text[-1] == '"':
                    parag += " " + text[1:-1]
                else:
                    parag += " " + text[1:]
    logging.info(f"orig_text:{text}, parag:{parag[1:]}")
    return parag[1:]


def convert_number(text):
    temp_str = text.split()
    new_string = []
    for word in temp_str:
        if word.isdigit():
            temp = p.number_to_words(word)
            new_string.append(temp)

        else:
            new_string.append(word)
    temp_str = " ".join(new_string)
    return temp_str
