
import datetime
import os
import pandas as pd

import logging
import spacy
from textblob import TextBlob

import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px
import pyLDAvis
import streamlit as st
from streamlit import components
from wordcloud import WordCloud

from data_analysis.topic_modeling import LatentDirichletAllocation
from neo4j_util.sentiment_api import get_tweets, get_tweet_replies_v2

from data.front_end_utils import (
    data_process,
    feature_extraction,
    visualize_ner,
    get_top_n_bigram,
    get_list_ner,
    display_text, subject_analysis, result_to_df, analyze_token_sentiment,
)

def draw_wordcloud(news_df, stop, asset="all assets", start_date=None, end_date=None):
    #start_date = datetime.datetime.combine(start_date, datetime.datetime.min.time())
    #end_date = datetime.datetime.combine(end_date, datetime.datetime.min.time())
    #dr = pd.date_range(start_date, end=end_date, tz='Asia/Tokyo')
    #logger.info(f'news_df.index:{news_df.index}')
    #logger.info(f'market_df.index:{market_df.index}')
    #logger.info(f'start_date:{type(start_date)}')
    #logging.info(f'asset:{asset}, start_date:{start_date}, end_date:{end_date}')
    #logger.info(f'news_df_draw_wordcloud:{news_df["text"]}')
    #logging.info(f'asset:{asset}')
    if asset.lower() == "all assets":
        headlines100k = news_df[news_df["time"].between(start_date, end_date)]['text']
        #logging.info(f'matched news:{headlines100k}')
    else:
        headlines100k = news_df.loc[news_df["assetName"] ==
                                    asset].loc[start_date:end_date, "text"].str.lower().values[-100000:]
    #logger.info(f'asset:{asset}')
    #logger.info(f'news:{headlines100k}')
    text = ' '.join(
        str(headline) for headline in headlines100k)
    #logger.info(f'draw_wordcloud:{text}')

    wordcloud = WordCloud(
        max_font_size=None,
        stopwords=stop,
        background_color='white',
        width=1200,
        height=850
    ).generate(text)

    fig1 = plt.figure(figsize = (3, 3))
    plt.subplot(1, 1, 1)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.subplots_adjust(wspace=.025, hspace=.025)

    # save image, display it, and delete after usage.
    plt.savefig('x',dpi=400)
    st.image('x.png')
    os.remove('x.png')
