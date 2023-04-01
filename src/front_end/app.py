
import datetime
import logging
import model_prediction
import news_analyzer
import nltk
import pandas as pd
import rvl
import sentiment_analyzer
import spacy
import streamlit as st
import token_analyzer
import trading_data
import visualization
from PIL import Image

from data.front_end_utils import (data_process,
                                  subject_analysis)
from market_data import ts_read_api
from neo4j_util.sentiment_api import (get_processed_tweets)
from util import logging_utils

nltk.download('stopwords')

logging_utils.init_logging()

nlp = spacy.load("en_core_web_sm")

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
# =================================================================================== #
#                                Sidebar                                              #
# =================================================================================== #
ds = Image.open("images/ats.jpg")
st.sidebar.image(ds)
navigated = st.sidebar.radio("Navigation:", [
    "Overview", "Visualization", "Model Predictions", "Trading Data", "New Analysis", "TVL vs MCAP Analysis", "XE Token Analyzer"], index=0)

# st.sidebar.title("Ats: Stock Market Analysis & Predictions")

# @st.cache_data()
market_df = None
news_df = None
assetNames = ["ES", "NQ", "RTY"]
futureAssetCodes = ["ES", "NQ", "RTY"]
from_date = datetime.date(2023, 3, 1)
to_date = datetime.date(2023, 3, 29)
min_date = datetime.date(2022, 9, 1)
max_date = datetime.date.today()

# @st.cache_data(ttl=600)


def load_data(from_date, to_date):
    df_vec = []
    for asset in futureAssetCodes:
        date_df = ts_read_api.get_time_series_by_range(
            asset, from_date, to_date)
        df_vec.append(date_df)
    market_df = pd.concat(df_vec)
    logging.info(f'market_index:{market_df.index}')
    logging.info(f'false index:{market_df[market_df.index==False]}')
    news_df = get_processed_tweets()
    news_df['index_time'] = news_df["time"]
    news_df = news_df.set_index("index_time")
    news_df = news_df.sort_index()
    news_df = subject_analysis(news_df)
    return market_df, news_df


market_df, news_df = load_data(from_date, to_date)
news_df = data_process(news_df)

if navigated == "XE Token Analyzer":
    token_analyzer.render_token_analyzer()

if navigated == "New Analysis":
    news_analyzer.render_new_analysis(news_df)

if navigated == "Model Predictions":
    model_prediction.render_model_prediction()

if navigated == "Trading Data":
    trading_data.render_trading_data()

if navigated == "TVL vs MCAP Analysis":
    rvl.render_tvl_mcap()

if navigated == "Overview":
    sentiment_analyzer.render_sentiment_analysis(
        market_df, news_df, assetNames, from_date, to_date, min_date, max_date)

if navigated == "Visualization":
    visualization.render_visualization(news_df)
