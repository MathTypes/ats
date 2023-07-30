# Usage
#   PYTHONPATH=.. streamlit run app.py -- --neo4j_host=bolt://34.94.237.162:7687 --neo4j_pass=Tq7ks8zY --ts_root=../../ts/1_min
#
import datetime
import logging
import model_prediction
import news_analyzer
import os
import pandas as pd
import rvl
import streamlit as st

import ats.front_end.sentiment_analyzer
import ats.front_end.token_analyzer
import ats.front_end.trading_data
import ats.front_end.visualization

from ats.data.front_end_utils import data_process
from ats.market_data import ts_read_api
from ats.neo4j_util import sentiment_api
from ats.util import config_utils
from ats.util import logging_utils
from ats.util import nlp_utils


parser = config_utils.get_arg_parser("Preprocess tweet")

try:
    args = parser.parse_args()
except SystemExit as e:
    # This exception will be raised if --help or invalid command line arguments
    # are used. Currently streamlit prevents the program from exiting normally
    # so we have to do a hard exit.
    os._exit(e.code)

config_utils.set_args(args)
logging_utils.init_logging()

nlp = nlp_utils.get_nlp()

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
# =================================================================================== #
#                                Sidebar                                              #
# =================================================================================== #
navigated = st.sidebar.radio(
    "Navigation:", ["Overview", "Trading Data", "New Analysis"], index=0
)

market_df = None
news_df = None
assetNames = ["ES", "NQ", "RTY"]
futureAssetCodes = ["ES", "NQ", "RTY"]
to_date = datetime.date.today()
from_date = to_date - datetime.timedelta(days=14)
min_date = datetime.date(2022, 9, 1)
max_date = datetime.date.today()


@st.cache_data(ttl=600)
def load_data(from_date, to_date):
    df_vec = []
    for asset in futureAssetCodes:
        date_df = ts_read_api.get_time_series_from_monthly(
            asset, from_date, to_date, "1_min"
        )
        date_df["assetName"] = asset
        date_df["assetCode"] = asset
        df_vec.append(date_df)
    market_df = pd.concat(df_vec)
    # logging.info(f'market_index:{market_df.index}')
    # logging.info(f'false index:{market_df[market_df.index==False]}')
    news_df = sentiment_api.get_processed_tweets_from_monthly(from_date, to_date)
    logging.info(f"news_df:{news_df}")
    return market_df, news_df


market_df, news_df = load_data(from_date, to_date)
news_df = data_process(news_df)

logging.info(f"navigated:{navigated}")

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
        market_df, news_df, assetNames, from_date, to_date, min_date, max_date
    )

if navigated == "Visualization":
    visualization.render_visualization(news_df)
