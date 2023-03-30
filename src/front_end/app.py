
import datetime
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

from src.data_analysis.topic_modeling import LatentDirichletAllocation
from neo4j_util.sentiment_api import get_tweets, get_tweet_replies_v2

from utils import (
    data_process,
    feature_extraction,
    visualize_ner,
    get_top_n_bigram,
    get_list_ner,
    display_text, subject_analysis, result_to_df, analyze_token_sentiment,
)

import nlp_util
from app_dir.data_sourcing import Data_Sourcing, data_update
from app_dir.indicator_analysis import Indications
from app_dir.graph import Visualization
from tensorflow.keras.models import load_model
import gc

import CONFIG
from utils_dir.coingecko_utils import CoinGeckoUtils
from utils_dir.miner_utils import MinerUtils

import re
import plotly.express as px

from github import Github
import pandas as pd

import sqlite3

from xml.dom.pulldom import default_bufsize
import numpy as np
import plotly.express as px
from defillama import DefiLlama
from traitlets import default

import datetime
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from PIL import Image
from wordcloud import WordCloud

from eda_utils import generate_color
from neo4j_util.sentiment_api import get_tweets, get_gpt_sentiments
from market_data import ts_read_api
from util import logging_utils


logging_utils.init_logging()

nlp = spacy.load("en_core_web_sm")


st.set_page_config(layout="wide")
# =================================================================================== #
#                                Sidebar                                              #
# =================================================================================== #
ds = Image.open("images/ats.jpg")
st.sidebar.image(ds)
navigated = st.sidebar.radio("Navigation:", [
        "Visualization", "Model Predictions", "Trading Data", "New Analysis", "TVL vs MCAP Analysis", "XE Token Analyzer", "2Sigma Charts"], index=0)

st.title("Streamlit News Analysis")
#st.sidebar.title("Ats: Stock Market Analysis & Predictions")

#@st.cache_data()
market_df = None
news_df = None
assetNames = ["ES", "NQ", "RTY"]
def load_data():
    #market_df = pd.read_parquet(f"{datapath}/{MARKET_DATA}")
    from_date = datetime.date(2023, 3, 1)
    to_date = datetime.date(2023, 3, 25)
    df_vec = []
    for asset in assetNames:
        for date in pd.date_range(from_date, to_date):
            date_df = ts_read_api.get_time_series_by_instr_date(asset, date)
            df_vec.append(date_df)
    market_df = pd.concat(df_vec)
    #news_df = pd.read_parquet(f"{datapath}/{NEWS_DATA}")
    news_df = get_gpt_sentiments()
    #news_df['assetName'] = "ES"
    #news_df['sentimentClass'] = 1
    news_df['index_time'] = news_df["time"]
    #logging.info(f'news_df:{news_df}')
    news_df = news_df.set_index("index_time")
    news_df = news_df.sort_index()

    market_df['price_diff'] = market_df['close'] - market_df['open']
    #market_df.index = pd.to_datetime(market_df.index)
    #news_df.index = pd.to_datetime(news_df.index)

    return market_df, news_df

#with st.spinner("Loading data..."):
market_df, news_df = load_data()
#logging.info(f'news_df:{news_df}')
#data = pd.read_csv("dataset/process_data.csv")
#data = get_tweets()
#logging.info(f'data:{data}')
#conv_data = get_gpt_sentiments()
news_df = data_process(news_df)

def render_model_prediction():
    gc.collect()
    data_update()

    def main(app_data):
        indication = 'Predicted'

        st.sidebar.subheader('Asset:')
        asset_options = sorted(['Cryptocurrency', 'Index Fund', 'Forex', 'Futures & Commodities', 'Stocks'])
        asset = st.sidebar.selectbox('', asset_options, index = 4)

        if asset in ['Index Fund', 'Forex', 'Futures & Commodities', 'Stocks']:
            exchange = 'Yahoo! Finance'
            app_data.exchange_data(exchange)

            if asset == 'Stocks':
                st.sidebar.subheader(f'Stock Index:')
                stock_indexes  = app_data.stock_indexes
                market = st.sidebar.selectbox('', stock_indexes, index = 11)
                app_data.market_data(market)
                assets = app_data.stocks
                asset = f'{market} Companies'
            elif asset == 'Index Fund':
                assets = app_data.indexes
            elif asset == 'Futures & Commodities':
                assets = app_data.futures
            elif asset == 'Forex':
                assets = app_data.forex
        
            st.sidebar.subheader(f'{asset}:')
            equity = st.sidebar.selectbox('', assets)

            if asset == 'Futures & Commodities':
                currency = 'USD'
                market = None
            elif asset == 'Index Fund':
                currency = 'Pts'
                market = None
            elif asset == 'Forex':
                currency = app_data.df_forex[(app_data.df_forex['Currencies'] == equity)]['Currency'].unique()[0]
                market = app_data.df_forex[(app_data.df_forex['Currencies'] == equity)]['Market'].unique()[0]
            elif asset == f'{market} Companies':
                currency = app_data.df_stocks[((app_data.df_stocks['Company'] == equity) & (app_data.df_stocks['Index Fund'] == market))]['Currency'].unique()[0]
                asset = 'Stock'
        
            st.sidebar.subheader('Interval:')
            interval = st.sidebar.selectbox('', ('5 Minute', '15 Minute', '30 Minute', '1 Hour', '1 Day', '1 Week'), index = 4)
            volitility_index = 0     

        elif asset in ['Cryptocurrency']:
            exchange = 'Binance'
            app_data.exchange_data(exchange)
            markets = app_data.markets
        
            st.sidebar.subheader('Market:')
            market = st.sidebar.selectbox('', markets, index = 3)
            app_data.market_data(market)
            assets = app_data.assets
            currency = app_data.currency
        
            st.sidebar.subheader('Crypto:')
            equity = st.sidebar.selectbox('', assets)

            st.sidebar.subheader('Interval:')
            interval = st.sidebar.selectbox('', ('1 Minute', '3 Minute', '5 Minute', '15 Minute', '30 Minute', '1 Hour', '6 Hour', '12 Hour', '1 Day', '1 Week'), index = 8)

            volitility_index = 2 
        
        label = asset
        
        st.sidebar.subheader('Trading Volatility:')
        risk = st.sidebar.selectbox('', ('Low', 'Medium', 'High'), index = volitility_index)

        st.title(f'Automated Technical Analysis.')
        st.subheader(f'{label} Data Sourced from {exchange}.')
        st.info(f'Predicting...')
    
        future_price = 1   
        analysis = Visualization(exchange, interval, equity, indication, action_model, price_model, market)
        analysis_day = Indications(exchange, '1 Day', equity, market)
        requested_date = analysis.df.index[-1]
        current_price = float(analysis.df['Adj Close'][-1])
        change = float(analysis.df['Adj Close'].pct_change()[-1]) * 100
        requested_prediction_price = float(analysis.requested_prediction_price)
        requested_prediction_action = analysis.requested_prediction_action

        risks = {'Low': [analysis_day.df['S1'].values[-1], analysis_day.df['R1'].values[-1]], 
                'Medium': [analysis_day.df['S2'].values[-1], analysis_day.df['R2'].values[-1]],   
                'High': [analysis_day.df['S3'].values[-1], analysis_day.df['R3'].values[-1]],}
        buy_price = float(risks[risk][0])
        sell_price = float(risks[risk][1])

        if change > 0:
            change_display = f'A **{float(change):,.2f}%** gain'
        elif change < 0:
            change_display = f'A **{float(change):,.2f}%** loss'
        else:
            change_display = 'UNCH'

        if exchange == 'Yahoo! Finance':
            current_price = f'{float(current_price):,.2f}'
            requested_prediction_price = f'{float(requested_prediction_price):,.2f}'
            buy_price = f'{float(buy_price):,.2f}'
            sell_price = f'{float(sell_price):,.2f}'
        else:
            current_price = f'{float(current_price):,.8f}'
            requested_prediction_price = f'{float(requested_prediction_price):,.8f}'
            buy_price = f'{float(buy_price):,.8f}'
            sell_price = f'{float(sell_price):,.8f}'

        if analysis.requested_prediction_action == 'Hold':
            present_statement_prefix = 'off from taking any action with'
            present_statement_suffix = ' at this time'
        else:
            present_statement_prefix = ''
            present_statement_suffix = ''
                
        accuracy_threshold = {analysis.score_action: 75., analysis.score_price: 75.}
        confidence = dict()
        for score, threshold in accuracy_threshold.items():
            if float(score) >= threshold:
                confidence[score] = f'*({score}% confident)*'
            else:
                confidence[score] = ''

        forcast_prefix = int(interval.split()[0]) * future_price
        if forcast_prefix > 1:
            forcast_suffix = str(interval.split()[1]).lower() + 's'
        else:
            forcast_suffix = str(interval.split()[1]).lower()

        asset_suffix = 'price'

        st.markdown(f'**Prediction Date & Time (UTC):** {str(requested_date)}.')
        st.markdown(f'**Current Price:** {currency} {current_price}.')
        st.markdown(f'**{interval} Price Change:** {change_display}.')
        st.markdown(f'**Recommended Trading Action:** You should **{requested_prediction_action.lower()}** {present_statement_prefix} this {label.lower()[:6]}{present_statement_suffix}. {str(confidence[analysis.score_action])}')
        st.markdown(f'**Estimated Forecast Price:** The {label.lower()[:6]} {asset_suffix} for **{equity}** is estimated to be **{currency} {requested_prediction_price}** in the next **{forcast_prefix} {forcast_suffix}**. {str(confidence[analysis.score_price])}')
        if requested_prediction_action == 'Hold':
            st.markdown(f'**Recommended Trading Margins:** You should consider buying more **{equity}** {label.lower()[:6]} at **{currency} {buy_price}** and sell it at **{currency} {sell_price}**.')

        prediction_fig = analysis.prediction_graph(asset)
    
        st.success(f'Historical {label[:6]} Price Action.')
        st.plotly_chart(prediction_fig, use_container_width = True)

        technical_analysis_fig = analysis.technical_analysis_graph()
        st.plotly_chart(technical_analysis_fig, use_container_width = True) 
    

    if __name__ == '__main__':
        import warnings
        #import gc
        warnings.filterwarnings("ignore") 
        gc.collect()
        action_model = load_model("action_prediction_model.h5")
        price_model = load_model("price_prediction_model.h5")
        app_data = Data_Sourcing()
        main(app_data = app_data)

def render_trading_data():
    @st.cache_data()
    def get_all_coins_df():
        return CoinGeckoUtils().get_all_coins_df()

    @st.cache_data()
    def get_all_exchanges_df():
        return CoinGeckoUtils().get_all_exchanges_df()

    @st.cache_data()
    def get_miner_stats_df():
        return MinerUtils().get_miner_stats_df()

    @st.cache_data()
    def get_coin_tickers_by_id_list(coins_id: list):
        return CoinGeckoUtils().get_coin_tickers_by_id_list(coins_id)

    #st.set_page_config(layout='wide')
    st.title("Data Available")

    with st.spinner(text='In progress'):
        exchanges_df = get_all_exchanges_df()
        coins_df = get_all_coins_df()
        miner_stats_df = get_miner_stats_df()
    miner_coins = coins_df.loc[coins_df["symbol"].isin(miner_stats_df["base"].str.lower().unique()), "name"]

    default_miner_coins = ["Avalanche"]

    st.write("---")
    st.write("## Exchanges and coins data")

    with st.expander('Coins data'):
        st.dataframe(coins_df)

    with st.expander('Exchanges data'):
        from_date = datetime.date(2023, 3, 1)
        to_date = datetime.date(2023, 3, 25)
        es_market_df = ts_read_api.get_time_series('ES', from_date, to_date)
        nq_market_df = ts_read_api.get_time_series('NQ', from_date, to_date)
        exchanges_df = pd.concat([es_market_df, nq_market_df])
        st.dataframe(exchanges_df)

    st.write("---")
    st.write("## Tickers filtered")


    st.write("### Coins filter")
    tokens = st.multiselect(
        "Select the tokens to analyze:",
        options=coins_df["name"],
        default=default_miner_coins)

    coins_id = coins_df.loc[coins_df["name"].isin(tokens), "id"].tolist()

    coin_tickers_df = get_coin_tickers_by_id_list(coins_id)
    coin_tickers_df["coin_name"] = coin_tickers_df.apply(lambda x: coins_df.loc[coins_df["id"] == x.token_id, "name"].item(), axis=1)
    st.write("### Exchanges filter")
    exchanges = st.multiselect(
        "Select the exchanges to analyze:",
        options=exchanges_df["assetName"],
        default=[exchange for exchange in CONFIG.MINER_EXCHANGES if exchange in exchanges_df["assetName"].unique()])


    with st.expander('Coins Tickers Data'):
        st.dataframe(coin_tickers_df)

def render_new_analysis():
    @st.cache_data()
    def get_table_data(database_name: str, table_name: str):
        conn = sqlite3.connect(database_name)
        orders = pd.read_sql_query(f"SELECT * FROM '{table_name}'", conn)
        return orders

    @st.cache_data()
    def get_all_tables(database_name: str):
        con = sqlite3.connect(database_name)
        cursor = con.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table_row[0] for table_row in cursor.fetchall()]
        return tables

    st.title("Database Analyzer")
    st.write("---")
    uploaded_file = st.file_uploader("Add your database")

    if uploaded_file is not None:
        with open(f"{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        tables = get_all_tables(uploaded_file.name)
        st.subheader("Tables of the database:")
        for table in tables:
            st.write(table)
            st.dataframe(get_table_data(uploaded_file.name, table))


#crypto_analysis = st.sidebar.checkbox("TVL vs MCAP Analysis")

def render_tvl_mcap():
    MIN_TVL = 1000000.
    MIN_MCAP = 1000000.

    @st.cache_data()
    def get_tvl_mcap_data():
        llama = DefiLlama()
        df = pd.DataFrame(llama.get_all_protocols())
        tvl_mcap_df = df.loc[(df["tvl"]>0) & (df["mcap"]>0), ["name", "tvl", "mcap", "chain", "category", "slug"]].sort_values(by=["mcap"], ascending=False)
        return tvl_mcap_df[(tvl_mcap_df["tvl"] > MIN_TVL) & (tvl_mcap_df["mcap"]> MIN_MCAP)]

    def get_protocols_by_chain_category(protocols: pd.DataFrame, group_by: list, nth: list):
        return protocols.sort_values('tvl', ascending=False).groupby(group_by).nth(nth).reset_index()

    st.title("TVL vs MCAP Analysis")
    st.write("---")
    st.code("💡 Source: [DefiLlama](https://defillama.com/)")
    with st.spinner(text='In progress'):
        tvl_mcap_df = get_tvl_mcap_data()

    default_chains = ["Ethereum", "Solana", "Binance", "Polygon", "Multi-Chain", "Avalanche"]

    st.sidebar.write("### Chains filter")
    chains = st.sidebar.multiselect(
        "Select the chains to analyze:",
        options=tvl_mcap_df["chain"].unique(),
        default=default_chains)

    scatter = px.scatter(
        data_frame=tvl_mcap_df[tvl_mcap_df["chain"].isin(chains)],
        x="tvl",
        y="mcap",
        color="chain",
        trendline="ols",
        log_x=True,
        log_y=True,
        height=800,
        hover_data=["name"],
        template="plotly_dark",
        title="TVL vs MCAP",
        labels={
            "tvl": 'TVL (USD)',
            'mcap': 'Market Cap (USD)'
        })

    st.plotly_chart(scatter, use_container_width=True)

    st.sidebar.write("---")
    st.sidebar.write("### SunBurst filter")
    groupby = st.sidebar.selectbox('Group by:', [['chain', 'category'], ['category', 'chain']])
    nth = st.sidebar.slider('Top protocols by Category', min_value=1, max_value=5)

    proto_agg = get_protocols_by_chain_category(tvl_mcap_df[tvl_mcap_df["chain"].isin(chains)], groupby, np.arange(0, nth, 1).tolist())
    groupby.append("slug")
    sunburst = px.sunburst(
        proto_agg, 
        path=groupby,
        values='tvl',
        height=800,
        title="SunBurst",
        template="plotly_dark",)


    st.plotly_chart(sunburst, use_container_width=True)

    st.sidebar.write("# Data filters")


#xe_token_analyze = st.sidebar.checkbox("XE Token Analyzer")

def render_token_analyzer():
    @st.cache_data()
    def get_all_coins_df():
        return CoinGeckoUtils().get_all_coins_df()

    @st.cache_data()
    def get_all_exchanges_df():
        return CoinGeckoUtils().get_all_exchanges_df()

    @st.cache_data()
    def get_miner_stats_df():
        return MinerUtils().get_miner_stats_df()

    @st.cache_data()
    def get_coin_tickers_by_id_list(coins_id: list):
        return CoinGeckoUtils().get_coin_tickers_by_id_list(coins_id)

    st.title("Cross Exchange Token Analyzer")
    st.write("---")
    with st.spinner(text='In progress'):
        exchanges_df = get_all_exchanges_df()
        coins_df = get_all_coins_df()
        miner_stats_df = get_miner_stats_df()
    miner_coins = coins_df.loc[coins_df["symbol"].isin(miner_stats_df["base"].str.lower().unique()), "name"]


    st.write("### Coins filter")
    tokens = st.multiselect(
        "Select the tokens to analyze:",
        options=coins_df["name"],
        default=CONFIG.DEFAULT_MINER_COINS)

    coins_id = coins_df.loc[coins_df["name"].isin(tokens), "id"].tolist()

    coin_tickers_df = get_coin_tickers_by_id_list(coins_id)
    coin_tickers_df["coin_name"] = coin_tickers_df.apply(lambda x: coins_df.loc[coins_df["id"] == x.token_id, "name"].item(), axis=1)

    st.sidebar.write("### Exchanges filter")
    exchanges = st.sidebar.multiselect(
        "Select the exchanges to analyze:",
        options=exchanges_df["name"],
        default=[exchange for exchange in CONFIG.MINER_EXCHANGES if exchange in exchanges_df["name"].unique()])

    height = len(coin_tickers_df["coin_name"].unique()) * 500
    fig = px.scatter(
        data_frame=coin_tickers_df[coin_tickers_df["exchange"].isin(exchanges)],
        x="volume",
        y="bid_ask_spread_percentage",
        color="exchange",
        log_x=True,
        log_y=True,
        facet_col="coin_name",
        hover_data=["trading_pair"],
        facet_col_wrap=1,
        height=height,
        template="plotly_dark",
        title="Spread and Volume Chart",
        labels={
            "volume": 'Volume (USD)',
            'bid_ask_spread_percentage': 'Bid Ask Spread (%)'
        })

    st.sidebar.write("Data filters")
    st.plotly_chart(fig, use_container_width=True)

if navigated == "XE Token Analyzer":
    render_token_analyzer()

def render_sentiment_analysis():
    stop = set(stopwords.words('english'))
    analysis = st.sidebar.radio("Choose analysis", [
        "Data exploration", "Aggregation charts", "Sentiment analysis"], index=0)

    assets = list(set(news_df.assetName.to_list()))
    assets_dict = dict(zip(market_df.assetName, market_df.assetCode))

    def get_asset_code(asset):
        return assets_dict.get(asset)


    def draw_wordcloud(asset="all assets", start_date=None, end_date=None):
        start_date = datetime.datetime.combine(start_date, datetime.datetime.min.time())
        end_date = datetime.datetime.combine(end_date, datetime.datetime.min.time())
        #dr = pd.date_range(start_date, end=end_date, tz='Asia/Tokyo')
        #logging.info(f'news_df.index:{news_df.index}')
        #logging.info(f'market_df.index:{market_df.index}')
        #logging.info(f'start_date:{type(start_date)}')
        #logging.info(f'asset:{asset}, start_date:{start_date}, end_date:{end_date}')
        logging.info(f'news_df_draw_wordcloud:{news_df["time"]}')
        if asset.lower() == "all assets":
            headlines100k = news_df[news_df["time"].between(start_date, end_date)]['text'].str.lower(
            ).values[-100000:]
        else:
            headlines100k = news_df.loc[news_df["assetName"] ==
                                        asset].loc[start_date:end_date, "text"].str.lower().values[-100000:]
        #logging.info(f'asset:{asset}')
        #logging.info(f'news:{headlines100k}')
        text = ' '.join(
            headline for headline in headlines100k if type(headline) == str)
        #logging.info(f'draw_wordcloud:{text}')

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
        def function(app_data):
            exchange = 'Yahoo! Finance'
            app_data.exchange_data(exchange)

            if asset == 'Stocks':
                st.sidebar.subheader(f'Stock Index:')
                stock_indexes  = app_data.stock_indexes
                market = st.sidebar.selectbox('', stock_indexes, index = 11)
                app_data.market_data(market)
                assets = app_data.stocks
        function(app_data = app_data)


    def mis_value_graph(data):
        data = [
            go.Bar(
                x=data.columns,
                y=data.isnull().sum(),
                name='Unknown Assets',
            ),
        ]
        layout = go.Layout(
            xaxis=dict(title='Columns'),
            yaxis=dict(title='Value Count'),
            showlegend=True,
            legend=dict(
                yanchor="bottom",
                xanchor="right",
           )
        )
        fig = go.Figure(data=data, layout=layout)
        st.plotly_chart(fig, use_container_width=True)


    if analysis.lower() == "data exploration":

        row1_1, row1_2 = st.columns(2)

        with row1_1:
            with st.expander("Chart description: Total Missing Value By Column"):
                st.write(
                    "We can see that asset with name 'Unknown' accounts for all the missing values")

            mis_value_graph(market_df)

        assetNameGB = market_df[market_df['assetName']
                                == 'Unknown'].groupby('assetCode')
        unknownAssets = assetNameGB.size().reset_index('assetCode')
        unknownAssets.columns = ['assetCode', "value"]
        unknownAssets = unknownAssets.sort_values("value", ascending=False)

        colors = []
        for i in range(len(unknownAssets)):
            colors.append(generate_color())

        data = [
            go.Bar(
                x=unknownAssets.assetCode.head(25),
                y=unknownAssets.value.head(25),
                name='Unknown Assets',
                marker=dict(
                    color=colors,
                    opacity=0.85
                )
            ),
        ]
        layout = go.Layout(
            xaxis=dict(title='Asset codes'),
            yaxis=dict(title='Value Count'),
            showlegend=True,
            legend=dict(
                yanchor="bottom",
                xanchor="right",
            )
        )
        fig = go.Figure(data=data, layout=layout)

        with row1_2:
            with st.expander("Chart description: Unknown assets by Asset code"):
                st.write(
                    "Here we have the distribution of assetCodes for samples with asset name 'Unknown'. We could use them to impute asset names when possible.")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Market crash / Wordcloud", expanded=True):
            row2_1, row2_2 = st.columns(2)

        with row2_1:
            no_of_drops = st.slider("Select no. of drops to show", min_value=1,
                                    max_value=20, value=10, step=1)

            grouped = market_df.groupby('time').agg(
                {'price_diff': ['std', 'min']}).reset_index()

            g = grouped.sort_values(('price_diff', 'std'), ascending=False)[
                :no_of_drops]
            #logging.info(f'g:{g}')
            g['min_text'] = 'Maximum price drop: ' + \
                (-1 * g['price_diff']['min']).astype(str)
            data = [go.Scatter(
                x=g['time'].dt.strftime(date_format='%Y-%m-%d').values,
                y=g['price_diff']['std'].values,
                mode='markers',
                marker=dict(
                    size=g['price_diff']['std'].values,
                    color=g['price_diff']['std'].values,
                    colorscale='Portland',
                    showscale=True
                ),
                text=g['min_text'].values
            )]

            layout = go.Layout(
                autosize=True,
                title=f"Top {no_of_drops} months by standard deviation of price change within a day",
                hovermode='closest',
                yaxis=dict(
                    title='price_diff',
                    ticklen=5,
                    gridwidth=2,
                ),
                showlegend=False
            )
            fig = go.Figure(data=data, layout=layout)
            st.plotly_chart(fig, use_container_width=True)

            st.info("We can see huge price fluctiations when market crashed. But this **must be  wrong**, as there was no huge crash on January 2010... which means that there must be some outliers on our data.")

        with row2_2:
            assets = list(news_df.assetName.unique())
            assets.insert(0, "All assets")

            selected_asset = st.selectbox(
                "Please select the assets",
                assets,
                index=0,
            )

            time_period = st.date_input("From/To", [datetime.date(
                2022, 12, 1), datetime.date(2023, 12, 31)], min_value=datetime.date(2022, 12, 1), max_value=datetime.date(2023, 12, 31))

            nlp_util.draw_wordcloud(news_df, stop, selected_asset, *time_period)

    elif analysis.lower() == "aggregation charts":

        selected_assets = st.sidebar.multiselect(
            "Please select the assets",
            assets,
            default=assets[0],
            format_func=get_asset_code
        )

        start_date = st.sidebar.date_input(
            "Starting date",
            value=datetime.date(2022, 12, 1),
            min_value=datetime.date(2022, 12, 1),
            max_value=datetime.date(2023, 12, 30)
        )

        end_date = st.sidebar.date_input(
            "End date",
            value=datetime.date(2022, 12, 30),
            min_value=datetime.date(2022, 12, 1),
            max_value=datetime.date(2023, 12, 30)
        )

        if selected_assets:
            data1 = []
            for asset in selected_assets:
                asset_df = market_df[(market_df['assetName']
                                      == asset)].loc[start_date:end_date]

                data1.append(go.Scatter(
                    x=asset_df.index.strftime(date_format='%Y-%m-%d').values,
                    y=asset_df['close'].values,
                    name=asset
                ))
            layout = go.Layout(
                dict(
                    title="Closing prices of chosen assets",
                    yaxis=dict(title='Price (USD)'),
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1,
                                     label="1m",
                                     step="month",
                                     stepmode="backward"),
                                dict(count=6,
                                     label="6m",
                                    step="month",
                                    stepmode="backward"),
                                dict(count=1,
                                     label="YTD",
                                    step="year",
                                    stepmode="todate"),
                                dict(count=1,
                                     label="1y",
                                    step="year",
                                    stepmode="backward"),
                                dict(step="all")
                            ])
                        ),
                        rangeslider=dict(visible=True),
                        type="date"
                    )
                ),
                # legend=dict(orientation="h")
                legend=dict(
                    yanchor="bottom",
                    xanchor="right",
                )
            )

            st.plotly_chart(dict(data=data1, layout=layout),
                            use_container_width=True)

            with st.expander("Description"):
                st.write("We can see some companies' stocks started trading later, some dissappeared. Disappearence could be due to bankruptcy, acquisition or other reasons.")

        @st.cache_data()
        def trends_quantile_chart(market_df):
            data = []
            for i in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
                price_df = market_df.groupby(market_df.index)[
                    'close'].quantile(i).reset_index()

                data.append(go.Scatter(
                    x=price_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
                    y=price_df['close'].values,
                    name=f'{i} quantile'
                ))
            return data

        data2 = trends_quantile_chart(market_df)

        layout1 = go.Layout(
            dict(
                title="Trends of closing prices by quantiles",
                yaxis=dict(title='Price (USD)'),
            ),
            legend=dict(
                orientation="h"
            )
        )

        st.plotly_chart(dict(data=data2, layout=layout1), use_container_width=True)
        with st.expander("See explanation"):
            st.write(
                """
                The graph above shows how markets fell and rise again during 2007-2016. Higher quantile prices have increased with time and lower quantile prices decreased. Maybe the gap between poor and rich increases... on the other hand maybe more "little" companies are ready to go to market and prices of their shares isn't very high.
                """
            )

    elif analysis.lower() == "sentiment analysis":
        logging.info(f'news_df:{news_df.columns}')
        assets = list(news_df.assetName.unique())
        #logging.info(f'assets:{assets}')
        selected_assets = st.sidebar.multiselect(
            "Please select the assets",
            assets,
            default=['BAC']
        )

        start_date, end_date = st.sidebar.date_input("Time period (from/to)", [datetime.date(
            2022, 1, 1), datetime.date(2023, 12, 31)], min_value=datetime.date(2022, 1, 1), max_value=datetime.date(2023, 12, 31))

        row1_1, row1_2 = st.columns(2)
        row2_1, row2_2 = st.columns(2)

        sentiment_dict = dict(
            negative="-1",
            positive="1"
        )

        def top10_mean_sentiment_plot(sentiment, start_date, end_date):

            df_sentiment = news_df.loc[news_df['sentimentClass']
                                       == sentiment_dict[sentiment], 'assetName'].loc[start_date:end_date]
            top10_sentiment = df_sentiment.value_counts().head(10)
            companies, sentiment_counts = top10_sentiment.index, top10_sentiment.values
            data = go.Bar(
                x=companies,
                y=sentiment_counts,
                marker=dict(
                    color=px.colors.qualitative.Set3[:10]
                ),
            )

            layout = go.Layout(
                title=f"Top 10 assets by {sentiment} sentiment",
                xaxis=dict(title="Assets"),
                yaxis=dict(title=f"{sentiment} sentiment count")
            )
            return dict(data=data, layout=layout)

        with row1_1:
            st.plotly_chart(top10_mean_sentiment_plot(
                "negative", start_date, end_date))

        with row1_2:
            st.plotly_chart(top10_mean_sentiment_plot(
                "positive", start_date, end_date))

        #sent_labels = ['negative', 'neutral', 'positive']
        sent_labels = ['positive']
        grouped_assets = news_df.loc[start_date:end_date].groupby("assetName")
        assets_sentiment_dict = {}
        for asset in selected_assets:
            if asset in grouped_assets.groups.keys():
                asset_group = grouped_assets.get_group(asset)
                counts = asset_group["sentimentClass"].value_counts()
                counts = counts.values/sum(counts.values)
                assets_sentiment_dict[asset] = list(counts)

        #logging.info(f'assets_sentiment_dict:{assets_sentiment_dict}')
        sentiment_df = pd.DataFrame.from_dict(
            assets_sentiment_dict, orient='index', columns=sent_labels)
        sentiment_df = pd.melt(sentiment_df.rename_axis('asset').reset_index(), id_vars=[
                               "asset"], value_vars=sent_labels, var_name='sentiment', value_name='count')
        #logging.info(f'sentiment_df:{sentiment_df}')
        fig = px.bar(
            sentiment_df,
            x="sentiment",
            y="count",
            color="asset",
            barmode='group',
        )

        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                title=""
            )
        )

        with row2_1:
            st.plotly_chart(fig)

        def calculate_mean_sentiment(asset, period="5min"):
            X = []
            Y = []
            asset_news_df = news_df[news_df["assetName"] == asset]

            for name, group in asset_news_df.groupby(pd.Grouper(freq=period)):
                d = name.strftime("%m/%d/%Y, %H:%M")
                counts = group["sentimentClass"].value_counts()
                #logging.info(f'counts:{counts}')
                #logging.info(f'counts_index:{counts.index}')
                counts.index = counts.index.astype("int8")
                if counts.size > 0:
                    mean_sentiment_score = np.average(counts.index, weights=counts)
                else:
                    mean_sentiment_score = 0
                X.append(d)
                Y.append(mean_sentiment_score)
            return X, Y

        data = []
        for asset in selected_assets:
            X, Y = calculate_mean_sentiment(asset)
            data.append(go.Scatter(
                x=X,
                y=Y,
                name=asset
            ))

        layout = dict(
            title="Mean sentiment score over time",
            plot_bgcolor="#FFFFFF",
            hovermode="x",
            xaxis=dict(
                title='Month',
            ),
            yaxis=dict(title='Mean sentiment'),
        )

        with row2_2:
            st.plotly_chart(dict(data=data, layout=layout))

            with st.expander("Mean sentiment score computation"):
                st.latex(
                    r'''score = \frac{-1 \cdot samples(negative) + 1 \cdot samples(positive)}{total\_samples}''')

def render_visualization():
    global news_df
    type = st.sidebar.radio("information type:", ("General", "Detailed"))
    s_d = st.sidebar.date_input("Start:", datetime.date(2023, 3, 1))
    e_d = st.sidebar.date_input("End:", datetime.date(2023, 4, 22))
    # =================================================================================== #
    #                                Display Dataset                                      #
    # =================================================================================== #

    col1, col2 = st.columns(2)
    with col2:
        st.header("Tweet")
        st.dataframe(news_df)

    start_day = pd.to_datetime(s_d)
    end_day = pd.to_datetime(e_d)
    logging.info(f'start_day:{start_day}, end_day:{end_day}')
    logging.info(f'before filtering:{news_df}')
    sub_data = news_df[news_df["time"].between(start_day, end_day)]
    logging.info(f'sub_data:{sub_data["keyword_subject"]}')
    count = sub_data.shape[0]
    # =================================================================================== #
    #                                General                                              #
    # =================================================================================== #
    if type == "General":
        # =================================================================================== #
        #                                Wordcloud                                            #
        # =================================================================================== #
        col3, col4 = st.columns(2)
        with col3:
            sub_data["keyword_subject"] = sub_data["keyword_subject"].apply(lambda x: str(x).replace('[','').replace(']',''))
            sub_data["keyword_text"] = sub_data["keyword_text"].apply(lambda x: str(x).replace('[','').replace(']',''))
            st.subheader("**Subject wordcloud:**")
            cloud_text = " ".join(sub_data["keyword_subject"])
            logging.info(f'cloud_text:{cloud_text}')
            wordcloud = WordCloud(
                colormap="Blues",
                background_color="white",
                width=1200,
                height=800,
            ).generate(cloud_text)
            # Generate plot
            plt.figure(figsize=(100, 100))
            fig = plt.figure()
            ax = plt.axes()
            ax.imshow(wordcloud)
            ax.axis("off")
            st.pyplot(fig)
        with col4:
            st.subheader("**Text wordcloud:**")
            cloud_text = " ".join(sub_data["keyword_text"])
            #logging.info(f'cloud_text:{cloud_text}')
            wordcloud = WordCloud(
                colormap="Reds",
                background_color="white",
                width=1200,
                height=800,
            ).generate(cloud_text)
            # Generate plot
            plt.figure(figsize=(100, 100))
            fig = plt.figure()
            ax = plt.axes()
            ax.imshow(wordcloud)
            ax.axis("off")
            st.pyplot(fig)
        # =================================================================================== #
        #                                Ngrams                                               #
        # =================================================================================== #
        with st.container():
            st.subheader("**Ngrams exploration:**")
            col3, col4 = st.columns(2)
            with col3:
                # TODO: fix missing lemma_text
                sub_data["lemma_text"] = sub_data["keyword_text"].apply(lambda x: str(x).replace('[','').replace(']',''))
                #logging.info(f"sub_data_lemma_text:{sub_data['lemma_text']}")
                common_words = get_top_n_bigram(
                    sub_data["lemma_text"], ngram_range=2, n=20
                )
                df_20_bi = pd.DataFrame(
                    common_words, columns=["Bigram", "count"]
                )
                fig = px.bar(df_20_bi, x="Bigram", y="count")
                fig.update_layout(
                    title={
                        "text": "Top 20 Bigrams in the reviews",
                        "x": 0.5,
                        "xanchor": "center",
                        "yanchor": "top",
                    }
                )
                st.plotly_chart(fig)
                with col4:
                    common_words = get_top_n_bigram(
                        sub_data["lemma_text"].apply(lambda x: str(x).replace('[','').replace(']','')), ngram_range=3, n=20
                    )
                    df_20_bi = pd.DataFrame(
                        common_words, columns=["Bigram", "count"]
                    )
                    fig = px.bar(df_20_bi, x="Bigram", y="count")
                    fig.update_layout(
                        title={
                            "text": "Top 20 trigrams in the reviews",
                            "x": 0.5,
                            "xanchor": "center",
                            "yanchor": "top",
                        }
                    )
                    st.plotly_chart(fig)
        # =================================================================================== #
        #                                Sentiment analysis                                   #
        # =================================================================================== #
        with st.container():
            st.subheader("**News subject analysis:**")
            col1, col2 = st.columns(2)
            with col1:
                sub_data = subject_analysis(sub_data)
                st.markdown("**Subject analysis:**")
                fig = px.pie(sub_data, names='sentiment', title='News Classifictaion',
                             color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig)
            with col2:
                #df = df[df["sub_date"].between(pd.to_datetime("2023-01-01"), pd.to_datetime("2023-12-31"))]
                df = news_df
                df = df[df["time"].between(pd.to_datetime("2023-01-01"), pd.to_datetime("2023-12-31"))]
                sub_df = subject_analysis(df)
                st.markdown("**Text analysis:**")
                fig = px.bar(sub_df, x='sub_date', y='polarity', title='News subject polarity over time')
                st.plotly_chart(fig)
        # =================================================================================== #
        #                                Topic recognition                                    #
        # =================================================================================== #
        with st.container():
            st.subheader("**Topic exploration:**")
            st.info("this process takes some time ⌛ ...")
            lda = LatentDirichletAllocation(sub_data)
            perplexity, coherance_lda, vis = lda.visualisation()
            html_string = pyLDAvis.prepared_data_to_html(vis)
            st.components.v1.html(html_string, width=1300, height=800, scrolling=True)
    # =================================================================================== #
    #                                Detailed                                             #
    # =================================================================================== #
    else:
        # =================================================================================== #
        #                                NER                                                  #
        # =================================================================================== #
        with st.container():
            st.subheader("**Select the subject of the new to visualise:**")
            option = st.multiselect(
                "News Subjects:",
                sub_data["subject"].tolist(),
                [df["subject"][100]],
            )
            st.markdown("**Name entity Recognition:**")
            text = df["text"][df["subject"] == option[0]].values[0]
            sen = nlp(text)
            visualize_ner(
                    sen, title="", labels=nlp.get_pipe("ner").labels
            )
            st.markdown("**sentiment analysis:**")
            col1, col2 = st.columns(2)
            with col1:
                st.info("Results")
                sentiment = TextBlob(text).sentiment

                if sentiment.polarity > 0:
                    st.markdown("Sentiment: Positive :smiley:")
                elif sentiment.polarity < 0:
                    st.markdown("Sentiment: Negative :angry:")
                else:
                    st.markdown("Sentiment: Neutral 😐")
                res = result_to_df(sentiment)
                st.dataframe(res)
                fig = px.bar(res, x='metric', y='value')
                st.plotly_chart(fig)
            with col2:
                st.info('Token Sentiment')
                data_pos, data_neg = analyze_token_sentiment(text)
                fig_pos = px.bar(data_pos, x='words', y='polarity')
                fig_neg = px.bar(data_neg, x='words', y='polarity')
                st.plotly_chart(fig_pos)
                st.plotly_chart(fig_neg)
                # st.write(analyze_token_sentiment(text))
            # =================================================================================== #
            #                                KG                                                   #
            # =================================================================================== #
            st.markdown("**Knowledge graph representation:**")
            st.info("Due to depolyment problem, I recommand to visualise this plot on the "
                    "knowledge_graph notebook")
            # st.info("this process takes some time ⌛ ...")

            # sub_rf = generate_relations(text)
            # G = nx.from_pandas_edgelist(sub_kg, "source", "target",
            #                             edge_attr=True, create_using=nx.MultiDiGraph())
            # col7, col8 = st.columns(2)
            # with col7:
            #     fig = plt.figure()
            #     d = dict(G.degree)
            #     pos = nx.spring_layout(G)
            #     nx.draw(G, with_labels=True, node_color='#FF6347',
            #             node_size=[v * 100 for v in d.values()])
            #
            #     st.pyplot(fig, use_container_width= true)
            # with col8:
            #     st.dataframe(rf)

        # =================================================================================== #
        #                                summary NER                                          #
        # =================================================================================== #
        with st.container():
            st.subheader("**NER exemple:**")

            options = st.multiselect(
                "Select the NER to visualise:",
                [
                    "CARDINAL",
                    "DATE",
                    "EVENT",
                    "FAC",
                    "GPE",
                    "LANGUAGE",
                    "LAW",
                    "LOC",
                    "MONEY",
                    "NORP",
                    "ORDINAL",
                    "ORG",
                    "PERCENT",
                    "PERSON",
                    "PRODUCT",
                    "QUANTITY",
                    "TIME",
                    "WORK_OF_ART",
                ],
            )
            st.info("this process takes some time ⌛ ...")
            text_ner, text_ner_count = get_list_ner(sub_data)
            display_text(
                sub_data,
                options,
                text_ner=text_ner,
                text_ner_count=text_ner_count,
            )


if navigated == "New Analysis": 
    render_new_analysis()

if navigated == "Model Predictions":
    render_model_prediction()

if navigated == "Trading Data":
    render_trading_data()

if navigated == "TVL vs MCAP Analysis":
    render_tvl_mcap()

if navigated == "2Sigma Charts":
    render_sentiment_analysis()

if navigated == "Visualization":
    render_visualization()