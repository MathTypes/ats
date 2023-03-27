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
nlp = spacy.load("en_core_web_sm")
st.set_page_config(layout="wide")

st.title("Streamlit News Analysis")

# =================================================================================== #
#                                Sidebar                                              #
# =================================================================================== #
ds = Image.open("images/news.jpg")
st.sidebar.image(ds)
st.sidebar.title("BBC News: Climate news analysis:")

st.sidebar.markdown("Navigation:")

viz = st.sidebar.checkbox("Visualisation")
if viz:
    type = st.sidebar.radio("information type:", ("General", "Detailed"))
    s_d = st.sidebar.date_input("Start:", datetime.date(2023, 3, 1))
    e_d = st.sidebar.date_input("End:", datetime.date(2023, 4, 22))

trade_setup = st.sidebar.checkbox("Model Predictions")
if trade_setup:
    gc.collect()
    data_update()

    def main(app_data):
        st.set_page_config(layout = "wide")
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
        import gc
        warnings.filterwarnings("ignore") 
        gc.collect()
        action_model = load_model("external/automating-technical-analysis/models/action_prediction_model.h5")
        price_model = load_model("external/automating-technical-analysis/models/price_prediction_model.h5")
        app_data = Data_Sourcing()
        main(app_data = app_data)

data_py_open = st.sidebar.checkbox("Trading Data")

if data_py_open:
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
        options=exchanges_df["name"],
        default=[exchange for exchange in CONFIG.MINER_EXCHANGES if exchange in exchanges_df["name"].unique()])


    with st.expander('Coins Tickers Data'):
        st.dataframe(coin_tickers_df)

hummingbot_DB_open = st.sidebar.checkbox("News Analysis")

if hummingbot_DB_open: 
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

crypto_analysis = st.sidebar.checkbox("TVL vs MCAP Analysis")

if crypto_analysis:
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

xe_token_analyze = st.sidebar.checkbox("XE Token Analyzer")

if xe_token_analyze:
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

# =================================================================================== #
#                                Display Dataset                                      #
# =================================================================================== #
#data = pd.read_csv("dataset/process_data.csv")
data = get_tweets()
logging.info(f'data:{data}')
conv_data = get_tweet_replies_v2()
df = data_process(data)

col1, col2 = st.columns(2)
with col1:
    bbc = Image.open("images/bbc.png")
    st.image(bbc)
with col2:
    st.header("Tweet")
st.dataframe(
        data[["id", "user", "time", "text", "source_url", "like_count", "perma_link", "last_update"]]
    )
col3, col4 = st.columns(2)
with col4:
    st.header("Conversation")
st.dataframe(
        conv_data[["tweet_id", "text"]]
    )

# =================================================================================== #
#                                Graphs visualisation                                 #
# =================================================================================== #
if viz:
    start_day = pd.to_datetime(s_d)
    end_day = pd.to_datetime(e_d)
    sub_data = df[df["time"].between(start_day, end_day)]
    count = sub_data.shape[0]
    st.markdown(
        '<p style="text-align: center; color:#A52A2A; font-size:50px; font-family:Arial Black">Number of '
        "BBC articles "
        "from {} to {}: {}</p> ".format(s_d, e_d, count),
        unsafe_allow_html=True,
    )
    # =================================================================================== #
    #                                General                                              #
    # =================================================================================== #
    if type == "General":
        # =================================================================================== #
        #                                Wordcloud                                            #
        # =================================================================================== #
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("**Subject wordcloud:**")
            Cloud_text = " ".join(sub_data["keyword_subject"])
            wordcloud = WordCloud(
                colormap="Blues",
                background_color="white",
                width=1200,
                height=800,
            ).generate(Cloud_text)
            # Generate plot
            plt.figure(figsize=(100, 100))
            fig = plt.figure()
            ax = plt.axes()
            ax.imshow(wordcloud)
            ax.axis("off")
            st.pyplot(fig)
        with col4:
            st.subheader("**Text wordcloud:**")
            Cloud_text = " ".join(sub_data["keyword_text"])
            wordcloud = WordCloud(
                colormap="Reds",
                background_color="white",
                width=1200,
                height=800,
            ).generate(Cloud_text)
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
                        sub_data["lemma_text"], ngram_range=3, n=20
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
                df = df[df["sub_date"].between(pd.to_datetime("2021-10-01"), pd.to_datetime("2021-12-30"))]
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


