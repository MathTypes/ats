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
import streamlit as st
from wordcloud import WordCloud

from eda_utils import generate_color
from neo4j_util.sentiment_api import get_tweets, get_conversations

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

logger = logging.getLogger(__name__)

root = logging.getLogger()
root.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
formatter = logging.Formatter(FORMAT)
ch.setFormatter(formatter)
root.addHandler(ch)


stop = set(stopwords.words('english'))

DATA_PATH = "./datasets"
MARKET_DATA = "market_pre2011.gzip"
NEWS_DATA = "news_pre2011.gzip"


#st.set_page_config(layout="wide")


@st.cache(show_spinner=False)
def load_data(datapath):
    #market_df = pd.read_parquet(f"{datapath}/{MARKET_DATA}")
    es_market_df = pd.read_csv(f'../../data/FUT/5_secs/ESH3/20230101.csv')
    es_market_df['assetName'] = "ES"
    es_market_df['assetCode'] = "ES"
    nq_market_df = pd.read_csv(f'../../data/FUT/5_secs/NQH3/20230101.csv')
    nq_market_df['assetName'] = "NQ"
    nq_market_df['assetCode'] = "NQ"
    market_df = pd.concat([es_market_df, nq_market_df])
    market_df = market_df.rename(columns={"date": "time"})
    market_df["time"] = pd.to_datetime(market_df["time"], infer_datetime_format=True)
    #logger.info(f'market_df_time:{market_df["time"]}')
    market_df = market_df.set_index("time")
    market_df = market_df.sort_index()
    #news_df = pd.read_parquet(f"{datapath}/{NEWS_DATA}")
    news_df = get_tweets()
    news_df = news_df.rename(columns={"created_at": "time", "raw_content":"headline"})
    news_df["time"] = news_df["time"].apply(lambda x: x.to_native())
    news_df["time"] = pd.to_datetime(news_df["time"], infer_datetime_format=True)
    news_df['assetName'] = "ES"
    news_df['sentimentClass'] = 1
    #logger.info(f'news_df:{news_df}')
    news_df = news_df.set_index("time")
    news_df = news_df.sort_index()

    market_df['price_diff'] = market_df['close'] - market_df['open']
    #market_df.index = pd.to_datetime(market_df.index)
    #news_df.index = pd.to_datetime(news_df.index)

    return market_df, news_df


with st.spinner("Loading data..."):
    market_df, news_df = load_data(DATA_PATH)


st.sidebar.title("Two Sigma EDA")
image = Image.open(f"compet_logo.jpg")
st.sidebar.image(image, use_column_width=True)
st.sidebar.write("Analyzing news data to predict stock prices. Understanding the predictive power of the news to predict financial outcomes and generate significant economic impact all over the world.")

analysis = st.sidebar.radio("Choose analysis", [
    "Data exploration", "Aggregation charts", "Sentiment analysis"], index=0)

assets = list(set(market_df.assetName.to_list()))

assets_dict = dict(zip(market_df.assetName, market_df.assetCode))


def get_asset_code(asset):
    return assets_dict.get(asset)


def draw_wordcloud(asset="all assets", start_date=None, end_date=None):
    start_date = datetime.datetime.combine(start_date, datetime.datetime.min.time())
    end_date = datetime.datetime.combine(end_date, datetime.datetime.min.time())
    #dr = pd.date_range(start_date, end=end_date, tz='Asia/Tokyo')
    #logger.info(f'news_df.index:{news_df.index}')
    #logger.info(f'market_df.index:{market_df.index}')
    #logger.info(f'start_date:{type(start_date)}')
    #logger.info(f'asset:{asset}, start_date:{start_date}, end_date:{end_date}')
    #logger.info(f'news_df_draw_wordcloud:{news_df}')
    if asset.lower() == "all assets":
        headlines100k = news_df[start_date:end_date]['headline'].str.lower(
        ).values[-100000:]
    else:
        headlines100k = news_df.loc[news_df["assetName"] ==
                                    asset].loc[start_date:end_date, "headline"].str.lower().values[-100000:]
    #logger.info(f'asset:{asset}')
    #logger.info(f'news:{headlines100k}')
    text = ' '.join(
        headline for headline in headlines100k if type(headline) == str)
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
        #logger.info(f'g:{g}')
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

        draw_wordcloud(selected_asset, *time_period)

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

    @st.cache(allow_output_mutation=True)
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

    assets = list(news_df.assetName.unique())
    logger.info(f'assets:{assets}')
    selected_assets = st.sidebar.multiselect(
        "Please select the assets",
        assets,
        default=['ES']
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

    logger.info(f'assets_sentiment_dict:{assets_sentiment_dict}')
    sentiment_df = pd.DataFrame.from_dict(
        assets_sentiment_dict, orient='index', columns=sent_labels)
    sentiment_df = pd.melt(sentiment_df.rename_axis('asset').reset_index(), id_vars=[
                           "asset"], value_vars=sent_labels, var_name='sentiment', value_name='count')
    logger.info(f'sentiment_df:{sentiment_df}')
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

    def calculate_mean_sentiment(asset, period="M"):
        X = []
        Y = []
        asset_news_df = news_df[news_df["assetName"] == asset]

        for name, group in asset_news_df.groupby(pd.Grouper(freq=period)):
            d = name.strftime("%b '%y")
            counts = group["sentimentClass"].value_counts()
            logging.info(f'counts:{counts}')
            logging.info(f'counts_index:{counts.index}')
            counts.index = counts.index.astype("int8")
            mean_sentiment_score = np.average(counts.index, weights=counts)
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
