
import plotly.graph_objs as go
from nltk.corpus import stopwords
import datetime
import pandas as pd

import logging

import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

from eda_utils import generate_color

import nlp_util
from app_dir.data_sourcing import Data_Sourcing, data_update
from app_dir.indicator_analysis import Indications
from app_dir.graph import Visualization
from tensorflow.keras.models import load_model
import gc


import plotly.express as px

import pandas as pd


import numpy as np
import plotly.express as px

import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np


def render_sentiment_analysis(market_df, news_df, assetNames, from_date, to_date, min_date, max_date):
    stop = set(stopwords.words('english'))
    analysis = st.sidebar.radio("Choose analysis", [
        "Sentiment analysis", "Data exploration", "Aggregation charts"], index=0)

    # assets = list(set(news_df.assetName.to_list()))
    assets = assetNames
    # assets_dict = dict(zip(market_df.assetName, market_df.assetCode))
    assets_dict = dict(zip(assetNames, assetNames))

    def get_asset_code(asset):
        return assets_dict.get(asset)

    def draw_wordcloud(asset="all assets", start_date=None, end_date=None):
        start_date = datetime.datetime.combine(
            start_date, datetime.datetime.min.time())
        end_date = datetime.datetime.combine(
            end_date, datetime.datetime.min.time())
        # dr = pd.date_range(start_date, end=end_date, tz='Asia/Tokyo')
        # logging.info(f'news_df.index:{news_df.index}')
        # logging.info(f'market_df.index:{market_df.index}')
        # logging.info(f'start_date:{type(start_date)}')
        # logging.info(f'asset:{asset}, start_date:{start_date}, end_date:{end_date}')
        # logging.info(f'news_df_draw_wordcloud:{news_df["time"]}')
        if asset.lower() == "all assets":
            headlines100k = news_df[news_df["time"].between(start_date, end_date)]['text'].str.lower(
            ).values[-100000:]
        else:
            headlines100k = news_df.loc[news_df["assetName"] ==
                                        asset].loc[start_date:end_date, "text"].str.lower().values[-100000:]
        # logging.info(f'asset:{asset}')
        # logging.info(f'news:{headlines100k}')
        text = ' '.join(
            headline for headline in headlines100k if type(headline) == str)
        # logging.info(f'draw_wordcloud:{text}')

        wordcloud = WordCloud(
            max_font_size=None,
            stopwords=stop,
            background_color='white',
            width=1200,
            height=850
        ).generate(text)

        fig1 = plt.figure(figsize=(3, 3))
        plt.subplot(1, 1, 1)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.subplots_adjust(wspace=.025, hspace=.025)

        # save image, display it, and delete after usage.
        plt.savefig('x', dpi=400)
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
            # logging.info(f'g:{g}')
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

            # st.info("We can see huge price fluctiations when market crashed. But this **must be  wrong**, as there was no huge crash on January 2010... which means that there must be some outliers on our data.")

        with row2_2:
            assets = assetNames
            assets.insert(0, "All assets")

            selected_asset = st.selectbox(
                "Please select the assets",
                assets,
                index=0,
            )

            time_period = st.date_input("From/To", [from_date,
                                                    to_date], min_value=from_date, max_value=to_date)

            nlp_util.draw_wordcloud(
                news_df, stop, selected_asset, *time_period)

            gc.collect()
            data_update()

            def main(app_data):
                indication = 'Predicted'

                exchange = 'Yahoo! Finance'
                app_data.exchange_data(exchange)

                st.sidebar.subheader(f'Stock Index:')
                stock_indexes = app_data.stock_indexes
                market = st.sidebar.selectbox('', stock_indexes, index=11)
                app_data.market_data(market)
                assets = app_data.stocks
                asset = f'{market} Companies'

                st.sidebar.subheader(f'{asset}:')
                equity = st.sidebar.selectbox('', assets)

                st.sidebar.subheader('Interval:')
                interval = st.sidebar.selectbox(
                    '', ('5 Minute', '15 Minute', '30 Minute', '1 Hour', '1 Day', '1 Week'), index=4)
                volitility_index = 0
                label = asset

                # st.sidebar.subheader('Trading Volatility:')
                risk = st.sidebar.selectbox(
                    '', ('Low', 'Medium', 'High'), index=volitility_index)

                # future_price = 1
                analysis = Visualization(
                    exchange, interval, equity, indication, action_model, price_model, market)
                analysis_day = Indications(exchange, '1 Day', equity, market)
                requested_date = analysis.df.index[-1]
                current_price = float(analysis.df['Adj Close'][-1])
                change = float(analysis.df['Adj Close'].pct_change()[-1]) * 100

                risks = {'Low': [analysis_day.df['S1'].values[-1], analysis_day.df['R1'].values[-1]],
                         'Medium': [analysis_day.df['S2'].values[-1], analysis_day.df['R2'].values[-1]],
                         'High': [analysis_day.df['S3'].values[-1], analysis_day.df['R3'].values[-1]], }
                buy_price = float(risks[risk][0])
                sell_price = float(risks[risk][1])

                prediction_fig = analysis.prediction_graph(asset)

                st.plotly_chart(prediction_fig, use_container_width=True)

                technical_analysis_fig = analysis.technical_analysis_graph()
                st.plotly_chart(technical_analysis_fig,
                                use_container_width=True)

            if __name__ == '__main__':
                import warnings
                # import gc
                warnings.filterwarnings("ignore")
                gc.collect()
                action_model = load_model("action_prediction_model.h5")
                price_model = load_model("price_prediction_model.h5")
                app_data = Data_Sourcing()
                main(app_data=app_data)
    elif analysis.lower() == "aggregation charts":

        selected_assets = st.sidebar.multiselect(
            "Please select the assets",
            assets,
            default=assets[0],
            format_func=get_asset_code
        )

        start_date = st.sidebar.date_input(
            "Starting date",
            value=from_date,
            min_value=min_date,
            max_value=max_date
        )

        end_date = st.sidebar.date_input(
            "End date",
            value=to_date,
            min_value=min_date,
            max_value=max_date
        )
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        if selected_assets:
            data1 = []
            for asset in selected_assets:
                # logging.info(f'asset:{asset}, start_date:{start_date}, end_date:{end_date}')
                # logging.info(f'market_df:{market_df[(market_df["assetName"] == asset)]}')
                asset_market_df = market_df[(market_df['assetName'] == asset)]
                asset_df = asset_market_df[asset_market_df["time"].between(
                    start_date, end_date)]

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

        # @st.cache_data()
        def trends_quantile_chart(market_df):
            data = []
            for i in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
                # price_df = market_df.groupby(market_df.index)[
                price_df = market_df.groupby('time')[
                    'close'].quantile(i).reset_index()
                # logging.info(f'price_df:{price_df}')
                data.append(go.Scatter(
                    x=price_df['time'].dt.strftime(
                        date_format='%Y-%m-%d').values,
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

        st.plotly_chart(dict(data=data2, layout=layout1),
                        use_container_width=True)
        with st.expander("See explanation"):
            st.write(
                """
                The graph above shows how markets fell and rise again during 2007-2016. Higher quantile prices have increased with time and lower quantile prices decreased. Maybe the gap between poor and rich increases... on the other hand maybe more "little" companies are ready to go to market and prices of their shares isn't very high.
                """
            )

    elif analysis.lower() == "sentiment analysis":
        # assets = list(news_df.assetName.unique())
        assets = assetNames
        # logging.info(f'assetNames:{assetNames}')
        # logging.info(f'assets:{assets}')
        selected_assets = st.sidebar.multiselect(
            "Please select the assets",
            assets,
            default=['ES']
        )
        logging.info(f'sentiment analysis:{from_date}, to_date:{to_date}')

        start_date, end_date = st.sidebar.date_input("Time period (from/to)",
                                                     [from_date, to_date], min_value=min_date, max_value=max_date)

        #row1_1, row1_2 = st.columns(1)
        #row2_1, row2_2 = st.columns(2)

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

        def calculate_mean_sentiment(asset, period="1h"):
            X = []
            Y = []
            logging.info(f'news_df:{news_df["assetName"]}')
            asset_news_df = news_df[news_df["assetName"] == asset]
            asset_news_df.index = pd.to_datetime(asset_news_df.index)
            logging.info(f'asset_news_df:{asset_news_df}')
            for name, group in asset_news_df.groupby(pd.Grouper(freq=period)):
                d = name.strftime("%m/%d/%Y, %H:%M")
                counts = group["sentimentClass"].value_counts()
                # logging.info(f'counts:{counts}')
                # logging.info(f'counts_index:{counts.index}')
                counts.index = counts.index.astype("int8")
                if counts.size > 0:
                    mean_sentiment_score = np.average(
                        counts.index, weights=counts)
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

        st.plotly_chart(dict(data=data, layout=layout))
