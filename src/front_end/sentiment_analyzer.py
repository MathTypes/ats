
import datetime
import gc
import logging
import os

import altair as alt
from streamlit_vega_lite import altair_component

# import function
# from streamlit_bokeh_events import streamlit_bokeh_events
import matplotlib.pyplot as plt
# from streamlit_plotly_events import plotly_events
import nlp_util
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from app_dir.data_sourcing import Data_Sourcing, data_update
from app_dir.graph import Visualization
from app_dir.indicator_analysis import Indications
from eda_utils import generate_color
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from wordcloud import WordCloud
import visualization


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
        if asset.lower() == "all assets":
            headlines100k = news_df[news_df["time"].between(start_date, end_date)]['text'].str.lower(
            ).values[-100000:]
        else:
            headlines100k = news_df.loc[news_df["assetName"] ==
                                        asset].loc[start_date:end_date, "text"].str.lower().values[-100000:]
        text = ' '.join(
            headline for headline in headlines100k if type(headline) == str)

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

            import warnings
            # import gc
            warnings.filterwarnings("ignore")
            gc.collect()
            action_model = load_model("action_prediction_model.h5")
            price_model = load_model("price_prediction_model.h5")
            app_data = Data_Sourcing()

            indication = 'Predicted'

            # st.sidebar.subheader('Asset:')
            # asset_options = sorted(['Cryptocurrency', 'Index Fund', 'Forex', 'Futures & Commodities', 'Stocks'])
            # asset = st.sidebar.selectbox('', asset_options, index = 4)

            # if asset in ['Index Fund', 'Forex', 'Futures & Commodities', 'Stocks']:
            exchange = 'Yahoo! Finance'
            app_data.exchange_data(exchange)

            # if asset == 'Stocks':
            st.sidebar.subheader(f'Stock Index:')
            stock_indexes = app_data.stock_indexes
            market = st.sidebar.selectbox('', stock_indexes, index=11)
            app_data.market_data(market)
            assets = app_data.stocks
            asset = f'{market} Companies'

            st.sidebar.subheader(f'{asset}:')
            equity = st.sidebar.selectbox('', assets)

            asset = 'Stock'

            st.sidebar.subheader('Interval:')
            interval = st.sidebar.selectbox(
                '', ('5 Minute', '15 Minute', '30 Minute', '1 Hour', '1 Day', '1 Week'), index=4)
            volitility_index = 0

            st.sidebar.subheader('Interval:')
            interval = st.sidebar.selectbox('', ('1 Minute', '3 Minute', '5 Minute', '15 Minute',
                                            '30 Minute', '1 Hour', '6 Hour', '12 Hour', '1 Day', '1 Week'), index=8)

            # volitility_index = 2

            label = asset

            st.sidebar.subheader('Trading Volatility:')
            risk = st.sidebar.selectbox(
                '', ('Low', 'Medium', 'High'), index=volitility_index)

            analysis = Visualization(
                exchange, interval, equity, indication, action_model, price_model, market)
            analysis_day = Indications(exchange, '1 Day', equity, market)
            requested_date = analysis.df.index[-1]
            current_price = float(analysis.df['Adj Close'][-1])
            change = float(analysis.df['Adj Close'].pct_change()[-1]) * 100
            requested_prediction_price = float(
                analysis.requested_prediction_price)
            requested_prediction_action = analysis.requested_prediction_action

            risks = {'Low': [analysis_day.df['S1'].values[-1], analysis_day.df['R1'].values[-1]],
                     'Medium': [analysis_day.df['S2'].values[-1], analysis_day.df['R2'].values[-1]],
                     'High': [analysis_day.df['S3'].values[-1], analysis_day.df['R3'].values[-1]], }
            buy_price = float(risks[risk][0])
            sell_price = float(risks[risk][1])

            prediction_fig = analysis.prediction_graph(asset)

            st.plotly_chart(prediction_fig, use_container_width=True)

            technical_analysis_fig = analysis.technical_analysis_graph()
            st.plotly_chart(technical_analysis_fig, use_container_width=True)

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
        assets = assetNames
        selected_assets = st.sidebar.multiselect(
            "Please select the assets",
            assets,
            default=['ES']
        )
        logging.info(f'sentiment analysis:{from_date}, to_date:{to_date}')

        start_date, end_date = st.sidebar.date_input("Time period (from/to)",
                                                     [from_date, to_date], min_value=min_date, max_value=max_date)

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
        period = datetime.timedelta(hours=1)

        def calculate_mean_sentiment(asset, period):
            X = []
            Y = []
            # logging.info(f'news_df:{news_df["assetName"]}')
            asset_news_df = news_df[news_df["assetName"] == asset]
            asset_news_df.index = pd.to_datetime(asset_news_df.index)
            # logging.info(f'asset_news_df:{asset_news_df}')
            for name, group in asset_news_df.groupby(pd.Grouper(freq=period)):
                logging.info(f'name:{name}')
                # d = name.strftime("%m/%d/%y %H:%M:%S")
                counts = group["sentimentClass"].value_counts()
                # logging.info(f'counts:{counts}')
                # logging.info(f'counts_index:{counts.index}')
                counts.index = counts.index.astype("int8")
                if counts.size > 0:
                    mean_sentiment_score = np.average(
                        counts.index, weights=counts)
                else:
                    mean_sentiment_score = 0
                X.append(name)
                Y.append(mean_sentiment_score)
            logging.info(f'x_df:{X}')
            return X, Y

        data = []
        selected_points_vec = []
        # @st.cache_data

        @st.cache_resource
        def altair_histogram(hist_data):
            color = alt.condition(alt.datum.slice == 'high-loss', alt.Color('assetName:N', scale=alt.Scale(
                domain=df.assetName.unique().tolist()), legend=None), alt.value("lightgray"))
            brushed = alt.selection_interval(encodings=["x"], name="brushed")
            opacity = alt.condition(brushed, alt.value(0.7), alt.value(0.25))
            # selected = alt.selection_single(on="mouseover", empty="none")
            # selected = alt.selection_single(on="click", empty="none", fields=['x'])
            line = alt.Chart(hist_data).mark_line().encode(
                x=alt.X('x:T', axis=None),
                y='y:Q',
                shape=alt.Shape('assetName:N', scale=alt.Scale(
                    range=['circle', 'diamond'])),
                tooltip=['x:N', 'y:N',
                         'assetName:N', 'label:N', 'pred:N'],
                opacity=opacity
                # color=alt.condition(selected, alt.value("red"), alt.value("steelblue"))
            ).add_selection(brushed).properties(
                width=1000,
                height=400
            )
            # callout = alt.Chart(hist_data.iloc[7:8]).mark_point(
            #    color='red', size=300, tooltip="Tooltip text here"
            # ).encode(
            #    x='x:T',
            #    y='y:Q'
            # )
            return line

        # logging.info(f'asset_size:{selected_assets}')
        for asset in selected_assets:
            X, Y = calculate_mean_sentiment(asset, period)
            df = pd.DataFrame({"x": X, "y": Y, "assetName": asset})
            logging.info(f'event_df:{df.shape}')
            # event_dict = altair_component(altair_chart=altair_histogram(df))
            selection = altair_component(altair_histogram(df))
            logging.info(f'selection:{selection}')
            r = selection.get("x")
            if r:
                start_day = datetime.datetime.fromtimestamp(r[0]*1000)
                start_day = pd.to_datetime(start_day).tz_localize('utc')
                to_day = datetime.datetime.fromtimestamp(r[1]*1000)
                to_day = pd.to_datetime(to_day).tz_localize('utc')
                filtered = df[(df.x >= start_day) & (df.x < to_day)]
                logging.info(f'start_day:{start_day}, end_day:{to_day}')
                logging.info(f'filter:{filtered}')
                st.write(filtered)

        if "_vgsid_" in selection:
            # the ids start at 1
            st.write(df.iloc[[selection["_vgsid_"][0] - 1]])
        else:
            st.info(
                "Hover over the chart above to see details about the Penguin here.")

        # for selected_points in selected_points_vec:
        #    if selected_points:
        #        end_day = datetime.datetime.strptime(
        #            selected_points[0]['x'], '%m/%d/%y %H:%M:%S')
        #        start_day = end_day - period
        #        logging.info(f'start_day:{start_day}, end_day:{end_day}')
        #        visualization.render_visualization(news_df, start_day, end_day)
        #    else:
        #        logging.info(f'no selection')
