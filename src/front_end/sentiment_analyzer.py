
import datetime
import functools
import gc
import logging
import os
import spacy_streamlit
import altair as alt
from streamlit_vega_lite import altair_component, vega_lite_component
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

import panel as pn
pn.extension('vega')
import visualization
import matplotlib.pyplot as plt
from util import nlp_utils
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
# from app_dir.data_sourcing import Data_Sourcing, data_update
from app_dir.data_sourcing import Data_Sourcing
from app_dir.graph import Visualization
from app_dir.indicator_analysis import Indications
from eda_utils import generate_color
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from wordcloud import WordCloud
import visualization


@functools.lru_cache
def cached_load_model(file_path):
    return load_model(file_path)


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
        start_date, end_date = st.sidebar.date_input("Time period (from/to)",
                                                     [from_date, to_date], min_value=min_date, max_value=max_date)
        logging.info(f'sentiment analysis:{start_date}, to_date:{end_date}')

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
        period = datetime.timedelta(minutes=5)

        def calculate_mean_sentiment(asset, period, analyst_rating):
            X = []
            Y = []
            # logging.info(f'news_df:{news_df["assetName"]}')
            asset_news_df = news_df[(news_df["assetName"] == asset) & (
                news_df["analyst_rating"] == analyst_rating)]
            asset_news_df.index = pd.to_datetime(asset_news_df.index)
            # logging.info(f'asset_news_df:{asset_news_df}')
            # logging.info(f"groupby:{asset_news_df.groupby(pd.Grouper(freq=period))}")
            for name, group in asset_news_df.groupby(pd.Grouper(freq=period)):
                # logging.info(f'name:{name}, type:{type(name)}')
                # logging.info(f'group:{group}')
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
            # logging.info(f'X:{X}, Y:{Y}')
            sentiment_df = pd.DataFrame.from_dict({"time": X, "sentiment": Y})
            # logging.info(f'sentiment_df_before_reindex:{sentiment_df}')
            sentiment_df = sentiment_df.set_index(["time"])
            sentiment_df = sentiment_df.sort_index()
            # sentiment_df.index = pd.to_datetime(sentiment_df.index)
            # logging.info(f'sentiment_df:{sentiment_df}')
            return sentiment_df

        data = []
        selected_points_vec = []
        # @st.cache_data

        # @functools.lru_cache(maxsize=32)
        # st.cache_data()
        def get_analysis(asset, start_date, end_date):
            logging.info(
                f'calling get_analysis, asset:{asset}')
            action_model = cached_load_model("action_prediction_model.h5")
            price_model = cached_load_model("price_prediction_model.h5")
            analysis = Visualization(
                "5 Minute", asset, action_model, price_model)
            analysis_day = Indications('1 Day', asset)
            prediction_fig = analysis.prediction_graph(asset)
            return analysis, analysis_day, prediction_fig 

        @st.cache_resource
        def altair_histogram(hist_data, sentiment_data):
            logging.info(f'hist_data:{hist_data.describe()}')
            logging.info(f'sentiment_data:{sentiment_data.describe()}')
            x_init = pd.to_datetime([hist_data["eventTime"].min(), hist_data["eventTime"].max()]).astype(int)
            logging.info(f'x_init:{x_init}')
            brush = alt.selection(type='interval', encodings=['x'], init={'x':list(x_init)})
            open_close_color = alt.condition("datum.Open <= datum.Close",
                                     alt.value("#06982d"),
                                     alt.value("#ae1325"))

            base = alt.Chart(hist_data).encode(
                x = 'eventTime:T',
                color=open_close_color,
            )
            
            rule = base.mark_rule().encode(
                alt.Y(
                    'Low:Q',
                    title='Price',
                    scale=alt.Scale(zero=False),
                ),
                alt.Y2('High:Q')
            )

            bar = base.mark_bar().encode(
                alt.Y('Open:Q'),
                alt.Y2('Close:Q')
            )

            market = (rule + bar).properties(
                width=400,
                height=300
            ).add_selection(brush)
            #market = rule
            #market = base

            sentiment_brush = alt.selection(type='interval', encodings=['x'])            
            sentiment = alt.Chart(sentiment_data).mark_line(interpolate="monotone", point=True).encode(
                x = alt.X('yearmonthdatehoursminutes(eventTime):T', scale=alt.Scale(domain=brush)),
                y = alt.Y('mean(sentimentClass):Q'),
                color = alt.Color('analyst_rating:N')
            ).transform_filter(
                brush
            ).properties(
                width=400,
                height=150
            ).add_selection(sentiment_brush)
            polarity = alt.Chart(sentiment_data).mark_line(interpolate="monotone", point=True).encode(
                x = alt.X('yearmonthdatehoursminutes(eventTime):T', scale=alt.Scale(domain=brush), bin=True),
                y = alt.Y('mean(sentimentClass):Q'),
                color=alt.Color('polarity', bin=alt.Bin(step=20), legend=None),
            ).transform_filter(
                brush
            ).properties(
                width=400,
                height=150
            )

            text_brush = alt.selection(type='single', encodings=['x'])  
            ranked_text = alt.Chart(sentiment_data).mark_text(align='right').encode(
                y=alt.Y('row_number:O', axis=None),
            ).transform_filter(
                brush
            ).transform_window(
                row_number='row_number()'
            ).transform_filter(
                'datum.row_number < 15'
            ).properties(
                height=300
            )
            tweet_id = ranked_text.encode(text='id:N').properties(
                title=alt.TitleParams(text='Id', align='right')
            )
            user = ranked_text.encode(text='user:N').properties(
                title=alt.TitleParams(text='User', align='right')
            )
            polarity = ranked_text.encode(text='polarity:Q').properties(
                title=alt.TitleParams(text='Polarity', align='right'),
                width=30,
            )
            text = alt.hconcat(tweet_id, user, polarity).add_selection(text_brush)
            base_filter = alt.Chart(sentiment_data).transform_filter(
                brush
            ).transform_joinaggregate(
                total='count(*)'
            ).transform_calculate(
                pct='1 / datum.total'
            )
            polarity_hg = base_filter.mark_bar().transform_bin(
                "pbin",
                field="polarity",
                bin=alt.Bin(maxbins=20)
            ).encode(
                x='pbin:N',
                y="sum(pct):Q",
            ).properties(
                width=300,
                height=100
            )
            sentiment_hg = base_filter.mark_bar().encode(
                alt.X('sentimentClass:N'),
                alt.Y('sum(pct):Q', axis=alt.Axis(format='%'))
            ).properties(
                width=300,
                height=100
            )
            chart = alt.vconcat(data=hist_data).configure_view(
                strokeOpacity=0
            )
            row1 = alt.hconcat()
            row1 |= market
            row1 |= text
            chart &= row1
            row2 = alt.hconcat()
            row2 |= polarity_hg
            row2 |= sentiment_hg
            chart &= row2
            #row3 = alt.hconcat()
            #row3 |= polarity_hg
            #chart &= row3
            #return (market & sentiment & text) | (polarity & sentiment_hg), brush
            #return (market & sentiment & text  & polarity & sentiment_hg), brush
            return chart, brush


        #@st.cache_resource
        def altair_histogram_sentiment(sentiment_data, brush):
            base = alt.Chart(sentiment_data).encode(
                x = 'eventTime:T',
            )
            polarity = base.mark_bar().encode(
                x='pbin:N',
                y="count()",
            ).properties(
                width=300,
                height=300
            )
            sentiment = base.mark_bar().encode(
                x='sentiment:N',
                y="count()",
                color=alt.condition(brush, alt.value("black"), alt.value("lightgray"))
            ).properties(
                width=300,
                height=300
            )
            return alt.hconcat(
                    polarity,
                    sentiment,
                    data=sentiment_data
                ).transform_bin(
                "pbin",
                field="polarity",
                bin=alt.Bin(maxbins=20)
            )
            text = alt.hconcat(tweet_id, user, polarity).add_selection(text_brush) # Combine data tables
            return (market & sentiment & text)
    
        def map_dt_to_dow(x):
            return x.weekday()

        def remove_sunday():
            global viz_market_df
            viz_market_df["eventTime"] = viz_market_df["eventTime"].apply(
                lambda x: map_dt_to_dow(x)
            )
            viz_market_df["eventTime"].drop(labels=['6'])
        # gc.collect()
        # data_update()

        # import warnings
        # import gc
        # warnings.filterwarnings("ignore")
        # gc.collect()

        st.sidebar.subheader(f'Analyst:')
        analysts = news_df.user.unique().tolist()
        # stock_indexes = app_data.stock_indexes
        market = "US S&P 500"
        selected_analysts = st.sidebar.multiselect(
            "Please select analysts",
            analysts,
            default=[]
        )

        # logging.info(f'analyst:{selected_analysts}')
        # app_data.market_data(market)
        assets = assetNames
        asset = selected_assets[0]

        analysis, analysis_day, prediction_fig = get_analysis(
            asset, start_date, end_date)
        #requested_date = analysis.df.index[-1]
        #current_price = float(analysis.df['Adj Close'][-1])
        #change = float(analysis.df['Adj Close'].pct_change()[-1]) * 100
        # logging.info(f'requested_date:{requested_date}')
        # logging.info(f'current_price:{current_price}')
        # logging.info(f'change:{change}')
        # requested_prediction_price = float(analysis.requested_prediction_price)
        # requested_prediction_action = analysis.requested_prediction_action

        # st.plotly_chart(prediction_fig, use_container_width=True)

        # technical_analysis_fig = analysis.technical_analysis_graph()
        # st.plotly_chart(technical_analysis_fig, use_container_width=True)

        # logging.info(f'asset_size:{selected_assets}')
        if len(selected_analysts) > 0:
            news_df = news_df[news_df["user"].isin(selected_analysts)]
            # logging.info(f'news_df:{news_df}')
        selections = {}
        rating_period = datetime.timedelta(hours=1)
        df_vec = []
        logging.info(f'selected_assets:{selected_assets}')
        #st.altair_chart(chart, use_container_width=True)
        #df = pd.read_json(penguins_url)

        #pn_row = pn.Row(vega_pane, pn.bind(filtered_table, vega_pane.selection.param.brush))
        #tabs = pn.Tabs(hvplot_pane)
        #st.vega_lite_chart(pn_row, use_container_width=True)
        if not selected_assets:
            return
        asset = selected_assets[0]
        analyst_rating = news_df.analyst_rating.unique().tolist()
        #for rating in analyst_rating:
        #    df = calculate_mean_sentiment(asset, rating_period, rating)
        #    df["analystRating"] = rating
        #    # logging.info(f"df_shape:{df.shape}")
            # logging.info(f"analysis.df_price:{analysis.df_price.shape}")
            # logging.info(f'df:{df}')
            # logging.info(f'analysis.df_price[1:5]:{analysis.df_price.iloc[1:5]}')
            # new_df = pd.concat([df, analysis.df_price], axis=1)
            # new_df = analysis.df_price.join(df)
        #    new_df = analysis.df_price.join(df)
        #    new_df["eventTime"] = new_df.index
        #    new_df["eventTime"] = new_df["eventTime"].apply(
        #        lambda x: x.timestamp()*1000)
        #    new_df["assetName"] = asset
        #    df_vec.append(new_df)
        viz_market_df = analysis.df_price[start_date:end_date]
        viz_market_df = viz_market_df.rename(columns={"Adj Close" : "Close"})
        viz_market_df["eventTime"] = viz_market_df.index
        viz_market_df["dow"] = viz_market_df["eventTime"].apply(
                    lambda x:map_dt_to_dow(x)
        )
        viz_market_df["eventTime"] = viz_market_df["eventTime"].apply(
                    lambda x: x.timestamp()*1000)
        viz_market_df["assetName"] = asset

        # logging.info(f"new_df_shape:{new_df.shape}")
        # new_df = new_df.rename({"Adj Close":"Close"})
        # new_df['sentiment'] = new_df.x.apply(lambda sentiment: x.value // 10**9)
        # logging.info(f'new_df:{new_df}')
        # new_df = new_df.reindex(["x", "analystRating"])
        # new_df.index = new_df.index.value // 10**9
        #selection = altair_component(altair_histogram(viz_df))
        sentiment_df = news_df[(news_df["assetName"] == asset)]
        sentiment_df.index = pd.to_datetime(sentiment_df.index)
        sentiment_df["eventTime"]=sentiment_df["time"]
        sentiment_df["eventTime"] = sentiment_df["eventTime"].apply(lambda x: x.timestamp()*1000)
        sentiment_df = sentiment_df.set_index(["time"])
        sentiment_df = sentiment_df.sort_index()
        sentiment_df = sentiment_df[start_date:end_date]
        logging.info(f'viz_market_df:{viz_market_df}')
        logging.info(f'sentiment_df:{sentiment_df}')

        row1_1, row1_2 = st.columns([1, 0.5])
        with row1_1:
            chart, brush = altair_histogram(viz_market_df, sentiment_df)
            event_dict = altair_component(altair_chart=chart)
            #logging.info(f'event_dict:{event_dict}')

        #with row1_2:
            #hist_chart = altair_histogram_sentiment(sentiment_df, brush)
            #hist_event_dict = altair_component(altair_chart=hist_chart)
            #logging.info(f'event_dict:{hist_event_dict}')
        #    pass
        #start_day = datetime.datetime.fromtimestamp(r[0]*1000)
        start_day = pd.to_datetime(start_date).tz_localize('utc')
        #to_day = datetime.datetime.fromtimestamp(r[1]*1000)
        to_day = pd.to_datetime(end_date).tz_localize('utc')
        filtered = news_df[(news_df.index >= start_day) & (news_df.index < to_day)]
        gb = GridOptionsBuilder.from_dataframe(filtered)
        gb.configure_pagination(
            paginationAutoPageSize=True)  # Add pagination
        gb.configure_side_bar()  # Add a sidebar
        # Enable multi-row selection
        gb.configure_selection(
            'multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children")
        gridOptions = gb.build()

        with row1_2:
            grid_response = AgGrid(
                filtered,
                gridOptions=gridOptions,
                data_return_mode='AS_INPUT',
                update_mode='MODEL_CHANGED',
                fit_columns_on_grid_load=False,
                theme='alpine',  # Add theme color to the table
                enable_enterprise_modules=True,
                height=350,
                width='100%',
                reload_data=True
            )
            visualization.render_visualization_df(filtered)

            data = grid_response['data']
            selected = grid_response['selected_rows']
            logging.info(f'grid_response_selected:{selected}')
            df = pd.DataFrame(selected)
            # logging.info(f'selected_df:{df}')
            if not df.empty:
                selected = news_df[news_df['id'] == df[0]]
                logging.info(f'selected:{selected}')
                models = ["en_core_web_sm", "en_core_web_md"]
                #default_text = "Sundar Pichai is the CEO of Google."
                spacy_streamlit.visualize(models, selected["full_text"])
        # st.write(filtered)            
        #r = event_dict.get("x")
        #if r:
        #selections[rating] = selection

        #    data = grid_response['data']
        #    selected = grid_response['selected_rows']
        #    logging.info(f'grid_response_selected:{selected}')
        #    df = pd.DataFrame(selected)
            # logging.info(f'selected_df:{df}')
        #    if not df.empty:
        #        visualization.render_visualization_df(df)
        #    # st.write(filtered)
