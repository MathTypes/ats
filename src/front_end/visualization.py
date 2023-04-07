
import datetime
import logging

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import pyLDAvis
import streamlit as st
from textblob import TextBlob
from util import nlp_utils
from wordcloud import WordCloud

from data.front_end_utils import (analyze_token_sentiment, display_text,
                                  get_list_ner, get_top_n_bigram, result_to_df,
                                  subject_analysis, visualize_ner)
from data_analysis.topic_modeling import LatentDirichletAllocation

def render_visualization(news_df, start_day, end_day):
    logging.info(f'start_day:{dir(start_day)}')
    news_df["time_str"] = news_df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    type = st.sidebar.radio("information type:", ("General", "Detailed"))
    s_d = st.sidebar.date_input("Start:", datetime.date(2023, 3, 1))
    e_d = st.sidebar.date_input("End:", datetime.date(2023, 4, 22))
    start_day = pd.to_datetime(start_day).tz_localize('utc')
    end_day = pd.to_datetime(end_day).tz_localize('utc')
    logging.info(f'start_day:{start_day}, end_day:{end_day}')
    sub_data = news_df[news_df["time"].between(start_day, end_day)]
    render_visualization_df(sub_data)

def render_visualization_df(sub_data):
    count = sub_data.shape[0]
    #logging.info(f'rendering:{sub_data}')
    # =================================================================================== #
    #                                General                                              #
    # =================================================================================== #
    if type == "General":
        # =================================================================================== #
        #                                Wordcloud                                            #
        # =================================================================================== #
        col3, col4 = st.columns(2)
        with col3:
            sub_data["keyword_subject"] = sub_data["keyword_subject"].apply(
                lambda x: str(x).replace('[', '').replace(']', ''))
            sub_data["keyword_text"] = sub_data["keyword_text"].apply(
                lambda x: str(x).replace('[', '').replace(']', ''))
            st.subheader("**Subject wordcloud:**")
            cloud_text = " ".join(sub_data["keyword_subject"])
            # logging.info(f'cloud_text:{cloud_text}')
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
            # logging.info(f'cloud_text:{cloud_text}')
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
                sub_data["lemma_text"] = sub_data["keyword_text"].apply(
                    lambda x: str(x).replace('[', '').replace(']', ''))
                # logging.info(f"sub_data_lemma_text:{sub_data['lemma_text']}")
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
                        sub_data["lemma_text"].apply(lambda x: str(x).replace('[', '').replace(']', '')), ngram_range=3, n=20
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
                df = sub_data
                sub_df = subject_analysis(df)
                st.markdown("**Text analysis:**")
                fig = px.bar(sub_df, x='sub_date', y='polarity',
                             title='News subject polarity over time')
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
            st.components.v1.html(html_string, width=1300,
                                  height=800, scrolling=True)
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
                [],
            )
            if len(option)>0:
                st.markdown("**Name entity Recognition:**")
                text = sub_data["text"][sub_data["subject"] == option[0]].values[0]
                nlp = nlp_utils.get_nlp()
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
