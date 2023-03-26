
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
from neo4j_util.sentiment_api import get_tweets, get_tweet_replies

from utils import (
    data_process,
    feature_extraction,
    visualize_ner,
    get_top_n_bigram,
    get_list_ner,
    display_text, subject_analysis, result_to_df, analyze_token_sentiment,
)

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

# =================================================================================== #
#                                Display Dataset                                      #
# =================================================================================== #
#data = pd.read_csv("dataset/process_data.csv")
data = get_tweets()
logging.info(f'data:{data}')
conv_data = get_tweet_replies()
df = data_process(data)

col1, col2 = st.columns(2)
with col1:
    bbc = Image.open("images/bbc.png")
    st.image(bbc)
with col2:
    st.header("Tweet")
st.dataframe(
        data[["id", "user", "created_at", "raw_content", "source_url", "like_count", "perma_link", "last_update"]]
    )
col3, col4 = st.columns(2)
with col4:
    st.header("Conversation")
st.dataframe(
        conv_data[["conv_id", "text"]]
    )

# =================================================================================== #
#                                Graphs visualisation                                 #
# =================================================================================== #
if viz:
    start_day = pd.to_datetime(s_d)
    end_day = pd.to_datetime(e_d)
    sub_data = df[df["sub_date"].between(start_day, end_day)]
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


