import pandas as pd
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import Counter
import streamlit as st
from nltk import pos_tag, word_tokenize

DATA_PATH = "./data/final_annotation.csv"
data = pd.read_csv(DATA_PATH)


def visualize_all(graph_type):
    """Data visualization
    Parameters
    ----------
    graph_type : string
        Graph type, "bar plot", "distribution plot", "word cloud"
    Returns
    -------
    graph
        The specified graph
    """
    if graph_type == "bar plot":
        plot = (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x=alt.X(alt.repeat(), type="nominal", axis=alt.Axis(labelAngle=45)),
                y="count()",
            )
            .properties(height=350, width=200)
            .repeat(repeat=["Service", "Environment", "Price"])
        )
        st.altair_chart(plot, use_container_width=True)
        return None

    if graph_type == "distribution plot":
        data_melt = data.melt(
            id_vars=["ReviewID", "Review"],
            var_name="aspect",
            value_name="label",
            value_vars=["Service", "Environment", "Price"],
        )
        my_order = [
            "Negative",
            "Neutral",
            "Positive",
            "Bad",
            "Moderate",
            "Good",
            "Cheap",
            "Reasonable",
            "Expensive",
            "Undefined",
        ]

        click = alt.selection_multi(fields=["aspect"], bind="legend")
        opacity = alt.condition(click, alt.value(0.8), alt.value(0.1))
        plot = (
            alt.Chart(data_melt)
            .mark_tick(opacity=0.6)
            .encode(
                x=alt.X("ReviewID", scale=alt.Scale(zero=False)),
                y=alt.Y("label", sort=my_order),
                color="aspect",
                opacity=opacity,
            )
            .add_selection(click)
            .properties(width=1200, height=400)
        )
        st.altair_chart(plot, use_container_width=True)
        return None

    if graph_type == "word cloud":
        text = data["Review"].str.cat(sep=" ")
        stopword_list = stopwords.words("English")
        wordcloud = WordCloud(max_font_size=40, stopwords=stopword_list).generate(text)
        fig, ax = plt.subplots()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(fig)
        return None


def visualize_selected(graph_type, feature):
    """Data visualization
    Parameters
    ----------
    graph_type : string
        Graph type, "bar plot", "distribution plot", "word cloud"
    feature : string
        "Price", "Environment", "Service"
    Returns
    -------
    graph
        The specified graph
    """
    if graph_type == "bar plot":
        plot = (
            alt.Chart(data)
            .mark_bar()
            .encode(x=alt.X(feature, axis=alt.Axis(labelAngle=360)), y="count()")
            .properties(height=450, width=150)
        )
        st.altair_chart(plot, use_container_width=True)
        return None

    if graph_type == "distribution plot":
        plot = (
            alt.Chart(data)
            .mark_tick(opacity=0.6)
            .encode(x=alt.X("ReviewID", scale=alt.Scale(zero=False)), y=alt.Y(feature))
            .properties(height=400, width=1200)
        )
        st.altair_chart(plot, use_container_width=True)
        return None

    if graph_type == "word cloud":
        total_text = data.query(f"{feature} != 'Undefined'")["Review"].str.cat(sep=" ")
        stopword_list = stopwords.words("English")
        filter_list = ["food", "place", "restaurant", "vancouver", "time"]
        text = " ".join(
            [
                w
                for w, pos in pos_tag(word_tokenize(total_text))
                if pos.startswith("NN")
                and w.lower() not in stopword_list
                and w.lower() not in filter_list
            ]
        )  # text only contains nouns and are not stop words
        wordcloud = WordCloud(max_font_size=40, stopwords=stopword_list).generate(text)
        fig, ax = plt.subplots()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(fig)
        return None


def all_plot_description(graph_type):
    st.subheader("Observations")
    if graph_type == "bar plot" or graph_type == "distribution plot":
        st.markdown(
            "Putting together, we can see that positive reviews ('Positive', 'Good') are much more than 'Neutral' \
            and 'Negative' reviews, which again, corresponds to the Pollyanna Hypothesis. Among the 'Undefined' label, \
                most of them are 'Price', indicating that people seldom comment on the price of food on Yelp."
        )

    elif graph_type == "word cloud":
        st.markdown(
            "The words that jumped out are “like”, “food”, “Vancouver”, “well”, “service”. \
            Apparently, people share their positive thoughts mostly about the food and service on Yelp. \
                This corresponds to the Pollyanna Hypothesis."
        )


def selected_plot_description(graph_type, feature):
    st.subheader("Observations")
    if graph_type == "word cloud":
        if feature == "Service":
            st.markdown(
                "Words such as “service”, “star”, “sauce” appears frequently in all three aspects, and besides these common words, \
                    “staff”, “experience”, and “people” stand out as we expect, since service is often provided by restaurant staffs \
                    in the context of restaurant reviews."
            )
        elif feature == "Environment":
            st.markdown(
                "Unlike the other two labels, environment tends to have a more wider variety of words. It includes “service”, \
                name of all kinds of foods (such as “beer”, “chicken”, “sauce”), “people”, indicating that the reviews which comment on the \
                    environment are likely to contain a lot of other information."
            )
        elif feature == "Price":
            st.markdown(
                "We can see “price”, “hour”, “order”, “menu” are mentioned a lot, which makes sense since they are all related to happy \
                hour menu. People tend to comment more on price if a restaurant serves happy hour menu. Also, the size of the word “price” is larger \
                    than all 'price' in other plots."
            )
    elif graph_type == "distribution plot":
        if feature == "Price":
            st.markdown(
                "Most of the labels are 'Undefined' and 'Reasonable'. Showing that most of the times, people agree on the price of the food \
                so they either do not comment on price or indicate that the price is reasonable."
            )
        elif feature == "Environment":
            st.markdown(
                "Good labels are significantly more than the other labels and there are only about 20 bad labels out of 400."
            )
        elif feature == "Service":
            st.markdown(
                "Positive labels are significantly much more than the other labels."
            )
    elif graph_type == "bar plot":
        if feature == "Service":
            st.markdown(
                "Positive labels are significantly much more than the other labels."
            )
        elif feature == "Environment":
            st.markdown(
                "Good labels are significantly much more than the other labels and there are only about 20 bad labels out of 400."
            )
        elif feature == "Price":
            st.markdown(
                "Most of the labels are 'Undefined' and 'Reasonable'. Showing that most of the times, people agree on the price of the food \
                so they either do not comment on price or indicate that the price is reasonable."
            )
