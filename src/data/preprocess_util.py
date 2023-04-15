import chart_studio
import re
import string
import emot
import collections
import ipywidgets
import contractions
import cufflinks
import nltk.tokenize

import pandas as pd
import numpy as np

from textblob import TextBlob
from google.colab import widgets
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


def pre_process(text):
    # Remove links
    text = re.sub("http://\S+|https://\S+", "", text)
    text = re.sub("http[s]?://\S+", "", text)
    text = re.sub(r"http\S+", "", text)

    # Convert HTML references
    text = re.sub("&amp", "and", text)
    text = re.sub("&lt", "<", text)
    text = re.sub("&gt", ">", text)
    # text = re.sub('\xao', ' ', text)

    # Remove new line characters
    text = re.sub("[\r\n]+", " ", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags
    text = re.sub(r"#\w+", "", text)

    # Remove multiple space characters
    text = re.sub("\s+", " ", text)

    # Convert to lowercase
    text = text.lower()
    return text


def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words="english").fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def drop_word(common_words):
    df1 = pd.DataFrame(common_words, columns=["TweetText", "count"])
    df1.groupby("TweetText").sum()["count"].sort_values(ascending=False).iplot(
        kind="bar",
        yTitle="Count",
        linecolor="black",
        title="Top 20 bigrams in Tweet before removing spams",
    )

    to_drop = [
        "LP LOCKED",
        "accumulated 1 ETH",
        "This guy accumulated over $100K",
        "help me sell a nickname",
        "As A Big Fuck You To The SEC",
        "Wanna be TOP G",
        "#walv",
        "#NFTProject",
        "#1000xgem",
        "$GALI",
        "NFT",
        "What the Soul of USA is",
        "#BUSD",
        "$FXMS",
        "#fxms",
        "#Floki",
        "#FLOKIXMAS",
        "#memecoin",
        "#lowcapgem",
        "#frogxmas",
        "Xmas token",
        "crypto space",
        "Busd Rewards",
        "TRUMPLON",
        "NO PRESALE",
        "#MIKOTO",
        "$HATI",
        "$SKOLL",
        "#ebaydeals",
        "CHRISTMAS RABBIT",
        "@cz_binance",
        "NFT Airdrop",
        "#NFT",
    ]
    df = df[~df["text"].str.contains("|".join(to_drop))]

    df["expanded_text"] = df["text"].apply(expand_contractions)
    df["processed_text"] = df["expanded_text"].apply(pre_process)
    df = df.drop("id", axis=1)
    return df


def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words="english").fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


# Define a function to extract emoticons


def extract_emoticons(text):
    res = emot_obj.emoji(text)
    return res["value"]


def apply_emoticons(df):
    # Apply the function to each row of the 'text' column
    df["emoticons"] = df["text"].apply(extract_emoticons)
    df["emoticons"].apply(lambda x: collections.Counter(x))
    combined_counts = sum(
        df["emoticons"].apply(lambda x: collections.Counter(x)), collections.Counter()
    )
    emoji_dict = dict(combined_counts)
    sorted_emoji_dict = dict(
        sorted(emoji_dict.items(), key=lambda x: x[1], reverse=True)
    )
    d = {k: v for i, (k, v) in enumerate(sorted_emoji_dict.items()) if i < 20}
    df_emojis = pd.DataFrame(list(d.items()), columns=["Emojis", "Count"])
    df_emojis.at[5, "Emojis"] = "❤️"
    df_emojis.at[6, "Emojis"] = "🤡"

    df_emojis.groupby("Emojis").sum()["Count"].sort_values(ascending=False).iplot(
        kind="bar",
        xTitle="Emojis",
        yTitle="Count",
        linecolor="black",
        title="The 20 most used emojis after removing spam",
    )


def apply_sentiment(df):
    df["vader_polarity"] = df["processed_text"].map(
        lambda text: sid.polarity_scores(text)["compound"]
    )
    df["blob_polarity"] = df["processed_text"].map(
        lambda text: TextBlob(text).sentiment.polarity
    )
    new_df = df[["vader_polarity", "blob_polarity"]]
    new_df = new_df.rename(
        columns={"vader_polarity": "Vader", "blob_polarity": "TextBlob"}
    )


stop_words = nltk.corpus.stopwords.words("english")


def remove_stop_words(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join([word for word in text.split() if word.lower() not in stop_words])


def apply_topics(df):
    df["stop_text"] = df["processed_text"].apply(lambda x: remove_stop_words(x))
    # We define a list of topics
    topics = [
        "free speech",
        "hunter biden",
        "twitter files",
        "freedom speech",
        "right wing",
        "donald trump",
    ]

    vader_sentiments = df["vader_polarity"].tolist()
    textblob_sentiments = df["blob_polarity"].tolist()
    text = df["stop_text"].tolist()

    # We create a new column Topic
    df["Topic"] = ""
    for topic in topics:
        df.loc[df["stop_text"].str.contains(topic), "Topic"] = topic

    # We create a new DataFrame with columns topic / sentiment / source
    data = []
    for topic in topics:
        topic_rows = df[df["Topic"] == topic]
        # Average sentiment per topic
        vader_sentiments = topic_rows["vader_polarity"].sum() / topic_rows.shape[0]
        textblob_sentiments = topic_rows["blob_polarity"].sum() / topic_rows.shape[0]
        # Append data
        data.append({"Topic": topic, "Sentiment": vader_sentiments, "Source": "Vader"})
        data.append(
            {"Topic": topic, "Sentiment": textblob_sentiments, "Source": "TextBlob"}
        )

    df_new = pd.DataFrame(data)


# https://github.com/Wazzabeee/NLP_Unsupervised_Sentiment_Analysis_Elon_Musk/blob/main/500_000_Tweets_on_Elon_Musk.ipynb


def expand_contractions(text):
    try:
        return contractions.fix(text)
    except:
        return text
