from functools import lru_cache
import logging
from neo4j import GraphDatabase
import os
import pandas as pd
from data.front_end_utils import (
    data_process,
    feature_extraction,
    visualize_ner,
    get_top_n_bigram,
    get_list_ner,
    display_text, subject_analysis, result_to_df, analyze_token_sentiment,
)
from nlp import keyword_util
from util import config_utils
from neo4j_util import driver

def map_to_market(x):
    for word in x:
        word = word.lower()
        if word in ["wfc", "bac", "banks", "market", "equity"]:
            return "ES"
        if word in ["nvda", "tsla", "tesla"]:
            return "NQ"
        if word in ["risk", "volatility"]:
            return "ES"
    return "ES"


def read_query(query, params={}):
    with driver.get_driver().session() as session:
        result = session.run(query, params)
        response = [r.values()[0] for r in result]
        return response


def get_article_text(title):
    text = read_query(
        "MATCH (a:Article {webTitle:$title}) RETURN a.bodyContent as response", {'title': title})
    return text


def get_tweet_id_by_range(start_date, end_date):
    query = """
            MATCH (t:Tweet)
            WHERE datetime({epochMillis: t.created_at})  > datetime($start_date)
              and datetime({epochMillis: t.created_at})  < datetime($end_date)
            RETURN t.id as id
            """
    text = read_query(query, params={"start_date": start_date,
                                     "end_date": end_date})
    return text


def get_tweets():
    query = """
            MATCH (t:Tweet)
            MATCH (rt:RepliedTweet)
            WHERE t.created_at is not null and rt.tweet_id=t.id and t.raw_content is not null
             and t.created_at is not null and not ("" in t.lemma_text)
             and rt.raw_content is not null
             and not ("" in t.keyword_subject)
            RETURN t.id as id, t.user as user,
                datetime({epochMillis: t.created_at}) as time,
                t.perma_link as perma_link, t.like_count as like_count,
                t.source_url as source_url, (rt.raw_content + t.raw_content) as text,
                t.last_update as last_update, t.keyword_subject as keyword_subject,
                t.lemma_text as lemma_text, t.keyword_text as keyword_text
            ORDER BY t.created_at DESC
            LIMIT 5
            """
    params = {}
    with driver.get_driver().session() as session:
        result = session.run(query, params)
        result_dict = [r.values() for r in result]
        # logging.info(f"result:{result_dict}")
        df = pd.DataFrame(result_dict, columns=result.keys())
        df["time"] = df["time"].apply(lambda x: x.to_native())
        df["time"] = pd.to_datetime(
            #    df["time"], infer_datetime_format=True).dt.date
            df["time"], infer_datetime_format=True)
        # logging.info(f'original_df_text:{df["text"]}')
        df["text"] = df["text"].apply(lambda x: str(x))
        df["assetName"] = df["keyword_subject"].apply(map_to_market)
        df["assetCode"] = df["assetName"]
        # logging.info(f'df_text:{df["text"]}')
        df = subject_analysis(df)
        # df["sentimentClass"] = df["sentimentClass"].apply(map_sentiment)
        # df["assetName"] = "Stocks"
        # df = keyword_util.add_subject_keyword(df)
        return df

@lru_cache
def get_processed_tweets_from_monthly(from_date, end_date):
    df_vec = []
    for month in pd.period_range(from_date, end_date, freq='M'):
        logging.info(f'month:{month}')
        path_dir = os.path.join(config_utils.get_ts_root(), "..", "news", "monthly", "twitter")
        month_file = os.path.join(path_dir,
                             month.strftime("%Y%m") + '.parquet')
        logging.info(f'reading:{month_file}')
        df_vec.append(pd.read_parquet(month_file))
    df = pd.concat(df_vec)
    logging.info(f'reading_df:{df}')
    logging.info(f'df_columns:{df.columns}')
    df = df.sort_index()
    logging.info(f'duplicate index:{df[df.index.duplicated()]}')
    df = df[from_date:end_date]
    df["time"] = df.index
    #df.index = df.index.astype('str')
    #df.drop_duplicates(subset=None, keep="first", inplace=True)
    # TODO(jeremy): Replace following line with proper duplicate. We should
    # keep one instead of dropping them all
    df = df[~df.index.duplicated()]
    return df


@lru_cache
def get_processed_tweets(start_date, end_date):
    query = """
            MATCH (t:Tweet)
            MATCH (p:Person)
            WHERE t.created_at is not null and t.full_text is not null
             and not ("" in t.keyword_subject)
             and p.name=t.user
            RETURN t.id as id, t.user as user,
                datetime({epochMillis: t.created_at}) as time,
                t.perma_link as perma_link, t.like_count as like_count,
                t.source_url as source_url, t.full_text as text,
                t.keyword_subject as keyword_subject,
                t.lemma_text as lemma_text, t.keyword_text as keyword_text,
                t.subject as subject,
                t.text_ner_names as text_ner_names,
                t.text_ner_count as text_ner_count,
                t.subject_ner_names as subject_ner_names,
                t.subject_ner_count as subject_ner_count,
                p.rating as analyst_rating
            ORDER BY t.created_at DESC
            LIMIT 10000
            """
    #params={"start_date": start_date, "end_date": end_date}
    params = {}
    with driver.get_driver().session() as session:
        result = session.run(query, params)
        result_dict = [r.values() for r in result]
        # logging.info(f"result:{result_dict}")
        df = pd.DataFrame(result_dict, columns=result.keys())
        df["time"] = df["time"].apply(lambda x: x.to_native())
        df["time"] = pd.to_datetime(
            #    df["time"], infer_datetime_format=True).dt.date
            df["time"], infer_datetime_format=True)
        # logging.info(f'original_df_text:{df["text"]}')
        df["text"] = df["text"].apply(lambda x: str(x))
        df["text_ner_names"] = df["text_ner_names"].apply(lambda x: str(x))        
        df["text_ner_count"] = df["text_ner_count"].apply(lambda x: str(x))        
        df["assetName"] = df["keyword_subject"].apply(map_to_market)
        df["assetCode"] = df["assetName"]
        df["analyst_rating"] = df['analyst_rating'].fillna(0)
        # logging.info(f'df_text:{df["text"]}')
        df = subject_analysis(df)
        # df["sentimentClass"] = df["sentimentClass"].apply(map_sentiment)
        # df["assetName"] = "Stocks"
        # df = keyword_util.add_subject_keyword(df)
        df['index_time'] = df["time"]
        df = df.set_index("index_time")
        df = df.sort_index()
        return df

def update_tweets_unprocessed_for_reply():
    query = """
            MATCH (rt:Tweet) with rt
            MATCH (t:Tweet)            
            WHERE t.reply_process_time is null and rt.in_reply_to_tweet_id=t.id
            MERGE (rt)-[r:Reply]->(t)
            SET t.reply_process_time=datetime()
            RETURN t
            LIMIT 10S;
            RETURN tweet_id, time, text
            LIMIT 100
            """
    params = {}
    with driver.get_driver().session() as session:
        result = session.run(query, params)
        df = pd.DataFrame([r.values() for r in result], columns=result.keys())
        df["time"] = df["time"].apply(lambda x: x.to_native())
        df["time"] = pd.to_datetime(
            df["time"], infer_datetime_format=True)
        df["text"] = df["text"].apply(lambda x: str(x))
        # df = keyword_util.add_subject_keyword(df)
        return df

def get_unprocessed_tweets():
    query = """
            MATCH (t:Tweet)<-[r:Reply]-(t1:Tweet)
            WHERE t.annotation_time is null and t.raw_content is not null
            WITH t
            LIMIT 100
            MATCH (rt:Tweet )-[r:Reply*..3]-(t)            
            with t.id as tweet_id, t.raw_content + collect(rt.raw_content) as text,
            datetime({epochMillis: t.created_at}) as time
            RETURN tweet_id, time, text
            LIMIT 100
            """
    params = {}
    with driver.get_driver().session() as session:
        result = session.run(query, params)
        df = pd.DataFrame([r.values() for r in result], columns=result.keys())
        df["time"] = df["time"].apply(lambda x: x.to_native())
        df["time"] = pd.to_datetime(
            df["time"], infer_datetime_format=True)
        df["text"] = df["text"].apply(lambda x: str(x))
        # df = keyword_util.add_subject_keyword(df)
        return df


def get_gpt_unprocessed_tweets():
    query = """MATCH (t:Tweet)
            WHERE t.last_gpt_process_time is null
            RETURN t.id as id, t.user as user,
                datetime({epochMillis: t.created_at}) as time,
                t.perma_link as perma_link, t.like_count as like_count,
                t.source_url as source_url, t.raw_content as text,
                t.last_update as last_update
            LIMIT 1000
            """
    params = {}
    with driver.get_driver().session() as session:
        result = session.run(query, params)
        df = pd.DataFrame([r.values() for r in result], columns=result.keys())
        df["time"] = df["time"].apply(lambda x: x.to_native())
        df["time"] = pd.to_datetime(
            df["time"], infer_datetime_format=True)
        return df


def get_conversation_ids():
    text = read_query(
        "MATCH (n:Conversation)-[CONTAINS] - (t:Tweet) return n.id, min(t.pubdate) as start_date, max(t.pubdate) as end_date order by start_date")
    return text


def get_tweet_ids_with_reply():
    text = read_query(
        "MATCH (n:Tweet) with n.in_reply_to_tweet_id as tweet_id, count(*) as replies where replies > 1 return tweet_id")
    return text


def get_tweets_by_conv_id(conv_id):
    text = read_query(
        "MATCH(c:Conversation {id:$conv_id})-[CONTAINS]->(t:Tweet) return t.raw_content as text order by t.created_at DESC",
        params={"conv_id": conv_id})
    # logging.error(f'text:{text}')
    return text


def get_tweets_replied_to(tweet_id):
    text = read_query(
        "MATCH(t:Tweet {in_reply_to_tweet_id:$tweet_id}) return t.raw_content as text order by t.created_at DESC",
        params={"tweet_id": tweet_id})
    return text


def map_sentiment(x):
    if x.lower() in ["negative"]:
        return -1
    if x.lower() in ["positive"]:
        return 1
    return 0


def get_gpt_sentiments():
    query = """
    MATCH (t:Tweet)-[r:SENTIMENT]->(e:Entity)
    MATCH (rt:RepliedTweet)
    WHERE t.created_at is not null and rt.tweet_id=t.id and t.raw_content is not null
    RETURN t.id as tweet_id, datetime({epochMillis: t.created_at}) as time,
            (t.raw_content + rt.text) as text, t.perma_link as perma_link,
            r.class as sentimentClass, r.rank as score, e.name as assetName, e.type as entity_type,
            rt.lemma_text as lemma_text,
            rt.keyword_text as keyword_text,
            rt.keyword_subject as keyword_subject
    ORDER BY t.created_at DESC
            """
    params = {}
    with driver.get_driver().session() as session:
        result = session.run(query, params)
        df = pd.DataFrame([r.values() for r in result], columns=result.keys())
        # df["assetName"] = df["assetName"].lower()
        df["time"] = df["time"].apply(lambda x: x.to_native())
        df["text"] = df["text"].apply(lambda x: str(x))
        df["time"] = pd.to_datetime(
            #    df["time"], infer_datetime_format=True).dt.date
            df["time"], infer_datetime_format=True)
        df["assetName"] = df["assetName"].apply(map_to_market)
        df["sentimentClass"] = df["sentimentClass"].apply(map_sentiment)
        df["assetCode"] = df["assetName"]
        # df["assetName"] = "Stocks"
        # df = keyword_util.add_subject_keyword(df)
        # logging.info(f'my_df:{df}')
        return df


def get_gpt_processed_replied_tweets():
    query = """
            MATCH (t:RepliedTweet), (r:Tweet)
            WHERE t.last_gpt_process_time is not null and r.id=t.tweet_id
            RETURN t.tweet_id as tweet_id, datetime({epochMillis: r.created_at}) as time,
            t.replies as replies, (r.raw_content+t.text) as text, r.perma_link as perma_link
            """
    params = {}
    with get_driver().session() as session:
        result = session.run(query, params)
        df = pd.DataFrame([r.values() for r in result], columns=result.keys())
        df["time"] = df["time"].apply(lambda x: x.to_native())
        df["time"] = pd.to_datetime(
            df["time"], infer_datetime_format=True)
        # df = keyword_util.add_subject_keyword(df)
        return df


def get_gpt_unprocessed_replied_tweets():
    query = """
            MATCH (t:RepliedTweet), (r:Tweet)
            WHERE t.last_gpt_process_time is null and r.id=t.tweet_id
            RETURN t.tweet_id as tweet_id, datetime({epochMillis: r.created_at}) as time,
            t.replies as replies, (r.raw_content+t.text) as text, r.perma_link as perma_link
            ORDER BY r.create_at DESC
            LIMIT 10;
            """
    params = {}
    with driver.get_driver().session() as session:
        result = session.run(query, params)
        df = pd.DataFrame([r.values() for r in result], columns=result.keys())
        df["time"] = df["time"].apply(lambda x: x.to_native())
        df["time"] = pd.to_datetime(
            #    df["time"], infer_datetime_format=True).dt.date
            df["time"], infer_datetime_format=True)
        return df


def get_conversations():
    conversation_ids = get_conversation_ids()
    df = pd.DataFrame(columns=['conv_id', 'text'])
    for conv_id in conversation_ids:
        tweets = get_tweets_by_conv_id(conv_id)
        df = df.append(
            {'conv_id': conv_id, 'text': "\n".join(tweets)}, ignore_index=True)
    return df


def get_tweet_replies():
    tweet_ids = get_tweet_ids_with_reply()
    df = pd.DataFrame(columns=['conv_id', 'text'])
    for conv_id in tweet_ids:
        tweets = get_tweets_replied_to(conv_id)
        df = df.append(
            {'conv_id': conv_id, 'text': "\n".join(tweets)}, ignore_index=True)
    return df


def get_tweet_replies_v2():
    query = "MATCH (t:Tweet) with t.in_reply_to_tweet_id as tweet_id, min(datetime({epochMillis: t.created_at})) as time, count(*) as replies,collect(t.raw_content) as text WHERE replies > 1 and t.created_at is not null RETURN tweet_id, time, replies, text"
    params = {}
    with driver.get_driver().session() as session:
        result = session.run(query, params)
        df = pd.DataFrame([r.values() for r in result], columns=result.keys())
        df["text"] = df["text"].apply(lambda x: "\n".join(x))
        # logging.info(f'df:{df}')
        df["time"] = df["time"].apply(lambda x: x.to_native())
        df["time"] = pd.to_datetime(
            df["time"], infer_datetime_format=True)
        df = df.set_index("time")
        df = df.sort_index()
        # df = keyword_util.add_subject_keyword(df)
        # logging.info(f'df:{df}')
        return df
    # result = read_query(query)
    # result["text"] = result["text"].apply(lambda x: "\n".join(x))
    # return result
