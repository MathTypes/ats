import logging
from neo4j import GraphDatabase
import pandas as pd

host = 'bolt://host.docker.internal:7687'
user = 'neo4j'
password = 'password'
driver = GraphDatabase.driver(host, auth=(user, password))


def read_query(query, params={}):
    with driver.session() as session:
        logging.error(f'query:{query}')
        logging.error(f'params:{params}')
        result = session.run(query, params)
        response = [r.values()[0] for r in result]
        return response


def get_article_text(title):
    text = read_query(
        "MATCH (a:Article {webTitle:$title}) RETURN a.bodyContent as response", {'title': title})
    return text


def get_tweets():
    query = """MATCH (t:Tweet)
            RETURN t.id as id, t.user as user,
                datetime({epochMillis: t.created_at}) as created_at,
                t.perma_link as perma_link, t.like_count as like_count,
                t.source_url as source_url, t.raw_content as raw_content,
                t.last_update as last_update"""
    params = {}
    with driver.session() as session:
        result = session.run(query, params)
        df = pd.DataFrame([r.values() for r in result], columns=result.keys())
        return df


def get_conversation_ids():
    text = read_query(
        "MATCH (n:Conversation)-[CONTAINS] - (t:Tweet) return n.id, min(t.pubdate) as start_date, max(t.pubdate) as end_date order by start_date")
    return text


def get_tweets_by_conv_id(conv_id):
    text = read_query(
        "MATCH(c:Conversation {id:$conv_id})-[CONTAINS]->(t:Tweet) return t.raw_content as text order by t.created_at DESC",
        params={"conv_id": conv_id})
    logging.error(f'text:{text}')
    return text


def get_conversations():
    conversation_ids = get_conversation_ids()
    df = pd.DataFrame(columns=['conv_id', 'text'])
    for conv_id in conversation_ids:
        tweets = get_tweets_by_conv_id(conv_id)
        df = df.append(
            {'conv_id': conv_id, 'text': "\n".join(tweets)}, ignore_index=True)
    return df
