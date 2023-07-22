import logging
import time


from datetime import datetime
from py2neo import Graph, Node, NodeMatcher
from ratelimiter import RateLimiter
from util import config_utils

# https://github.com/akashjorss/Twitter_Sentiment_Analysis_Using_Neo4j


class Neo4j:
    def __init__(self):
        # initialize the self.graph
        config_utils.get_args
        self.graph = Graph(
            config_utils.get_neo4j_host(),
            auth=("neo4j", config_utils.get_neo4j_password()),
        )
        # self.graph = Graph(scheme="bolt", host="localhost", port=7687, secure=True, auth=('neo4j', 'password'))
        self.matcher = NodeMatcher(self.graph)

    def delete_all(self):
        self.graph.delete_all()

    # @RateLimiter(max_calls=1, period=60)
    def update_processed_text(self, df):
        tx = self.graph.begin()
        for index, row in df.iterrows():
            row["tweet_id"]
            # retrieve company node from the remote self.graph
            self.graph.evaluate(
                """MATCH(t:Tweet {id:$tweet_id})
                   SET
                    t.subject=$subject,
                    t.lemma_text=$lemma_text,
                    t.keyword_text=$keyword_text,
                    t.symbol=$symbol,
                    t.full_text=t.raw_content + " " + $full_text,
                    t.text_ner_names=$text_ner_names,
                    t.text_ner_count=$text_ner_count,
                    t.lemma_subject=$lemma_subject,
                    t.keyword_subject=$keyword_subject,
                    t.subject_ner_names=$subject_ner_names,
                    t.subject_ner_count=$subject_ner_count,
                    t.polarity=$polarity,
                    t.sentiment=$sentiment,
                    t.sentiment_class=$sentiment_class,
                    t.annotation_time=$annotation_time
                """,
                {
                    "tweet_id": row["tweet_id"],
                    "full_text": row["text"],
                    "subject": row["subject"],
                    "lemma_text": row["lemma_text"],
                    "keyword_text": row["keyword_text"],
                    "symbol": row["symbol"],
                    "text_ner_names": row["text_ner_names"],
                    "text_ner_count": row["text_ner_count"],
                    "lemma_subject": row["lemma_subject"],
                    "keyword_subject": row["keyword_subject"],
                    "subject_ner_names": row["subject_ner_names"],
                    "subject_ner_count": row["subject_ner_count"],
                    "polarity": row["polarity"],
                    "sentiment": row["sentiment"],
                    "sentiment_class": row["sentimentClass"],
                    "annotation_time": int(datetime.now().timestamp() * 1000),
                },
            )
            # logging.info(f'result:{result}')
        tx.commit()

    @RateLimiter(max_calls=1, period=1)
    def update_gpt_entities(self, df):
        tx = self.graph.begin()
        for index, row in df.iterrows():
            row["tweet_id"]
            entity_name = row["entity_name"]
            entity_type = row["entity_class"]
            entity_node = self.graph.evaluate(
                "MERGE(n:AtsEntity {name:$entity_name, type:$entity_type}) RETURN n",
                entity_name=entity_name,
                entity_type=entity_type,
            )
            # retrieve company node from the remote self.graph
            self.graph.evaluate(
                """
                MATCH (t:Tweet {id:$tweet_id})
                MATCH (e:Entity {name:$entity_name, type:$entity_type})
                MERGE (t)-[r:SENTIMENT {class:$rate, score:$score, update_time:$update_time}]->(e)
                 RETURN r
                """,
                {
                    "tweet_id": row["tweet_id"],
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "rate": row["sentiment_class"],
                    "score": row["sentiment_score"],
                    "update_time": int(datetime.now().timestamp() * 1000),
                },
            )
        tx.commit()

    # @RateLimiter(max_calls=1, period=3)
    def load_data(self, tx, tweet):
        """
        Loads one tweet at a time
        :param tweet: a json doc with following schema
        {
            "type": "record",
            "name": "tweet",
            "keys" : [
                {"name": "company", "type": "string"},
                {"name": "sentiment", "type": "integer"},
                {"name": "id", "type": "string"},
                {"name": "date", "type": "string"},
                {"name": "time", "type": "string"},
                {"name": "retweet_count", "type": "integer"}
                {"name":"hashtags", "type":array}
                ]
        }
        :return: None
        """
        # repeat above for all nodes
        tweet_node = self.graph.evaluate(
            "MATCH(n:Tweet {id:$tweet_id}) RETURN n", tweet_id=tweet.id
        )
        if tweet_node is not None:
            logging.error(f"found tweet_node:{tweet_node}")
            return

        # retrieve company node from the remote self.graph
        # user_node = self.graph.evaluate(
        #    "MATCH(n:User {name:$user}) RETURN n",
        #    user=tweet.user.username)
        # if remote node is null, create company node
        # if user_node is None:
        #    user_node = Node("User", name=tweet.user.username,
        #                     source="tw",
        #                     id=tweet.user.id,
        #                     last_update=int(datetime.now().timestamp() * 1000))
        #    tx.create(user_node)
        # print("Node created:", company)

        # retweetedTweetId = tweet.retweetedTweet.id if tweet.retweetedTweet else None
        # Skip deleted tweets
        if not hasattr(tweet, "user"):
            return
        retweetedTweetId = None
        tweet_node = Node(
            "Tweet",
            id=tweet.id,
            user=tweet.user.username,
            user_id=tweet.user.id,
            created_at=int(tweet.date.timestamp() * 1000),
            last_update=int(datetime.now().timestamp() * 1000),
            perma_link=tweet.url,
            lang=tweet.lang,
            source=tweet.source,
            source_url=tweet.sourceUrl,
            in_reply_to_tweet_id=tweet.inReplyToTweetId,
            like_count=tweet.likeCount,
            quote_count=tweet.quoteCount,
            reply_count=tweet.replyCount,
            retweet_count=tweet.retweetCount,
            retweeted_tweet_id=retweetedTweetId,
            raw_content=tweet.rawContent,
            rendered_content=tweet.renderedContent,
        )
        tx.create(tweet_node)

        # reply_node = self.graph.evaluate(
        #    "MATCH(n:Reply {id:$conv_id}) RETURN n",
        #    conv_id=tweet.conversationId)
        # if conversation_node is None:
        #    conversation_node = Node("Conversation", id=tweet.conversationId,
        #                             last_update=int(datetime.now().timestamp()*1000))
        #    tx.create(conversation_node)
        #    child = Relationship(conversation_node, "CONTAINS", tweet_node)
        #    tx.create(child)

        # create relationships
        # check if describes already exists
        # post = Relationship(
        #    user_node, "POST", tweet_node,
        #    created_at=int(tweet.date.timestamp() * 1000),
        #    last_update=int(datetime.now().timestamp() * 1000))
        # tx.create(post)
        # created_on = Relationship(tweet_node, "CREATED_ON", datetime)
        # tx.create(describes)
        # tx.create(created_on)
        # print("Relationships created")

        # create hashtag nodes and connect them with tweet nodes
        # if tweet.hashtags:
        #    for hashtag in tweet.hashtags:
        #        hashtag_node = self.matcher.match(
        #            "Hashtag", name=hashtag).first()
        #        # hashtag_node = self.graph.evaluate("MATCH(n) WHERE n.name = {hashtag} return n", hashtag=hashtag)
        #        if hashtag_node is None:
        #            hashtag_node = Node("Hashtag", name=hashtag)
        #            tx.create(hashtag_node)
        # about = Relationship(hashtag_node, "ABOUT", tweet_node)
        # tx.create(about)

        #        contains_hashtag = Relationship(
        #            tweet_node, "TAG", hashtag_node)
        #        tx.create(contains_hashtag)

    # Yield successive n-sized
    # chunks from l.
    def divide_chunks(self, l, n):
        # looping till length l
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # @unit_of_work(timeout=60, metadata={"app_name": "tweet_upload"})
    def bulk_load(self, tweets):
        """
        Bulk loads list of tweets
        :param self:
        :param tweets:
        :return:
        """
        # begin transaction
        RETRIES = 3
        x = list(self.divide_chunks(tweets, 1000))
        for i, chunk in enumerate(x):
            count = 0
            while count < RETRIES:
                try:
                    tx = self.graph.begin()
                    for t in chunk:
                        self.load_data(tx, t)
                        print("Tweet loaded into neo4j")
                    # commit transaction
                    logging.info(f"before commit chunk {i}")
                    tx.commit()
                    logging.info(f"after commit chunk {i}")
                    break
                except Exception as e:
                    logging.error(f"bulk_load exception: {e}")
                    time.sleep(3)
                count = count + 1

    def prune_graph(self):
        self.graph.evaluate(
            "MATCH (t:Tweet)-[:CONTAINS]->(n) WITH n as n, count(t) as tweet_count WHERE tweet_count "
            "< 2 DETACH DELETE n"
        )
        print("Graph pruned!")
