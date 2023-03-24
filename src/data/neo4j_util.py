import json
import logging
import sys

from py2neo import Graph, Node, NodeMatcher, Relationship
from ratelimiter import RateLimiter

# https://github.com/akashjorss/Twitter_Sentiment_Analysis_Using_Neo4j


class Neo4j:
    def __init__(self):
        # initialize the self.graph
        self.graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
        # self.graph = Graph(scheme="bolt", host="localhost", port=7687, secure=True, auth=('neo4j', 'password'))
        self.matcher = NodeMatcher(self.graph)

    def delete_all(self):
        self.graph.delete_all()

    @RateLimiter(max_calls=5, period=1)
    def load_data(self, tweet):
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
        # begin transaction
        tx = self.graph.begin()
        logging.error(f'tweet:{tweet}')
        # retrieve company node from the remote self.graph
        user_node = self.graph.evaluate(
            "MATCH(n:User) WHERE n.name = '{user}' return n", user=tweet.user.username)
        # if remote node is null, create company node
        if user_node is None:
            user_node = Node("User", name=tweet.user.username)
            tx.create(user_node)
            # print("Node created:", company)

        # repeat above for all nodes
        tweet_node = self.graph.evaluate(
            f"MATCH(n:Tweet) WHERE n.id = {tweet.id} return n")
        if tweet_node is not None:
            tx.commit()
            return

        retweetedTweetId = tweet.retweetedTweet.id if tweet.retweetedTweet else None
        tweet_node = Node("Tweet", id=tweet.id,
                          pubdate=str(tweet.date),
                          permalink=tweet.url,
                          lang=tweet.lang,
                          source=tweet.source,
                          source_url=tweet.sourceUrl,
                          like_count=tweet.likeCount,
                          quote_count=tweet.quoteCount,
                          reply_count=tweet.replyCount,
                          retweet_count=tweet.retweetCount,
                          retweetedTweet=retweetedTweetId,
                          raw_content=tweet.rawContent,
                          rendered_content=tweet.renderedContent)
        tx.create(tweet_node)
        # print("Node created:", tweet_node)

        conversation_node = self.graph.evaluate(
            f"MATCH(n:Conversation) WHERE n.id = {tweet.conversationId} return n")
        if conversation_node is None:
            conversation_node = Node("Conversation", id=tweet.conversationId)
            tx.create(conversation_node)
            child = Relationship(conversation_node, "CHILD", tweet_node)
            tx.create(child)

            contains_tweet = Relationship(
                tweet_node, "CONTAINS", conversation_node)
            tx.create(contains_tweet)
            # print("Node created:", tweet_node)

        # create relationships
        # check if describes already exists
        # describes = Relationship(tweet_node, "DESCRIBES", company)
        # created_on = Relationship(tweet_node, "CREATED_ON", datetime)
        # tx.create(describes)
        # tx.create(created_on)
        # print("Relationships created")

        # create hashtag nodes and connect them with tweet nodes
        if tweet.hashtags:
            for hashtag in tweet.hashtags:
                hashtag_node = self.matcher.match(
                    "Hashtag", name=hashtag).first()
                # hashtag_node = self.graph.evaluate("MATCH(n) WHERE n.name = {hashtag} return n", hashtag=hashtag)
                if hashtag_node is None:
                    hashtag_node = Node("Hashtag", name=hashtag)
                    tx.create(hashtag_node)
                    about = Relationship(hashtag_node, "ABOUT", tweet_node)
                    tx.create(about)

                contains_hashtag = Relationship(
                    tweet_node, "CONTAINS", hashtag_node)
                tx.create(contains_hashtag)

        # commit transaction
        tx.commit()

    def bulk_load(self, tweets):
        """
        Bulk loads list of tweets
        :param self:
        :param tweets:
        :return:
        """
        for t in tweets:
            self.load_data(t)
            print("Tweet loaded into neo4j")

    def prune_graph(self):
        self.graph.evaluate('MATCH (t:Tweet)-[:CONTAINS]->(n) WITH n as n, count(t) as tweet_count WHERE tweet_count '
                            '< 2 DETACH DELETE n')
        print('Graph pruned!')
