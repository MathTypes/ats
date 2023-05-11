import tweepy
  

bearer_token = 'AAAAAAAAAAAAAAAAAAAAAAypmwEAAAAASU9euOx4vMId%2FbuLcQBNWSEnjqg%3Dj4pPU1gV2zcTL9i8nduFGnbgpT9WGJx8Kl2fc6oPJtuQ4Et8Bv'
client = tweepy.Client(bearer_token = bearer_token)
# Replace with your own search query
query = '$qqq'

# Replace with your own search query
#query = 'from:DougKass -is:retweet'

# Replace with time period of your choice
start_time = '2010-01-01T00:00:00Z'

# Replace with time period of your choice
end_time = '2023-04-29T00:00:00Z'
# Replace with your own search query
#query = 'covid -is:retweet'
# Replace with your own search query
#query = 'from:suhemparack -is:retweet'

# Replace with time period of your choice
#start_time = '2020-01-01T00:00:00Z'

# Replace with time period of your choice
#end_time = '2020-08-01T00:00:00Z'

tweets = client.search_recent_tweets(query=query, tweet_fields=['context_annotations', 'created_at'], max_results=100, wait_on_rate_limit=True)
print(f"tweets:{tweets}")
print(f"tweets.length:{tweets.length}")
exit(0)


tweets = client.search_recent_tweets(query=query, tweet_fields=['context_annotations', 'created_at'], max_results=10)
print(f"tweets:{tweets}")
exit(0)


# Replace the limit=1000 with the maximum number of Tweets you want
for tweet in tweepy.Paginator(client.search_all_tweets, query=query,
                              tweet_fields=['context_annotations', 'created_at'], max_results=100).flatten(limit=1000):
    print(tweet.id)

exit(0)


for tweet in tweets.data:
    print(tweet.text)
    if len(tweet.context_annotations) > 0:
        print(tweet.context_annotations)
# Replace with your own search query
query = 'from:suhemparack -is:retweet'

# Replace with time period of your choice
start_time = '2020-01-01T00:00:00Z'

# Replace with time period of your choice
end_time = '2020-08-01T00:00:00Z'
# Replace with your own search query
query = 'covid -is:retweet'

# Replace the limit=1000 with the maximum number of Tweets you want
for tweet in tweepy.Paginator(client.search_all_tweets, query=query,
                              tweet_fields=['context_annotations', 'created_at'], max_results=100).flatten(limit=1000):
    print(tweet.id)

# Replace the limit=1000 with the maximum number of Tweets you want
for tweet in tweepy.Paginator(client.search_recent_tweets, query=query,
                              tweet_fields=['context_annotations', 'created_at'], max_results=100).flatten(limit=1000):
    print(tweet.id)
tweets = client.search_all_tweets(query=query, tweet_fields=['context_annotations', 'created_at'],
                                  start_time=start_time,
                                  end_time=end_time, max_results=100)

#client
#customer_key = "aVF2UzJjMnR3ajZvWG4tUlh1RkY6MTpjaQ"
#customer_secrete = "nveVHK4mp6pZaYFKXgGUmdFbChuUdnvZBTT2uDW8CmGs5TI2x6"
#auth = tweepy.OAuth1UserHandler(
#  consumer_key, consumer_secret, access_token, access_token_secret
#)

def get_tweets(username):
    api = tweepy.API(auth, wait_on_rate_limit=True)
    tweets = []
    last_id = None

    #tweets = client.get_users_tweets(id=id, tweet_fields=['context_annotations','created_at','geo'])
    #for tweet in tweets.data:
    #    print("\nTweet text: ", tweet.text)
    #    print("Created at: ", tweet.created_at)

    while True:
        try:
            ts = api.user_timeline(screen_name=username, count=200, max_id=last_id)
        except tweepy.errors.Unauthorized as e:
            return tweets
        except tweepy.errors.NotFound as e:
            return tweets    
        if len(ts) == 0:
            break
        tweets.extend([t._json for t in ts])
        last_id = ts[-1].id -1
    
    df = pd.json_normalize(tweets)
    #df.to_csv(f'{username}.csv')
    #print(df)
    return df

# Replace with your own search query
query = 'covid -is:retweet has:media'

tweets = client.search_recent_tweets(query=query, tweet_fields=['context_annotations', 'created_at'],
                                     media_fields=['preview_image_url'], expansions='attachments.media_keys',
                                     max_results=100)

# Get list of media from the includes object
media = {m["media_key"]: m for m in tweets.includes['media']}

for tweet in tweets.data:
    attachments = tweet.data['attachments']
    media_keys = attachments['media_keys']
    print(tweet.text)
    if media[media_keys[0]].preview_image_url:
        print(media[media_keys[0]].preview_image_url)


# Replace user ID
id = '2244994945'

tweets = client.get_users_tweets(id=id, tweet_fields=['context_annotations','created_at','geo'])

for tweet in tweets.data:
    print(tweet)
# Replace user ID
id = '2244994945'

tweets = client.get_users_mentions(id=id, tweet_fields=['context_annotations','created_at','geo'])

for tweet in tweets.data:
    print(tweet)

import tweepy

client = tweepy.Client(bearer_token='REPLACE_ME')

# Replace user ID
id = '2244994945'

users = client.get_users_followers(id=id, user_fields=['profile_image_url'])

for user in users.data:
    print(user.id)

import tweepy

client = tweepy.Client(bearer_token='REPLACE_ME')

# Replace user ID
id = '2244994945'

users = client.get_users_following(id=id, user_fields=['profile_image_url'])

for user in users.data:
    print(user.id)

