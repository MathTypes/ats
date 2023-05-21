import logging
#from google.cloud import firestore_v1 as firestore
from firebase_admin import firestore
from util import logging_utils

import firebase_admin
from firebase_admin import credentials
from firebase import firebase

logging_utils.init_logging()
cred = credentials.Certificate("/Users/jianjunchen/repo/secrets/keen-rhino-386415-firebase-adminsdk-9jpg7-968c43811b.json")
default_app = firebase_admin.initialize_app(cred)
db = firestore.client()
firebase = firebase.FirebaseApplication('https://keen-rhino-386415.firebaseio.com', None)

def delete_collection(coll_ref, batch_size):
    docs = coll_ref.list_documents(page_size=batch_size)
    deleted = 0

    for doc in docs:
        if not "realtime" in doc.id and not "nitter" in doc.id and not "selenium_" in doc.id:
            print(f'Deleting doc {doc.id} => {doc.get().to_dict()}')
            doc.delete()
            deleted = deleted + 1
        if "nitter_recent_user_pghosh1" in doc.id:
            print(f'Deleting doc {doc.id} => {doc.get().to_dict()}')
            doc.delete()
            deleted = deleted + 1

    if deleted >= batch_size:
        return delete_collection(coll_ref, batch_size)

doc_ref = db.collection(u'unique_ids')
delete_collection(doc_ref, 100)

exit(0)
data = {'1657852295558709248': {'tweet_id': '1657852295558709248', 'username': 'davidcharting', 'name': 'David', 'profile_picture': 'https://pbs.twimg.com/profile_images/1641528445149519895/f5IC8trq_x96.jpg', 'replies': 0, 'retweets': 0, 'likes': 0, 'is_retweet': False, 'posted_time': '2023-05-14T20:56:03+00:00', 'content': 'Relatively light week until Powell speaks on Friday. Hopefully the $SPY does not stay trapped in this 404-417 range.\n\nThere is also a range within that range… first level of support is at 408.75 and first level of resistance is at 414.75.', 'hashtags': [], 'mentions': [], 'images': ['https://pbs.twimg.com/media/FwHf4BXWwAYuF9r?format=jpg&name=medium', 'https://pbs.twimg.com/media/FwHf4BTXoAAcos0?format=jpg&name=medium'], 'videos': [], 'tweet_url': 'https://twitter.com/davidcharting/status/1657852295558709248', 'link': ''}}
#doc_ref = db.collection(u'recent_tweet')
#postdata = data
# Assumes any auth/headers you need are already taken care of.
#result = firebase.post('/tweet', postdata, {'print': 'pretty'})
doc_ref = db.collection(u'recent_tweet')
for key, value in data.items():
    logging.info(f"key:{key}, value:{value}")
    doc_ref.document(key).set(value)