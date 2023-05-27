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

def is_power_user(username):
    return username in ['PharmD_KS','pghosh1','DougKass','JimScalpert','eliant_capital','Mayhem4Markets', 'Tricky_OW', 'DoubleWideCap']

def delete_tweets(coll_ref, batch_size):
    docs = coll_ref.list_documents(page_size=batch_size)
    deleted = 0

    for doc in docs:
        data = doc.get().to_dict()
        if not is_power_user(data["username"]):
            doc.delete()
            deleted = deleted + 1

    if deleted >= batch_size:
        return delete_tweets(coll_ref, batch_size)

#doc_ref = db.collection(u'unique_ids')
#delete_collection(doc_ref, 100)

doc_ref = db.collection(u'recent_tweet_by_user')
delete_tweets(doc_ref, 100)
exit(0)
