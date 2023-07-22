import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase import firebase
import datetime

cred = credentials.Certificate(
    "/Users/jianjunchen/repo/secrets/keen-rhino-386415-firebase-adminsdk-9jpg7-968c43811b.json"
)
default_app = firebase_admin.initialize_app(cred)
db = firestore.client()
firebase = firebase.FirebaseApplication(
    "https://keen-rhino-386415.firebaseio.com", None
)


def create():
    today = datetime.datetime.now()
    db.collection("NYSE").document("AMZN").set(
        {
            "name": "Amazon",
            "creationDate": today,
            "lastClose": 3443.63,
            "indices": ["NDX", "OEX", "S5COND", "SPX"],
        }
    )


def read():
    doc = db.collection("NYSE").document("AMZN").get()
    return doc.to_dict()


def read_collection():
    docs = db.collection("NYSE").where("lastClose", ">", 500).stream()
    for doc in docs:
        stock = doc.to_dict()
        print(stock.name)


def update_price():
    doc_ref = db.collection("NYSE").document("AMZN").update({"lastClose": 3465.00})


def remove_indices():
    doc_ref = (
        db.collection("NYSE")
        .document("AMZN")
        .update({"indices": firestore.ArrayRemove(["NDX"])})
    )


def add_indices():
    doc_ref = (
        db.collection("NYSE")
        .document("AMZN")
        .update({"indices": firestore.ArrayUnion(["SPY"])})
    )


def filter_indices():
    doc_ref = db.collection("NYSE").document("AMZN")
    doc = doc_ref.get()
    doc = doc.to_dict()
    doc_ref.update({"indices": [x for x in doc["indices"] if x.startswith("S")]})


create()
