import pandas as pd
ref = db.collection(u'user')
docs = ref.stream()
items = list(map(lambda x: {**x.to_dict(), 'id': x.id}, docs))
df = pd.DataFrame(items) # , columns=['id', 'email']
df.set_index('id', inplace=True)