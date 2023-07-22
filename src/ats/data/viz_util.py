import chart_studio
import emot
import cufflinks
import nltk.tokenize

import plotly.io as pio
import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer

username = ""
api_key = ""

chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

pd.set_option("display.max_colwidth", None)

nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

sid = SentimentIntensityAnalyzer()
emot_obj = emot.core.emot()

cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme="pearl")

pio.renderers.default = "colab"
