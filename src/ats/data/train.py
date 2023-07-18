from snorkel.augmentation import transformation_function
from nltk.corpus import wordnet as wn
import nltk
import random
import argparse
import glob
import logging
import os
import re

from utils import load_unlabeled_dataset
from textblob import TextBlob

from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import labeling_function
from snorkel.slicing import slicing_function
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from snorkel.augmentation import ApplyOnePolicy, PandasTFApplier

DISPLAY_ALL_TEXT = True
pd.set_option("display.max_colwidth", 0 if DISPLAY_ALL_TEXT else 50)

parser = argparse.ArgumentParser(description="Train label generation")
parser.add_argument(
    "-v", "--verbose", help="increase output verbosity", action="store_true"
)
parser.add_argument("--input_path", type=str, required=True)

args = parser.parse_args()
if args.verbose:
    logging.basicConfig(level=logging.DEBUG)

df_train = load_unlabeled_dataset(glob.glob(args.input_path))

ABSTAIN = -1
DOWN = 0
UP = 1


@labeling_function()
def lf_keyword_up(x):
    return UP if "up" in x.text.lower() else ABSTAIN


@labeling_function()
def lf_keyword_strong(x):
    return UP if "strong" in x.text.lower() else ABSTAIN


@labeling_function()
def lf_keyword_weak(x):
    return DOWN if "weak" in x.text.lower() else ABSTAIN


@labeling_function()
def lf_keyword_crash(x):
    return DOWN if "crash" in x.text.lower() else ABSTAIN


@labeling_function()
def lf_keyword_long(x):
    return UP if "long" in x.text.lower() else ABSTAIN


@labeling_function()
def lf_keyword_short(x):
    return DOWN if "short" in x.text.lower() else ABSTAIN


@labeling_function()
def lf_keyword_down(x):
    return DOWN if "down" in x.text.lower() else ABSTAIN


# Define the set of labeling functions (LFs)
lfs = [
    lf_keyword_up,
    lf_keyword_strong,
    lf_keyword_weak,
    lf_keyword_down,
    lf_keyword_crash,
    lf_keyword_long,
    lf_keyword_short,
]

# Apply the LFs to the unlabeled training data
applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)

# Train the label model and compute the training labels
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, n_epochs=500, log_freq=50, seed=123)
df_train["label"] = label_model.predict(L=L_train, tie_break_policy="abstain")


# %%
df_train = df_train[df_train.label != ABSTAIN]
print(f"df_train:{df_train}")


nltk.download("wordnet", quiet=True)


def get_synonyms(word):
    """Get the synonyms of word from Wordnet."""
    lemmas = set().union(*[s.lemmas() for s in wn.synsets(word)])
    return list(set(l.name().lower().replace("_", " ") for l in lemmas) - {word})


@transformation_function()
def tf_replace_word_with_synonym(x):
    """Try to replace a random word with a synonym."""
    words = x.text.lower().split()
    idx = random.choice(range(len(words)))
    synonyms = get_synonyms(words[idx])
    if len(synonyms) > 0:
        x.text = " ".join(words[:idx] + [synonyms[0]] + words[idx + 1 :])
        return x


# %% [markdown]
# Next, we apply this transformation function to our training dataset:


tf_policy = ApplyOnePolicy(n_per_original=2, keep_original=True)
tf_applier = PandasTFApplier([tf_replace_word_with_synonym], tf_policy)
df_train_augmented = tf_applier.apply(df_train)


@slicing_function()
def short_link(x):
    """Return whether text matches common pattern for shortened ".ly" links."""
    return int(bool(re.search(r"\w+\.ly", x.text)))


train_text = df_train_augmented.text.tolist()
X_train = CountVectorizer(ngram_range=(1, 2)).fit_transform(train_text)

clf = LogisticRegression(solver="lbfgs")
clf.fit(X=X_train, y=df_train_augmented.label.values)