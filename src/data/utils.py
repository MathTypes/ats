import glob
import logging
import os
import subprocess
import pandas as pd


def load_unlabeled_dataset(filenames):
    """Load spam training dataset without any labels."""
    dfs = []
    for i, filename in enumerate(filenames, start=1):
        df = pd.read_csv(filename, dtype="unicode")
        # Lowercase column names
        df.columns = map(str.lower, df.columns)
        # Rename fields
        df = df.rename(columns={"Text": "text"})
        # Remove comment_id, label fields
        df = df.drop("tweet id", axis=1)
        df = df.drop("unnamed: 0", axis=1)
        df = df.drop("unnamed: 0.1", axis=1)
        df = df.drop("index", axis=1)
        # df = df.drop("label", axis=1)
        # Shuffle order
        # df = df[["text"]]
        df = df.sample(frac=1, random_state=123).reset_index(drop=True)
        dfs.append(df)
    return pd.concat(dfs)