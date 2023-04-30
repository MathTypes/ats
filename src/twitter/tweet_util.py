import os
import logging

import pandas as pd

def get_last_tweet_id(symbol_output_dir):
    df_vec = []
    for filename in os.listdir(symbol_output_dir):
        id_file = os.path.join(symbol_output_dir, filename)
        logging.info(f'get_last_tweet_id:{id_file}')
        if not id_file.endswith(".csv"):
            continue
        if not os.path.isfile(id_file):
            continue
        df = pd.read_csv(id_file)
        df_vec.append(df)
    if df_vec:
        df = pd.concat(df_vec)
        if "tweet_id" in df.columns:
            return df["tweet_id"].max()
        elif "Id" in df.columns:
            return df["Id"].max()
    return None
