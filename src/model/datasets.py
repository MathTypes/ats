import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torchfitter.utils.preprocessing import tabular_to_sliding_dataset

np.random.seed(0)

import pyarrow.parquet as pq
import logging

def generate_stock_returns():
    data = pd.read_parquet("data/FUT/30min/ES", engine='fastparquet')
    #data = pd.read_pickle("data/XAUUSD.pkl")
    #data = pq.read_table("data/ES_30min.pkl")
    #data = data.to_pandas()
    logging.info(f"data:{data.head()}")
    logging.info(f"data:{data.info()}")
    #exit(0)
    #data = data.drop(columns=["Volume"])
    data = data.pct_change().dropna()
    data["Time"] = data.index
    data["Time"] = data["Time"].apply(lambda x:x.timestamp()).astype(np.float32)
    val_idx = int(len(data) * 0.7)
    tst_idx = int(len(data) * 0.8)

    logging.info(f"data:{data.values.shape}")
    logging.info(f"data:{data.values[:30]}")
    X_train, y_train, X_val, y_val, X_test, y_test = tabular_to_sliding_dataset(
        data[["Time", "Open", "High", "Low", "Close", "Volume"]].values,
        validation_idx=val_idx,
        test_idx=tst_idx,
        n_past=20,
        n_future=5
    )
    #X_train = X_train[:, 1:, :]
    #y_train = y_train[:, 1:, :]
    #X_val = X_val[:, 1:, :]
    #y_val = y_val[:, 1:, :]
    #X_test = X_test[:, 1:, :]
    #y_test = y_test[:, 1:, :]
    #y_train = y_train[:,3,:]
    #y_val = y_val[:,3,:]
    #y_test = y_test[:,3,:]
    logging.info(f"data:{X_train[:30]}")
    logging.info(f"data:{y_train[:30]}")
    logging.info(f"data:{X_val[:30]}")
    logging.info(f"data:{y_val[:30]}")
    logging.info(f"data:{X_test[:30]}")
    logging.info(f"data:{y_test[:30]}")
    logging.info(f"X_train:{X_train.shape}")
    logging.info(f"y_train:{y_train.shape}")
    logging.info(f"X_val:{X_val.shape}")
    logging.info(f"y_val:{y_val.shape}")
    logging.info(f"X_test:{X_test.shape}")
    logging.info(f"y_test:{y_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test