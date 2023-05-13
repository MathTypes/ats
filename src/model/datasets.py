from numpy.lib import recfunctions as rfn
import numpy as np
import pandas as pd
from typing import Iterable, List, Union
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

np.random.seed(0)

import pyarrow.parquet as pq
import logging

def tabular_to_sliding_dataset(
    dataset: np.ndarray,
    validation_idx: int,
    test_idx: int,
    n_past: int,
    n_future: int,
    make_writable: bool = True,
    scaler: Union[TransformerMixin, BaseEstimator] = None,
) -> List[np.ndarray]:
    """Convert a tabular or 3D dataset to a sliding window dataset (4D).

    This function expects a datatype that supports the array protocol.
    E.g.: Pandas DataFrame or NumPy arrays.

    Parameters
    ----------
    dataset : array-like
        Array-like object.
    validation_idx : int
        Index to create the validation set.
    test_idx : int
        Index to create the testing set.
    n_past : int
        Number of past steps to make predictions. It will be used to
        generate the features.
    n_future : int
        Number of future steps to predict. It will be used to generate
        the labels.
    make_writable : bool, optional, default: True
        Make the resulting arrays writable by creating a copy of the view.
    scaler : sklearn.base.TransformerMixin, optional, default: None
        If not None, the data will be normalized with the passed scaler.
        Assumes distribution does not vary over time.

    Returns
    -------
    output : list of numpy.ndarray
        A list containing the resulting arrays. They appear in this order:
            * X_train: Features train set.
            * y_train: Labels train set.
            * X_val: Features validation set.
            * y_val: Labels validation set.
            * X_test: Features test set.
            * y_test : Labels test set.

    Warning
    -------
    This function is very memory-consuming.

    See Also
    --------
    torchfitter.utils.preprocessing.train_test_val_split

    TODO
    ----
    * Allow spliting by percentage.
    * Allow single-feature forecasting instead of multi-forecasting.
    * Use `train_test_val_split` to abstract the splitting.
    * Allow selecting the target column.
    """

    def get_features_and_labels(array, n_past, n_future):
        """
        Convenient sub-function that wraps the functionality to create a
        rolling view and select the past as features and the future as labels.
        """
        window_length = n_past + n_future
        roll_view = np.lib.stride_tricks.sliding_window_view(
            array, window_length, axis=0
        )
        X = roll_view[:, :, :n_past]
        y = roll_view[:, :, n_past:]
        return X, y

    # type-agnostic
    arr = dataset.__array__()
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    # split
    train = arr[:validation_idx]
    validation = arr[validation_idx:test_idx]
    test = arr[test_idx:]

    if scaler is not None:
        scaler.fit(train)

        train = scaler.transform(train)
        validation = scaler.transform(validation)
        test = scaler.transform(test)

    # get a rolling view of each data chunk
    output = []
    for chunk in [train, validation, test]:
        X, y = get_features_and_labels(
            array=chunk, n_past=n_past, n_future=n_future
        )

        # make a copy to generate a writable array
        if make_writable:
            _tup = (X.copy(), y.copy())
        else:
            _tup = (X, y)

        output.append(_tup)

    # unpack and return
    output = [item for sublist in output for item in sublist]
    return output

def generate_stock_returns(n_past, n_futures):
    data = pd.read_parquet("data/token/FUT/30min/ES", engine='fastparquet')
    logging.info(f"data:{data.head()}")
    logging.info(f"data:{data.info()}")
    data["Time"] = data.index
    data["Time"] = data["Time"].apply(lambda x:x.timestamp()).astype(np.float32)
    val_idx = max(int(len(data) * 0.7), len(data) - 2048*16)
    tst_idx = max(int(len(data) * 0.8), len(data) - 2048)

    X_train, y_train, X_val, y_val, X_test, y_test = tabular_to_sliding_dataset(
        data[["Time", "Open", "High", "Low", "Close", "Volume"]].values,
        validation_idx=val_idx,
        test_idx=tst_idx,
        n_past=n_past,
        n_future=n_futures
    )
    logging.info(f"X_train:{X_train.shape}")
    logging.info(f"X_train:{X_train[:1]}")
    logging.info(f"y_train:{y_train.shape}")
    logging.info(f"y_train:{y_train[:1]}")
    X_train = X_train[:, 1:, :]
    y_train = y_train[:, 4, :]
    X_val = X_val[:, 1:, :]
    y_val = y_val[:, 4, :]
    X_test = X_test[:, 1:, :]
    y_test = y_test[:, 4, :]
    logging.info(f"X_train:{X_train.shape}")
    logging.info(f"y_train:{y_train.shape}")
    logging.info(f"X_val:{X_val.shape}")
    logging.info(f"y_val:{y_val.shape}")
    logging.info(f"X_test:{X_test.shape}")
    logging.info(f"y_test:{y_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def generate_stock_tokens():
    data = pd.read_parquet("data/token/FUT/30min/ES", engine='fastparquet')
    logging.info(f"data.OpenPct:{data.OpenPct.shape}, type:{type(data.OpenPct)}, dtype:{data.OpenPct.dtype}")
    logging.info(f"data.OpenPct.:{data.OpenPct[0:10]}")
    data["Time"] = data.index
    data["Time"] = data["Time"].apply(lambda x:float(x.timestamp()))
    data = data.dropna()
    open = data.OpenPct.to_numpy()
    high = data.HighPct.to_numpy()
    low = data.LowPct.to_numpy()
    close = data.ClosePct.to_numpy()
    volume = data.VolumePct.to_numpy()
    time = data.Time.to_numpy().astype(np.float32)
    logging.info(f"before open:{open.shape}, type:{type(open)}, dtype:{open.dtype}")
    logging.info(f"before volume:{volume.shape}, type:{type(volume)}, dtype:{volume.dtype}")
    values = np.stack((open, high, low, close, volume, time), axis=-1)
    logging.info(f"before values1:{values.shape}, type:{type(values)}, dtype:{values.dtype}")
    val_idx = max(int(len(data) * 0.7), len(data) - 2048*16)
    tst_idx = max(int(len(data) * 0.8), len(data) - 2048)

    logging.info(f"values1:{values.shape}, type:{type(values)}, dtype:{values.dtype}")
    logging.info(f"time_shape:{data.Time.shape}")
    logging.info(f"values:{values.shape}")
    logging.info(f"data:{values[0:5,...]}")
    X_train = values[:val_idx]
    X_val = values[val_idx:tst_idx]
    X_test = values[tst_idx:]
    logging.info(f"X_train:{X_train.shape}")
    logging.info(f"X_val:{X_val.shape}")
    logging.info(f"X_train:{X_train.dtype}, type:{type(X_train)}, {X_train.shape}")
    logging.info(f"X_train:{X_train.dtype}, type:{type(X_train)}")
    logging.info(f"X_train:{X_train[:30]}")
    logging.info(f"X_val:{X_val[:30]}")
    logging.info(f"X_test:{X_test[:30]}")
    
    return X_train, X_val, X_test
