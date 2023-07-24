from typing import Dict, List, Tuple, Union, Optional, Callable
from dataclasses import dataclass


@dataclass
class PredictionInput:
    x: Optional[Dict] = None
    idx: Optional[int] = None
    prediction_date_time: Optional[str] = None
    x_time: Optional[List] = None
    train_data_rows: Optional[List] = None
    train_data_row: Optional[Dict] = None
    decoder_time_idx: Optional[int] = None


@dataclass
class PredictionOutput:
    idx: Optional[int] = None
    out: Optional[Dict] = None
    y_hats: Optional[List] = None
    y_quantiles: Optional[List] = None
    embedding: Optional[List] = None
    interp_output: Optional[Dict] = None
