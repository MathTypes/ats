from typing import Dict, List, Tuple, Union, Optional, Callable
from dataclasses import dataclass


@dataclass
class PredictionInput:
    x: Dict
    idx: int
    prediction_date_time: str
    x_time: List
    train_data_rows: List
    train_data_row: Dict
    decoder_time_idx: int


@dataclass
class PredictionOutput:
    idx: int
    out: Dict
    y_hats: List
    quantile: List
    embedding: List
    interp_output: Dict
