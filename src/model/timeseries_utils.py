from typing import List, Optional, Tuple, Union
import os
import numpy as np
from torch import nn, Tensor
from typing import Optional, Any, Union, Callable, Tuple
import torch
import pandas as pd
from pathlib import Path


def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Modified from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:

        dim1: int, for both src and tgt masking, this must be target sequence
              length

        dim2: int, for src masking this must be encoder sequence length (i.e.
              the length of the input sequence to the model),
              and for tgt masking, this must be target sequence length


    Return:

        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * np.float32("-inf"), diagonal=1)


def get_indices_input_target(
    num_obs, input_len, step_size, forecast_horizon, target_len
):
    """
    Produce all the start and end index positions of all sub-sequences.
    The indices will be used to split the data into sub-sequences on which
    the models will be trained.

    Returns a tuple with four elements:
    1) The index position of the first element to be included in the input sequence
    2) The index position of the last element to be included in the input sequence
    3) The index position of the first element to be included in the target sequence
    4) The index position of the last element to be included in the target sequence


    Args:
        num_obs (int): Number of observations in the entire dataset for which
                        indices must be generated.

        input_len (int): Length of the input sequence (a sub-sequence of
                         of the entire data sequence)

        step_size (int): Size of each step as the data sequence is traversed.
                         If 1, the first sub-sequence will be indices 0-input_len,
                         and the next will be 1-input_len.

        forecast_horizon (int): How many index positions is the target away from
                                the last index position of the input sequence?
                                If forecast_horizon=1, and the input sequence
                                is data[0:10], the target will be data[11:taget_len].

        target_len (int): Length of the target / output sequence.
    """

    input_len = round(input_len)  # just a precaution
    start_position = 0
    stop_position = num_obs - 1  # because of 0 indexing

    subseq_first_idx = start_position
    subseq_last_idx = start_position + input_len
    target_first_idx = subseq_last_idx + forecast_horizon
    target_last_idx = target_first_idx + target_len
    print("target_last_idx is {}".format(target_last_idx))
    print("stop_position is {}".format(stop_position))
    indices = []
    while target_last_idx <= stop_position:
        indices.append(
            (subseq_first_idx, subseq_last_idx, target_first_idx, target_last_idx)
        )
        subseq_first_idx += step_size
        subseq_last_idx += step_size
        target_first_idx = subseq_last_idx + forecast_horizon
        target_last_idx = target_first_idx + target_len

    return indices


def get_indices_entire_sequence(
    data: pd.DataFrame, window_size: int, step_size: int
) -> list:
    """
    Produce all the start and end index positions that is needed to produce
    the sub-sequences.

    Returns a list of tuples. Each tuple is (start_idx, end_idx) of a sub-
    sequence. These tuples should be used to slice the dataset into sub-
    sequences. These sub-sequences should then be passed into a function
    that slices them into input and target sequences.

    Args:
        num_obs (int): Number of observations (time steps) in the entire
                       dataset for which indices must be generated, e.g.
                       len(data)
        window_size (int): The desired length of each sub-sequence. Should be
                           (input_sequence_length + target_sequence_length)
                           E.g. if you want the model to consider the past 100
                           time steps in order to predict the future 50
                           time steps, window_size = 100+50 = 150
        step_size (int): Size of each step as the data sequence is traversed
                         by the moving window.
                         If 1, the first sub-sequence will be [0:window_size],
                         and the next will be [1:window_size].
    Return:
        indices: a list of tuples
    """

    stop_position = len(data) - 1  # 1- because of 0 indexing

    # Start the first sub-sequence at index position 0
    subseq_first_idx = 0
    subseq_last_idx = window_size
    indices = []
    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_last_idx))
        subseq_first_idx += step_size
        subseq_last_idx += step_size

    return indices


def read_data(
    data_dir: Union[str, Path] = "data", timestamp_col_name: str = "timestamp"
) -> pd.DataFrame:
    """
    Read data from csv file and return pd.Dataframe object

    Args:
        data_dir: str or Path object specifying the path to the directory
                  containing the data
        target_col_name: str, the name of the column containing the target variable
        timestamp_col_name: str, the name of the column or named index
                            containing the timestamps
    """

    # Ensure that `data_dir` is a Path object
    data_dir = Path(data_dir)

    # Read csv file
    csv_files = list(data_dir.glob("*.csv"))

    if len(csv_files) > 1:
        raise ValueError("data_dir contains more than 1 csv file. Must only contain 1")
    elif len(csv_files) == 0:
        raise ValueError("data_dir must contain at least 1 csv file.")
    data_path = csv_files[0]

    print("Reading file in {}".format(data_path))

    data = pd.read_csv(
        data_path,
        parse_dates=[timestamp_col_name],
        index_col=[timestamp_col_name],
        infer_datetime_format=True,
        low_memory=False,
    )

    # Make sure all "n/e" values have been removed from df.
    if is_ne_in_df(data):
        raise ValueError("data frame contains 'n/e' values. These must be handled")

    data = to_numeric_and_downcast_data(data)

    # Make sure data is in ascending order by timestamp
    data.sort_values(by=[timestamp_col_name], inplace=True)

    return data


def is_ne_in_df(df: pd.DataFrame):
    """
    Some raw data files contain cells with "n/e". This function checks whether
    any column in a df contains a cell with "n/e". Returns False if no columns
    contain "n/e", True otherwise
    """

    for col in df.columns:
        true_bool = df[col] == "n/e"
        if any(true_bool):
            return True

    return False


def to_numeric_and_downcast_data(df: pd.DataFrame):
    """
    Downcast columns in df to smallest possible version of it's existing data
    type
    """
    fcols = df.select_dtypes("float").columns
    icols = df.select_dtypes("integer").columns
    df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")
    df[icols] = df[icols].apply(pd.to_numeric, downcast="int")

    return df


class TimeSeriesFeatureEmbedder(nn.Module):
    """
    Embed a sequence of categorical features.

    Args:
        cardinalities (`list[int]`):
            List of cardinalities of the categorical features.
        embedding_dims (`list[int]`):
            List of embedding dimensions of the categorical features.
    """

    def __init__(self, cardinalities: List[int], embedding_dims: List[int]) -> None:
        super().__init__()

        self.num_features = len(cardinalities)
        self.embedders = nn.ModuleList(
            [nn.Embedding(c, d) for c, d in zip(cardinalities, embedding_dims)]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.num_features > 1:
            # we slice the last dimension, giving an array of length
            # self.num_features with shape (N,T) or (N)
            cat_feature_slices = torch.chunk(features, self.num_features, dim=-1)
        else:
            cat_feature_slices = [features]

        return torch.cat(
            [
                embed(cat_feature_slice.squeeze(-1))
                for embed, cat_feature_slice in zip(self.embedders, cat_feature_slices)
            ],
            dim=-1,
        )


class TimeSeriesStdScaler(nn.Module):
    """
    Standardize features by calculating the mean and scaling along some given dimension `dim`, and then normalizes it
    by subtracting from the mean and dividing by the standard deviation.

    Args:
        dim (`int`):
            Dimension along which to calculate the mean and standard deviation.
        keepdim (`bool`, *optional*, defaults to `False`):
            Controls whether to retain dimension `dim` (of length 1) in the scale tensor, or suppress it.
        minimum_scale (`float`, *optional*, defaults to 1e-5):
            Default scale that is used for elements that are constantly zero along dimension `dim`.
    """

    def __init__(self, dim: int, keepdim: bool = False, minimum_scale: float = 1e-5):
        super().__init__()
        if not dim > 0:
            raise ValueError(
                "Cannot compute scale along dim = 0 (batch dimension), please provide dim > 0"
            )
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale

    @torch.no_grad()
    def forward(
        self, data: torch.Tensor, weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        denominator = weights.sum(self.dim, keepdim=self.keepdim)
        denominator = denominator.clamp_min(1.0)
        loc = (data * weights).sum(self.dim, keepdim=self.keepdim) / denominator

        variance = (((data - loc) * weights) ** 2).sum(
            self.dim, keepdim=self.keepdim
        ) / denominator
        scale = torch.sqrt(variance + self.minimum_scale)
        return (data - loc) / scale, loc, scale


class TimeSeriesMeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along dimension `dim`, and scales the data
    accordingly.

    Args:
        dim (`int`):
            Dimension along which to compute the scale.
        keepdim (`bool`, *optional*, defaults to `False`):
            Controls whether to retain dimension `dim` (of length 1) in the scale tensor, or suppress it.
        default_scale (`float`, *optional*, defaults to `None`):
            Default scale that is used for elements that are constantly zero. If `None`, we use the scale of the batch.
        minimum_scale (`float`, *optional*, defaults to 1e-10):
            Default minimum possible scale that is used for any item.
    """

    def __init__(
        self,
        dim: int = -1,
        keepdim: bool = True,
        default_scale: Optional[float] = None,
        minimum_scale: float = 1e-10,
    ):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale
        self.default_scale = default_scale

    @torch.no_grad()
    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # shape: (N, [C], T=1)
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        num_observed = observed_indicator.sum(self.dim, keepdim=True)

        scale = ts_sum / torch.clamp(num_observed, min=1)

        # If `default_scale` is provided, we use it, otherwise we use the scale
        # of the batch.
        if self.default_scale is None:
            batch_sum = ts_sum.sum(dim=0)
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            default_scale = self.default_scale * torch.ones_like(scale)

        # apply default scale where there are no observations
        scale = torch.where(num_observed > 0, scale, default_scale)

        # ensure the scale is at least `self.minimum_scale`
        scale = torch.clamp(scale, min=self.minimum_scale)
        scaled_data = data / scale

        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)

        return scaled_data, torch.zeros_like(scale), scale


class TimeSeriesNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along dimension `dim`, and therefore applies no scaling to the input data.

    Args:
        dim (`int`):
            Dimension along which to compute the scale.
        keepdim (`bool`, *optional*, defaults to `False`):
            Controls whether to retain dimension `dim` (of length 1) in the scale tensor, or suppress it.
    """

    def __init__(self, dim: int, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = torch.ones_like(data, requires_grad=False).mean(
            dim=self.dim, keepdim=self.keepdim
        )
        loc = torch.zeros_like(data, requires_grad=False).mean(
            dim=self.dim, keepdim=self.keepdim
        )
        return data, loc, scale


def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log likelihood loss from input distribution with respect to target.
    """
    return -input.log_prob(target)


def weighted_average(
    input_tensor: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None
) -> torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given `dim`, masking values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    Args:
        input_tensor (`torch.FloatTensor`):
            Input tensor, of which the average must be computed.
        weights (`torch.FloatTensor`, *optional*):
            Weights tensor, of the same shape as `input_tensor`.
        dim (`int`, *optional*):
            The dim along which to average `input_tensor`.

    Returns:
        `torch.FloatTensor`: The tensor with values averaged along the specified `dim`.
    """
    if weights is not None:
        weighted_tensor = torch.where(
            weights != 0, input_tensor * weights, torch.zeros_like(input_tensor)
        )
        sum_weights = torch.clamp(
            weights.sum(dim=dim) if dim else weights.sum(), min=1.0
        )
        return (
            weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()
        ) / sum_weights
    else:
        return input_tensor.mean(dim=dim)
