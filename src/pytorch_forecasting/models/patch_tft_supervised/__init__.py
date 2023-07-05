"""
The temporal fusion transformer is a powerful predictive model for forecasting timeseries
"""
from copy import copy
import logging
from typing import Dict, List, Tuple, Union, Optional, Callable

from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import torch
from torch import nn
from torchmetrics import Metric as LightningMetric

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, MultiLoss, QuantileLoss
from pytorch_forecasting.models.base_model import BaseModelWithCovariates
from pytorch_forecasting.models.nn import LSTM, MultiEmbedding
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import (
    AddNorm,
    GateAddNorm,
    GatedLinearUnit,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    VariableSelectionNetwork,
)
from pytorch_forecasting.utils import create_mask, detach, integer_histogram, masked_op, padded_stack, to_list
#from collections import OrderedDict
from .layers.PatchTST_layers import *
from .layers.RevIN import RevIN


class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0, flatten=False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model*num_patch

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)


    def forward(self, x):                     
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * num_patch]
                z = self.linears[i](z)                    # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)         # x: [bs x nvars x forecast_len]
        else:
            x = self.flatten(x)     # x: [bs x nvars x (d_model * num_patch)]    
            x = self.dropout(x)
            x = self.linear(x)      # x: [bs x nvars x forecast_len]
        return x.transpose(2,1)     # [bs x forecast_len x nvars]


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """

        x = x.transpose(2,3)                     # [bs x nvars x num_patch x d_model]
        x = self.linear( self.dropout(x) )      # [bs x nvars x num_patch x patch_len]
        x = x.permute(0,2,1,3)                  # [bs x num_patch x nvars x patch_len]
        return x


class Patch(nn.Module):
    def __init__(self, seq_len, stride, c_in, num_patch, patch_len, 
                 n_layers=3, d_model=128, n_heads=16, shared_embedding=True,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()
        self.seq_len = seq_len
        self.stride = stride
        self.num_patch = num_patch
        tgt_len = patch_len  + stride*(self.num_patch-1)
        self.s_begin = seq_len - tgt_len
        self.n_vars = c_in
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding        
        #logging.info(f"seq_len:{seq_len}, stride:{stride}, num_patch:{num_patch}")
        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not shared_embedding: 
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars): self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            self.W_P = nn.Linear(patch_len, d_model)      

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x) -> torch.Tensor:          
        """
        x: [bs x seq_len x n_vars]
        """
        x = x[:, self.s_begin:, :]
        #logging.info(f"x_before_unfold:{x.shape}")
        x = x.permute(0, 2, 1)
        # x: [bs x n_vars x seq_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        #logging.info(f"x_unfold:{x.shape}")
        # x: [bs x n_vars x patch_num x patch_len]
        bs, n_vars, patch_num, patch_len = x.shape
        #logging.info(f"x_permute:{x.shape}, bs:{bs}, n_vars:{n_vars}, patch_num:{patch_num}, patch_len:{patch_len}")
        if x.size(1)==0:
            return torch.reshape(x, (bs*n_vars, patch_num, self.d_model) )
        # Input encoding
        #logging.info(f"x:{x.shape}")
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars): 
                z = self.W_P[i](x[:,i,:,:])
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.W_P(x)
        # x: [bs x n_vars x patch_num x d_model]
        #u = torch.reshape(x, (bs*n_vars, patch_num, self.d_model) )
        #logging.info(f"u.shape:{u.shape}")
        # u: [bs x num_patch x n_vars x d_model]
        #u = self.dropout(u + self.W_pos)
        return x

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, num_patch, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.num_patch = num_patch
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(num_patch, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(num_patch, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # x: [bs x nvars x d_model]
        #logging.info(f"flatten_head: x.shape:{x.shape}")
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:])
                # z: [bs x d_model * patch_num]
                z = self.linears[i](z)
                # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
                x = torch.stack(x_out, dim=1)
                # x: [bs x nvars x target_window]
        else:
            # [x : bs x num_patch x d_model]
            x = x.permute(0, 2, 1)
            # [x: bs x d_model x num_patch]
            #x = self.flatten(x)
            #logging.info(f"flatten_head after flatten: x.shape:{x.shape}")
            x = self.linear(x)
            # [x: bs x d_model x target_window
            x = x.permute(0, 2, 1)
            # [x: bs x target_window x d_model]
            #logging.info(f"flatten_head after lineage: x.shape:{x.shape}")
            x = self.dropout(x)
        return x

class PatchTftSupervised(BaseModelWithCovariates):
    def __init__(
            self,
            c_in:int, target_dim:int, patch_len:int, stride:int, num_patch:int,
            prediction_num_patch:int,
            n_layers:int=3, d_model=128, n_heads=16, shared_embedding=True, d_ff:int=256,
            norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu",
            res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
            pe:str='zeros', learn_pe:bool=True, head_dropout = 0,
            head_type = "prediction", individual = False,
            y_range:Optional[tuple]=None, verbose:bool=False,
            lstm_layers: int = 1,
            output_size: Union[int, List[int], Dict[str, List[int]]] = 7,
            loss: MultiHorizonMetric = None,
            attention_head_size: int = 4,
            max_encoder_length: int = 10,
            static_categoricals: List[str] = [],
            static_reals: List[str] = [],
            time_varying_categoricals_encoder: List[str] = [],
            time_varying_categoricals_decoder: List[str] = [],
            categorical_groups: Dict[str, List[str]] = {},
            time_varying_reals_encoder: List[str] = [],
            time_varying_reals_decoder: List[str] = [],
            x_reals: List[str] = [],
            x_categoricals: List[str] = [],
            hidden_size: int = 8,
            hidden_continuous_size: int = 8,
            hidden_continuous_sizes: Dict[str, int] = {},
            embedding_sizes: Dict[str, Tuple[int, int]] = {},
            embedding_paddings: List[str] = [],
            embedding_labels: Dict[str, np.ndarray] = {},
            learning_rate: float = 1e-3,
            log_interval: Union[int, float] = -1,
            log_val_interval: Union[int, float] = None,
            log_gradient_flow: bool = False,
            reduce_on_plateau_patience: int = 1000,
            monotone_constaints: Dict[str, int] = {},
            share_single_variable_networks: bool = False,
            causal_attention: bool = True,
            logging_metrics: nn.ModuleList = None,
            loss_per_head: Dict = None,
            **kwargs,
    ):
        """
        PatchTST for forecasting timeseries - use its :py:meth:`~from_dataset` method if possible.

        Implementation of the article
        `Temporal Fusion Transformers for Interpretable Multi-horizon Time Series
        Forecasting <https://arxiv.org/pdf/1912.09363.pdf>`_. The network outperforms DeepAR by Amazon by 36-69%
        in benchmarks.

        Enhancements compared to the original implementation (apart from capabilities added through base model
        such as monotone constraints):

        * static variables can be continuous
        * multiple categorical variables can be summarized with an EmbeddingBag
        * variable encoder and decoder length by sample
        * categorical embeddings are not transformed by variable selection network (because it is a redundant operation)
        * variable dimension in variable selection network are scaled up via linear interpolation to reduce
          number of parameters
        * non-linear variable processing in variable selection network can be shared among decoder and encoder
          (not shared by default)

        Tune its hyperparameters with
        :py:func:`~pytorch_forecasting.models.temporal_fusion_transformer.tuning.optimize_hyperparameters`.

        Args:

            output_size: number of outputs (e.g. number of quantiles for QuantileLoss and one target or list
                of output sizes or dictionry from head name to list of output sizes).
            loss: loss function taking prediction and targets
            attention_head_size: number of attention heads (4 is a good default)
            max_encoder_length: length to encode (can be far longer than the decoder length but does not have to be)
            static_categoricals: names of static categorical variables
            static_reals: names of static continuous variables
            time_varying_categoricals_encoder: names of categorical variables for encoder
            time_varying_categoricals_decoder: names of categorical variables for decoder
            time_varying_reals_encoder: names of continuous variables for encoder
            time_varying_reals_decoder: names of continuous variables for decoder
            categorical_groups: dictionary where values
                are list of categorical variables that are forming together a new categorical
                variable which is the key in the dictionary
            x_reals: order of continuous variables in tensor passed to forward function
            x_categoricals: order of categorical variables in tensor passed to forward function
            hidden_continuous_size: default for hidden size for processing continous variables (similar to categorical
                embedding size)
            hidden_continuous_sizes: dictionary mapping continuous input indices to sizes for variable selection
                (fallback to hidden_continuous_size if index is not in dictionary)
            embedding_sizes: dictionary mapping (string) indices to tuple of number of categorical classes and
                embedding size
            embedding_paddings: list of indices for embeddings which transform the zero's embedding to a zero vector
            embedding_labels: dictionary mapping (string) indices to list of categorical labels
            learning_rate: learning rate
            log_interval: log predictions every x batches, do not log if 0 or less, log interpretation if > 0. If < 1.0
                , will log multiple entries per batch. Defaults to -1.
            log_val_interval: frequency with which to log validation set metrics, defaults to log_interval
            log_gradient_flow: if to log gradient flow, this takes time and should be only done to diagnose training
                failures
            reduce_on_plateau_patience (int): patience after which learning rate is reduced by a factor of 10
            monotone_constaints (Dict[str, int]): dictionary of monotonicity constraints for continuous decoder
                variables mapping
                position (e.g. ``"0"`` for first position) to constraint (``-1`` for negative and ``+1`` for positive,
                larger numbers add more weight to the constraint vs. the loss but are usually not necessary).
                This constraint significantly slows down training. Defaults to {}.
            share_single_variable_networks (bool): if to share the single variable networks between the encoder and
                decoder. Defaults to False.
            causal_attention (bool): If to attend only at previous timesteps in the decoder or also include future
                predictions. Defaults to True.
            logging_metrics (nn.ModuleList[LightningMetric]): list of metrics that are logged during training.
                Defaults to nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()]).
            **kwargs: additional arguments to :py:class:`~BaseModel`.
        """
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()])
        if loss is None:
            loss = QuantileLoss()
        self.loss_per_head = loss_per_head
        self.save_hyperparameters()
        #logging.info(f"hparams:{self.hparams}")
        # store loss function separately as it is a module
        assert isinstance(loss, LightningMetric), "Loss has to be a PyTorch Lightning `Metric`"
        #logging.info(f"kwargs:{kwargs}")
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)
        #logging.info(f"after hparams:{self.hparams}")
        self.d_model = d_model
        self.skipped_patch = int(patch_len/stride - 1)
        # processing inputs
        # embeddings
        self.input_embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
            max_embedding_size=self.hparams.hidden_size,
        )

        # continuous variable processing
        self.prescalers = nn.ModuleDict(
            {
                name: nn.Linear(1, self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size))
                for name in self.hparams.x_reals
            }
        )

        # variable selection
        # variable selection for static variables
        static_input_sizes = {
            name: self.input_embeddings.output_size[name] for name in self.hparams.static_categoricals
        }
        static_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                for name in self.hparams.static_reals
            }
        )
        self.static_variable_selection = VariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={name: True for name in self.hparams.static_categoricals},
            dropout=self.hparams.dropout,
            prescalers=self.prescalers,
        )
        self.prediction_num_patch = prediction_num_patch
        # variable selection for encoder and decoder
        variable_hidden_size = d_model 
        encoder_input_sizes = {
            name: self.input_embeddings.output_size[name] for name in self.hparams.time_varying_categoricals_encoder
        }
        encoder_input_sizes.update(
            {
                #name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                name: self.hparams.hidden_continuous_sizes.get(name, d_model)
                for name in self.hparams.time_varying_reals_encoder
            }
        )
        #logging.info(f"encoder_input_sizes:{encoder_input_sizes}")
        decoder_input_sizes = {
            name: self.input_embeddings.output_size[name] for name in self.hparams.time_varying_categoricals_decoder
        }
        decoder_input_sizes.update(
            {
                #name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                name: self.hparams.hidden_continuous_sizes.get(name, d_model)
                for name in self.hparams.time_varying_reals_decoder
            }
        )
        #logging.info(f"decoder_input_sizes:{decoder_input_sizes}")

        # create single variable grns that are shared across decoder and encoder
        if self.hparams.share_single_variable_networks:
            self.shared_single_variable_grns = nn.ModuleDict()
            for name, input_size in encoder_input_sizes.items():
                self.shared_single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, variable_hidden_size),
                    variable_hidden_size,
                    self.hparams.dropout,
                )
            for name, input_size in decoder_input_sizes.items():
                if name not in self.shared_single_variable_grns:
                    self.shared_single_variable_grns[name] = GatedResidualNetwork(
                        input_size,
                        min(input_size, variable_hidden_size),
                        variable_hidden_size,
                        self.hparams.dropout,
                    )
                    
        input_embedding_flags = {name: True for name in self.hparams.time_varying_reals_encoder +
                                 self.hparams.time_varying_categoricals_encoder}
        # Set to d_model instead of hparams.hidden_size since patch output converts each variable to same d_model
        #logging.info(f"input_embedding_flags:{input_embedding_flags}")
        #logging.info(f"encoder_input_sizes:{encoder_input_sizes}")
        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=d_model,
            #input_embedding_flags={name: True for name in self.hparams.time_varying_categoricals_encoder},
            input_embedding_flags=input_embedding_flags,
            dropout=self.hparams.dropout,
            context_size=d_model,
            # No scaler since all reals are already mapped to embeddings
            #prescalers=self.prescalers,
            prescalers={},
            single_variable_grns={}
            if not self.hparams.share_single_variable_networks
            else self.shared_single_variable_grns,
            reduce=True
        )
        #logging.info(f"encoder_variable_selection:{self.encoder_variable_selection}")
        #logging.info(f"decoder_input_sizes:{decoder_input_sizes}")
        self.decoder_variable_selection = VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=d_model,
            input_embedding_flags=input_embedding_flags,
            dropout=self.hparams.dropout,
            context_size=d_model,
            prescalers={},
            single_variable_grns={}
            if not self.hparams.share_single_variable_networks
            else self.shared_single_variable_grns,
            reduce = True
        )
        #logging.info(f"decoder_variable_selection:{self.decoder_variable_selection}")

        # static encoders
        # for variable selection
        self.static_context_variable_selection = GatedResidualNetwork(
            input_size=variable_hidden_size,
            hidden_size=variable_hidden_size,
            output_size=variable_hidden_size,
            dropout=self.hparams.dropout,
        )

        # for hidden state of the lstm
        self.static_context_initial_hidden_lstm = GatedResidualNetwork(
            input_size=variable_hidden_size,
            hidden_size=variable_hidden_size,
            output_size=variable_hidden_size,
            dropout=self.hparams.dropout,
        )

        # for cell state of the lstm
        self.static_context_initial_cell_lstm = GatedResidualNetwork(
            input_size=variable_hidden_size,
            hidden_size=variable_hidden_size,
            output_size=variable_hidden_size,
            dropout=self.hparams.dropout,
        )

        # for post lstm static enrichment
        self.static_context_enrichment = GatedResidualNetwork(
            variable_hidden_size, variable_hidden_size, variable_hidden_size, self.hparams.dropout
        )

        # lstm encoder (history) and decoder (future) for local processing
        #logging.info(f"lstm_input_size:{self.hparams.hidden_size}")
        self.lstm_encoder = LSTM(
            input_size=variable_hidden_size,
            #hidden_size=self.hparams.hidden_size,
            hidden_size=variable_hidden_size,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.lstm_decoder = LSTM(
            input_size=variable_hidden_size,
            hidden_size=variable_hidden_size,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
            batch_first=True,
        )

        # skip connection for lstm
        self.post_lstm_gate_encoder = GatedLinearUnit(variable_hidden_size, dropout=self.hparams.dropout)
        self.post_lstm_gate_decoder = self.post_lstm_gate_encoder
        # self.post_lstm_gate_decoder = GatedLinearUnit(self.hparams.hidden_size, dropout=self.hparams.dropout)
        self.post_lstm_add_norm_encoder = AddNorm(variable_hidden_size, trainable_add=False)
        # self.post_lstm_add_norm_decoder = AddNorm(self.hparams.hidden_size, trainable_add=True)
        self.post_lstm_add_norm_decoder = self.post_lstm_add_norm_encoder

        # static enrichment and processing past LSTM
        self.static_enrichment = GatedResidualNetwork(
            input_size=variable_hidden_size,
            hidden_size=variable_hidden_size,
            output_size=variable_hidden_size,
            dropout=self.hparams.dropout,
            context_size=variable_hidden_size,
        )
        self.stride = stride
        assert head_type in ['pretrain', 'prediction', 'regression', 'classification'], 'head type should be either pretrain, prediction, or regression'
        # Backbone
        #logging.info(f"c_in:{c_in}, num_patch:{num_patch}, patch_len:{patch_len}, d_model:{d_model}, d_ff:{d_ff}, target_dim:{target_dim}")
        #logging.info(f"x_categoricals:{x_categoricals}, x_reals:{x_reals}, max_encoder_length:{max_encoder_length}, stride:{stride}")
        #exit(0)
        
        self.x_cat_patch = Patch(max_encoder_length, stride, c_in=len(x_categoricals), num_patch=num_patch, patch_len=patch_len,
                                 n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                                 shared_embedding=shared_embedding, d_ff=d_ff,
                                 attn_dropout=attn_dropout, dropout=dropout, act=act,
                                 res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                 pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
        self.x_cont_patch = Patch(max_encoder_length, stride, c_in=len(x_reals), num_patch=num_patch, patch_len=patch_len,
                                  n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                                  shared_embedding=shared_embedding, d_ff=d_ff,
                                  attn_dropout=attn_dropout, dropout=dropout, act=act,
                                  res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
        # Head
        self.n_vars = c_in
        self.head_type = head_type

        if head_type == "pretrain":
            self.head = PretrainHead(d_model, patch_len, head_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == "prediction":
            self.head = PredictionHead(individual, self.n_vars, d_model, num_patch, target_dim, head_dropout)
        elif head_type == "regression":
            self.head = RegressionHead(self.n_vars, d_model, target_dim, head_dropout, y_range)
        elif head_type == "classification":
            self.head = ClassificationHead(self.n_vars, d_model, target_dim, head_dropout)
        # attention for long-range processing
        self.multihead_attn = InterpretableMultiHeadAttention(
            #d_model=self.hparams.hidden_size, n_head=self.hparams.attention_head_size, dropout=self.hparams.dropout
            d_model=d_model, n_head=self.hparams.attention_head_size, dropout=self.hparams.dropout
        )
        self.position_head = PredictionHead(individual, self.n_vars, d_model, num_patch, target_dim, head_dropout)
        self.post_attn_gate_norm = GateAddNorm(
            #self.hparams.hidden_size, dropout=self.hparams.dropout, trainable_add=False
            d_model, dropout=self.hparams.dropout, trainable_add=False
        )
        self.pos_wise_ff = GatedResidualNetwork(
            #self.hparams.hidden_size, self.hparams.hidden_size, self.hparams.hidden_size, dropout=self.hparams.dropout
            d_model, d_model, d_model, dropout=self.hparams.dropout
        )

        # output processing -> no dropout at this late stage
        #self.pre_output_gate_norm = GateAddNorm(self.hparams.hidden_size, dropout=None, trainable_add=False)
        self.pre_output_gate_norm = GateAddNorm(d_model, dropout=None, trainable_add=False)

        self.individual = False
        #self.head_nf = d_model * num_patch
        self.head_nf = d_model
        #logging.info(f"n_vars:{self.n_vars}, head_nf:{self.head_nf}, target_dim:{target_dim}")
        self.flatten_head = Flatten_Head(self.individual, self.n_vars, self.prediction_num_patch, self.head_nf, target_dim, head_dropout=head_dropout)
                

        output_size = self.hparams.output_size
        returns_output_size = None
        position_output_size = None
        if isinstance(output_size, Dict):
            returns_output_size = output_size["returns_prediction"]
            if "position_optimization" in output_size:
                position_output_size = output_size["position_optimization"]
        #logging.info(f"returns_output_size:{returns_output_size}")
        #logging.info(f"position_output_size:{position_output_size}")
        if self.n_head_targets(head="returns_prediction") > 1:  # if to run with multiple targets
            self.output_layer = nn.ModuleList(
                [nn.Linear(d_model, output_size) for output_size in returns_output_size]
            )
        else:
            self.output_layer = nn.Linear(d_model, returns_output_size)
        self.position_output_layer = None
        if position_output_size:
            if self.n_head_targets(head="position_optimization") > 1:  # if to run with multiple targets
                self.position_output_layer = nn.ModuleList(
                    [nn.Linear(d_model, output_size) for output_size in position_output_size]
                )
            else:
                self.position_output_layer = nn.Linear(d_model, position_output_size)
        #logging.info(f"output_layer:{self.output_layer}")
        #logging.info(f"position_output_layer:{self.position_output_layer}")


    def n_head_targets(self, head) -> int:
        """
        Number of targets to forecast.

        Based on loss function.

        Returns:
            int: number of targets
        """
        loss = self.loss_per_head[head]
        if isinstance(loss, MultiLoss):
            return len(loss.metrics)
        else:
            return 1

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: List[str] = None,
        **kwargs,
    ):
        """
        Create model from dataset.

        Args:
            dataset: timeseries dataset
            allowed_encoder_known_variable_names: List of known variables that are allowed in encoder, defaults to all
            **kwargs: additional arguments such as hyperparameters for model (see ``__init__()``)

        Returns:
            TemporalFusionTransformer
        """
        # add maximum encoder length
        # update defaults
        new_kwargs = copy(kwargs)
        new_kwargs["c_in"] = len(dataset.reals)
        new_kwargs["target_dim"] = dataset.max_prediction_length
        new_kwargs["x_reals"] = dataset.reals
        new_kwargs["max_encoder_length"]=dataset.max_encoder_length
        new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs, QuantileLoss()))

        # create class and return
        return super().from_dataset(
            dataset, allowed_encoder_known_variable_names=allowed_encoder_known_variable_names, **new_kwargs
        )

    def expand_static_context(self, context, timesteps):
        """
        add time dimension to static context
        """
        #logging.info(f"static_context:{context.shape}, timestep:{timesteps}")
        return context[:, None].expand(-1, timesteps, -1)

    def get_attention_mask(self, encoder_lengths: torch.LongTensor, decoder_lengths: torch.LongTensor):
        """
        Returns causal mask to apply for self-attention layer.
        """
        decoder_length = decoder_lengths.max()
        if self.hparams.causal_attention:
            # indices to which is attended
            attend_step = torch.arange(decoder_length, device=self.device)
            # indices for which is predicted
            predict_step = torch.arange(0, decoder_length, device=self.device)[:, None]
            # do not attend to steps to self or after prediction
            decoder_mask = (attend_step >= predict_step).unsqueeze(0).expand(encoder_lengths.size(0), -1, -1)
        else:
            # there is value in attending to future forecasts if they are made with knowledge currently
            #   available
            #   one possibility is here to use a second attention layer for future attention (assuming different effects
            #   matter in the future than the past)
            #   or alternatively using the same layer but allowing forward attention - i.e. only
            #   masking out non-available data and self
            decoder_mask = create_mask(decoder_length, decoder_lengths).unsqueeze(1).expand(-1, decoder_length, -1)
        # do not attend to steps where data is padded
        encoder_mask = create_mask(encoder_lengths.max(), encoder_lengths).unsqueeze(1).expand(-1, decoder_length, -1)
        # combine masks along attended time - first encoder and then decoder
        mask = torch.cat(
            (
                encoder_mask,
                decoder_mask,
            ),
            dim=2,
        )
        return mask

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        input dimensions: n_samples x time x variables
        """
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        max_encoder_length = int(encoder_lengths.max())
        max_decoder_length = int(decoder_lengths.max())
        new_encoder_length = int(max_encoder_length / self.stride) - 1
        new_decoder_length = int(max_decoder_length / self.stride) - 1
        encoder_lengths = torch.ones_like(encoder_lengths) * new_encoder_length
        decoder_lengths = torch.ones_like(decoder_lengths) * new_decoder_length
        #logging.info(f"encoder_lengths:{encoder_lengths}, new_encoder_length:{new_encoder_length}")
        #logging.info(f"decoder_lengths:{decoder_lengths}, new_decoder_length:{new_decoder_length}")
        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)  # concatenate in time dimension
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)  # concatenate in time dimension
        #logging.info(f"x_cat:{x_cat.shape}")
        #logging.info(f"x_cont:{x_cont.shape}")
        x_cont = self.x_cont_patch(x_cont)
        # bs x n_vars x num_patch x d_model
        #logging.info(f"after_x_cont:{x_cont.shape}")
        x_cat = self.x_cat_patch(x_cat)
        #exit(0)
        timesteps = x_cont.size(2)-self.skipped_patch  # encode + decode length - 1 (spanning encode/decode)
        #logging.info(f"after_x_cat:{x_cat.shape}, timesteps:{timesteps}")
        input_vectors = self.input_embeddings(x_cat)
        input_vectors.update(
            {
                name: x_cont[:, idx, ...]
                #name: x_cont[:, idx, ...].unsqueeze(1)
                for idx, name in enumerate(self.hparams.x_reals)
                if name in self.reals
            }
        )
        #logging.info(f"input_vectors['close_back_cumsum']:{input_vectors['close_back_cumsum'].shape}, timesteps:{timesteps}")

        # Embedding and variable selection
        if len(self.static_variables) > 0:
            # static embeddings will be constant over entire batch
            static_embedding = {name: input_vectors[name][:, 0] for name in self.static_variables}
            static_embedding, static_variable_selection = self.static_variable_selection(static_embedding)
        else:
            static_embedding = torch.zeros(
                (x_cont.size(0), self.d_model), dtype=self.dtype, device=self.device
            )
            static_variable_selection = torch.zeros((x_cont.size(0), 0), dtype=self.dtype, device=self.device)

        static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(static_embedding), timesteps
        )

        #logging.info(f"self.encoder_variables:{self.encoder_variables}, new_encoder_length:{new_encoder_length}")
        embeddings_varying_encoder = {
            name: input_vectors[name][:, :new_encoder_length] for name in self.encoder_variables
        }
        #logging.info(f"embeddings_varying_encoder:{embeddings_varying_encoder['relative_time_idx'].shape}")
        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            embeddings_varying_encoder,
            static_context_variable_selection[:, :new_encoder_length],
        )

        #logging.info(f"self.decoder_variables:{self.decoder_variables}")
        embeddings_varying_decoder = {
            # When patching, there is one spanning encoder and decoder and needs to be filtered out
            name: input_vectors[name][:, new_encoder_length+self.skipped_patch:] for name in self.decoder_variables  # select decoder
        }
        #logging.info(f"embeddings_varying_decoder before initial variable:{embeddings_varying_decoder['relative_time_idx'].shape}")
        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_variable_selection(
            embeddings_varying_decoder,
            #static_context_variable_selection[:, new_encoder_length+1:],
            static_context_variable_selection[:, new_encoder_length:],
        )
        #logging.info(f"embeddings_varying_decoder_after_variale_selection:{embeddings_varying_decoder.shape}")

        # LSTM
        # calculate initial state
        input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )
        input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(self.hparams.lstm_layers, -1, -1)
        #logging.info(f"input_hidden:{input_hidden.shape}, input_cell:{input_cell.shape}")

        # run local encoder
        #logging.info(f"embeddings_varying_encoder before reshape:{embeddings_varying_encoder.shape}")
        # We need to reshape it into 3 dim since that is what is expected by LSTM
        embeddings_varying_encoder = torch.reshape(embeddings_varying_encoder,
                                                   (embeddings_varying_encoder.shape[0],
                                                    embeddings_varying_encoder.shape[1],
                                                    -1))
        #logging.info(f"embeddings_varying_encoder after reshape:{embeddings_varying_encoder.shape}, encoder_lengths:{encoder_lengths}, input_hidden:{input_hidden}, input_cell:{input_cell}")
        encoder_output, (hidden, cell) = self.lstm_encoder(
            embeddings_varying_encoder, (input_hidden, input_cell),
            lengths=encoder_lengths, enforce_sorted=False
        )
        #logging.info(f"encoder_output:{encoder_output.shape}")
        
        #logging.info(f"embeddings_varying_decoder_before_reshape:{embeddings_varying_decoder.shape}")
        embeddings_varying_decoder = torch.reshape(embeddings_varying_decoder,
                                                   (embeddings_varying_decoder.shape[0],
                                                    embeddings_varying_decoder.shape[1],
                                                    -1))
        #logging.info(f"embedding_varying_decoder:{embeddings_varying_decoder.shape}, hidden:{hidden.shape}, cell:{cell.shape}")
        # run local decoder
        decoder_output, _ = self.lstm_decoder(
            embeddings_varying_decoder,
            (hidden, cell),
            lengths=decoder_lengths,
            enforce_sorted=False,
        )
        #logging.info(f"deocder_output:{decoder_output.shape}")
        # skip connection over lstm
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(lstm_output_encoder, embeddings_varying_encoder)

        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(lstm_output_decoder, embeddings_varying_decoder)

        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)
        #logging.info(f"lstm_output:{lstm_output.shape}")
        # static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output, self.expand_static_context(static_context_enrichment, timesteps)
        )

        # Attention
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, new_encoder_length:],  # query only for predictions
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(encoder_lengths=encoder_lengths, decoder_lengths=decoder_lengths),
        )
        #logging.info(f"attn_output:{attn_output.shape}, attn_input:{attn_input.shape}")
        # skip connection over attention
        attn_output = self.post_attn_gate_norm(attn_output, attn_input[:, new_encoder_length:])

        output = self.pos_wise_ff(attn_output)
        #logging.info(f"pos_wise_ff_output:{output.shape}")
        # skip connection over temporal fusion decoder (not LSTM decoder despite the LSTM output contains
        # a skip from the variable selection network)
        output = self.pre_output_gate_norm(output, lstm_output[:, new_encoder_length:])
        #logging.info(f"output_shape before:{output.shape}")

        #output = torch.reshape(output, (output.shape(0), n_vars, output.shape[-2],output.shape[-1]))
        # z: [bs x nvars x d_model]
        #output = output.permute(0, 2, 3, 1)
        # z: [bs x nvars x d_model]
        embedding = self.flatten_head(output)
        # z: [bs x nvars x target_dim]
        #logging.info(f"output_shape after flatten:{output.shape}")
        #logging.info(f"output_layer:{self.output_layer}")
        if self.n_head_targets(head="returns_prediction") > 1:  # if to run with multiple targets
            output = [output_layer(embedding) for output_layer in self.output_layer]
        else:
            output = self.output_layer(embedding)
        position_output = None
        if self.position_output_layer:
            if self.n_head_targets(head="position_optimization") > 1:  # if to run with multiple targets
                position_output = [output_layer(embedding) for output_layer in self.position_output_layer]
            else:
                position_output = self.position_output_layer(embedding)
        # Remove last dimension if it is 1
        #logging.info(f"output before squeeze:{output[0].shape}, {output[1].shape}")
        if isinstance(output, List):
          output = [ torch.squeeze(val, dim=-1) for val in output]
        else:
          output = torch.squeeze(output, dim=-1)
        if position_output:
            if isinstance(position_output, List):
                position_output = [ torch.squeeze(val, dim=-1) for val in position_output]
            else:
                position_output = torch.squeeze(position_output, dim=-1)
            output = [output, position_output]
        #logging.info(f"output:{output.shape}")
        #logging.info(f"position_output:{position_output.shape}")
        
        return self.to_network_output(
            prediction=self.transform_output(output,
                                             target_scale=x["target_scale"]),
            encoder_attention=attn_output_weights[..., :new_encoder_length],
            decoder_attention=attn_output_weights[..., new_encoder_length:],
            static_variables=static_variable_selection,
            encoder_variables=encoder_sparse_weights,
            decoder_variables=decoder_sparse_weights,
            decoder_lengths=decoder_lengths,
            encoder_lengths=encoder_lengths,
        )

    def on_fit_end(self):
        if self.log_interval > 0:
            self.log_embeddings()

    def create_log(self, x, y, out, batch_idx, **kwargs):
        log = super().create_log(x, y, out, batch_idx, **kwargs)
        if self.log_interval > 0:
            log["interpretation"] = self._log_interpretation(out)
        return log

    def _log_interpretation(self, out):
        # calculate interpretations etc for latter logging
        interpretation = self.interpret_output(
            detach(out),
            reduction="sum",
            attention_prediction_horizon=0,  # attention only for first prediction horizon
        )
        return interpretation

    def on_epoch_end(self, outputs):
        """
        run at epoch end for training or validation
        """
        if self.log_interval > 0 and not self.training:
            self.log_interpretation(outputs)

    def interpret_output(
        self,
        out: Dict[str, torch.Tensor],
        reduction: str = "none",
        attention_prediction_horizon: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        interpret output of model

        Args:
            out: output as produced by ``forward()``
            reduction: "none" for no averaging over batches, "sum" for summing attentions, "mean" for
                normalizing by encode lengths
            attention_prediction_horizon: which prediction horizon to use for attention

        Returns:
            interpretations that can be plotted with ``plot_interpretation()``
        """
        # take attention and concatenate if a list to proper attention object
        batch_size = len(out["decoder_attention"])
        if isinstance(out["decoder_attention"], (list, tuple)):
            # start with decoder attention
            # assume issue is in last dimension, we need to find max
            max_last_dimension = max(x.size(-1) for x in out["decoder_attention"])
            first_elm = out["decoder_attention"][0]
            # create new attention tensor into which we will scatter
            decoder_attention = torch.full(
                (batch_size, *first_elm.shape[:-1], max_last_dimension),
                float("nan"),
                dtype=first_elm.dtype,
                device=first_elm.device,
            )
            # scatter into tensor
            for idx, x in enumerate(out["decoder_attention"]):
                decoder_length = out["decoder_lengths"][idx]
                decoder_attention[idx, :, :, :decoder_length] = x[..., :decoder_length]
        else:
            decoder_attention = out["decoder_attention"].clone()
            decoder_mask = create_mask(out["decoder_attention"].size(1), out["decoder_lengths"])
            decoder_attention[decoder_mask[..., None, None].expand_as(decoder_attention)] = float("nan")

        if isinstance(out["encoder_attention"], (list, tuple)):
            # same game for encoder attention
            # create new attention tensor into which we will scatter
            first_elm = out["encoder_attention"][0]
            encoder_attention = torch.full(
                (batch_size, *first_elm.shape[:-1], self.hparams.max_encoder_length),
                float("nan"),
                dtype=first_elm.dtype,
                device=first_elm.device,
            )
            # scatter into tensor
            for idx, x in enumerate(out["encoder_attention"]):
                encoder_length = out["encoder_lengths"][idx]
                encoder_attention[idx, :, :, self.hparams.max_encoder_length - encoder_length :] = x[
                    ..., :encoder_length
                ]
        else:
            # roll encoder attention (so start last encoder value is on the right)
            encoder_attention = out["encoder_attention"].clone()
            shifts = encoder_attention.size(3) - out["encoder_lengths"]
            new_index = (
                torch.arange(encoder_attention.size(3), device=encoder_attention.device)[None, None, None].expand_as(
                    encoder_attention
                )
                - shifts[:, None, None, None]
            ) % encoder_attention.size(3)
            encoder_attention = torch.gather(encoder_attention, dim=3, index=new_index)
            # expand encoder_attentiont to full size
            if encoder_attention.size(-1) < self.hparams.max_encoder_length:
                encoder_attention = torch.concat(
                    [
                        torch.full(
                            (
                                *encoder_attention.shape[:-1],
                                self.hparams.max_encoder_length - out["encoder_lengths"].max(),
                            ),
                            float("nan"),
                            dtype=encoder_attention.dtype,
                            device=encoder_attention.device,
                        ),
                        encoder_attention,
                    ],
                    dim=-1,
                )

        # combine attention vector
        attention = torch.concat([encoder_attention, decoder_attention], dim=-1)
        attention[attention < 1e-5] = float("nan")

        # histogram of decode and encode lengths
        encoder_length_histogram = integer_histogram(out["encoder_lengths"], min=0, max=self.hparams.max_encoder_length)
        decoder_length_histogram = integer_histogram(
            out["decoder_lengths"], min=1, max=out["decoder_variables"].size(1)
        )

        # mask where decoder and encoder where not applied when averaging variable selection weights
        encoder_variables = out["encoder_variables"].squeeze(-2).clone()
        encode_mask = create_mask(encoder_variables.size(1), out["encoder_lengths"])
        encoder_variables = encoder_variables.masked_fill(encode_mask.unsqueeze(-1), 0.0).sum(dim=1)
        encoder_variables /= (
            out["encoder_lengths"]
            .where(out["encoder_lengths"] > 0, torch.ones_like(out["encoder_lengths"]))
            .unsqueeze(-1)
        )

        decoder_variables = out["decoder_variables"].squeeze(-2).clone()
        decode_mask = create_mask(decoder_variables.size(1), out["decoder_lengths"])
        decoder_variables = decoder_variables.masked_fill(decode_mask.unsqueeze(-1), 0.0).sum(dim=1)
        decoder_variables /= out["decoder_lengths"].unsqueeze(-1)

        # static variables need no masking
        static_variables = out["static_variables"].squeeze(1)
        # attention is batch x time x heads x time_to_attend
        # average over heads + only keep prediction attention and attention on observed timesteps
        attention = masked_op(
            attention[
                :, attention_prediction_horizon, :, : self.hparams.max_encoder_length + attention_prediction_horizon
            ],
            op="mean",
            dim=1,
        )

        if reduction != "none":  # if to average over batches
            static_variables = static_variables.sum(dim=0)
            encoder_variables = encoder_variables.sum(dim=0)
            decoder_variables = decoder_variables.sum(dim=0)

            attention = masked_op(attention, dim=0, op=reduction)
        else:
            attention = attention / masked_op(attention, dim=1, op="sum").unsqueeze(-1)  # renormalize

        interpretation = dict(
            attention=attention.masked_fill(torch.isnan(attention), 0.0),
            static_variables=static_variables,
            encoder_variables=encoder_variables,
            decoder_variables=decoder_variables,
            encoder_length_histogram=encoder_length_histogram,
            decoder_length_histogram=decoder_length_histogram,
        )
        return interpretation

    def plot_prediction(
        self,
        x: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        idx: int,
        plot_attention: bool = True,
        add_loss_to_title: bool = False,
        show_future_observed: bool = True,
        ax=None,
        row=1,
        col=1,
        **kwargs,
    ) -> plt.Figure:
        """
        Plot actuals vs prediction and attention

        Args:
            x (Dict[str, torch.Tensor]): network input
            out (Dict[str, torch.Tensor]): network output
            idx (int): sample index
            plot_attention: if to plot attention on secondary axis
            add_loss_to_title: if to add loss to title. Default to False.
            show_future_observed: if to show actuals for future. Defaults to True.
            ax: matplotlib axes to plot on

        Returns:
            plt.Figure: matplotlib figure
        """

        # plot prediction as normal
        
        fig = super().plot_prediction(
            x,
            out,
            idx=idx,
            add_loss_to_title=add_loss_to_title,
            show_future_observed=show_future_observed,
            ax=ax, row=row, col=col,
            **kwargs,
        )

        # add attention on secondary axis
        if False:
            interpretation = self.interpret_output(out.iget(slice(idx, idx + 1)))
            #fig = list(ax)
            #for f in to_list(fig):
            for f in [ax]:
                encoder_length = x["encoder_lengths"][0]
                f.add_trace(go.Scatter(x=torch.arange(-encoder_length, 0),
                                       y=interpretation["attention"][0, -encoder_length:].detach().cpu(),
                                       name="Attention", yaxis="y2", showlegend=False), secondary_y=True,
                            row=row, col=col)
                #f.update_layout(
                #    yaxis2=dict(title="Attention",
                #                overlaying="y",
                #                side="right",),
                #)
                #ax = f.axes[0]
                #ax2 = ax.twinx()
                #ax2.set_ylabel("Attention")
                #ax2.plot(
                #    torch.arange(-encoder_length, 0),
                #    interpretation["attention"][0, -encoder_length:].detach().cpu(),
                #    alpha=0.2,
                #    color="k",
                #)
                #f.tight_layout()
        return fig

    def plot_interpretation(self, interpretation: Dict[str, torch.Tensor], ax=None,
                            cells = None,
                            **kwargs) -> Dict[str, plt.Figure]:
        """
        Make figures that interpret model.

        * Attention
        * Variable selection weights / importances

        Args:
            interpretation: as obtained from ``interpret_output()``

        Returns:
            dictionary of matplotlib figures
        """
        figs = {}
        # attention
        #fig = make_subplots(1, 1)
        fig = ax
        #fig, ax = plt.subplots()
        #logging.info(f"interpretation:{interpretation}")
        attention = interpretation["attention"].detach().cpu()
        attention = attention / attention.sum(-1).unsqueeze(-1)
        #fig.update_layout(
        #    xaxis=dict(
        #        title="Time index"
        #    ),
        #    yaxis=dict(
        #        title="Attention"
        #    )
        #)
        fig.add_trace(
            go.Scatter(x=np.arange(-self.hparams.max_encoder_length, attention.size(0) -
                                   self.hparams.max_encoder_length),
                       y=attention, name="Attention", showlegend=False), row=cells[0]["row"], col=cells[0]["col"])
        #ax.plot(
        #    np.arange(-self.hparams.max_encoder_length, attention.size(0) - self.hparams.max_encoder_length), attention
                       #)
        #ax.set_xlabel("Time index")
        #ax.set_ylabel("Attention")
        #ax.set_title("Attention")
        figs["attention"] = fig

        # variable selection
        def make_selection_plot(title, values, labels, row, col):
            #fig, ax = plt.subplots(figsize=(7, len(values) * 0.25 + 2))
            #fig = make_subplots(1, 1)
            #fig.update_layout(
            #    autosize=False,
            #    width=10,
            #    xaxis=dict(
            #        title="Importance in %",
            #    ),
            #    height=len(values) * 0.25 + 10,)
            order = np.argsort(values)
            values = values / values.sum(-1).unsqueeze(-1)
            #logging.info(f"labels:{labels}, values:{values}")
            fig.add_trace(
                go.Bar(
                    #x=np.arange(len(values)),
                    y=np.asarray(labels)[order],
                    #y=values[order] * 100,
                    x=values[order] * 100,
                    name=title, showlegend=False,
                    orientation='h'), row=row, col=col)
            #ax.barh(np.arange(len(values)), values[order] * 100, tick_label=np.asarray(labels)[order])
            #ax.set_title(title)
            #ax.set_xlabel("Importance in %")
            #plt.tight_layout()
            return fig

        figs["static_variables"] = make_selection_plot(
            "Static variables importance", interpretation["static_variables"].detach().cpu(), self.static_variables,
            cells[1]["row"], cells[1]["col"]
        )
        figs["encoder_variables"] = make_selection_plot(
            "Encoder variables importance", interpretation["encoder_variables"].detach().cpu(), self.encoder_variables,
            cells[2]["row"], cells[2]["col"]
        )
        figs["decoder_variables"] = make_selection_plot(
            "Decoder variables importance", interpretation["decoder_variables"].detach().cpu(), self.decoder_variables,
            cells[3]["row"], cells[3]["col"]
        )

        return figs
    
    def log_interpretation(self, outputs):
        """
        Log interpretation metrics to tensorboard.
        """
        # extract interpretations
        interpretation = {
            # use padded_stack because decoder length histogram can be of different length
            name: padded_stack([x["interpretation"][name].detach() for x in outputs], side="right", value=0).sum(0)
            for name in outputs[0]["interpretation"].keys()
        }
        # normalize attention with length histogram squared to account for: 1. zeros in attention and
        # 2. higher attention due to less values
        attention_occurances = interpretation["encoder_length_histogram"][1:].flip(0).float().cumsum(0)
        attention_occurances = attention_occurances / attention_occurances.max()
        attention_occurances = torch.cat(
            [
                attention_occurances,
                torch.ones(
                    interpretation["attention"].size(0) - attention_occurances.size(0),
                    dtype=attention_occurances.dtype,
                    device=attention_occurances.device,
                ),
            ],
            dim=0,
        )
        interpretation["attention"] = interpretation["attention"] / attention_occurances.pow(2).clamp(1.0)
        interpretation["attention"] = interpretation["attention"] / interpretation["attention"].sum()

        figs = self.plot_interpretation(interpretation)  # make interpretation figures
        label = self.current_stage
        # log to tensorboard
        for name, fig in figs.items():
            self.logger.experiment.add_figure(
                f"{label.capitalize()} {name} importance", fig, global_step=self.global_step
            )

        # log lengths of encoder/decoder
        for type in ["encoder", "decoder"]:
            fig, ax = plt.subplots()
            lengths = (
                padded_stack([out["interpretation"][f"{type}_length_histogram"] for out in outputs])
                .sum(0)
                .detach()
                .cpu()
            )
            if type == "decoder":
                start = 1
            else:
                start = 0
            ax.plot(torch.arange(start, start + len(lengths)), lengths)
            ax.set_xlabel(f"{type.capitalize()} length")
            ax.set_ylabel("Number of samples")
            ax.set_title(f"{type.capitalize()} length distribution in {label} epoch")

            self.logger.experiment.add_figure(
                f"{label.capitalize()} {type} length distribution", fig, global_step=self.global_step
            )

    def log_embeddings(self):
        """
        Log embeddings to tensorboard
        """
        for name, emb in self.input_embeddings.items():
            labels = self.hparams.embedding_labels[name]
            self.logger.experiment.add_embedding(
                emb.weight.data.detach().cpu(), metadata=labels, tag=name, global_step=self.global_step
            )
