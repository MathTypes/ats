"""
Implementation of ``nn.Modules`` for temporal fusion transformer.
"""
import math
import logging
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDistributed(nn.Module):
    def __init__(self, module: nn.Module, batch_first: bool = False):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


class TimeDistributedInterpolation(nn.Module):
    def __init__(self, output_size: int, batch_first: bool = False, trainable: bool = False):
        super().__init__()
        self.output_size = output_size
        self.batch_first = batch_first
        self.trainable = trainable
        if self.trainable:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float32))
            self.gate = nn.Sigmoid()

    def interpolate(self, x):
        upsampled = F.interpolate(x.unsqueeze(1), self.output_size, mode="linear", align_corners=True).squeeze(1)
        if self.trainable:
            upsampled = upsampled * self.gate(self.mask.unsqueeze(0)) * 2.0
        return upsampled

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.interpolate(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.interpolate(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit"""

    def __init__(self, input_size: int, hidden_size: int = None, dropout: float = None):
        super().__init__()

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout
        self.hidden_size = hidden_size or input_size
        self.fc = nn.Linear(input_size, self.hidden_size * 2)

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                torch.nn.init.zeros_(p)
            elif "fc" in n:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        x = F.glu(x, dim=-1)
        return x


class ResampleNorm(nn.Module):
    def __init__(self, input_size: int, output_size: int = None, trainable_add: bool = True):
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.output_size = output_size or input_size

        if self.input_size != self.output_size:
            self.resample = TimeDistributedInterpolation(self.output_size, batch_first=True, trainable=False)

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_size != self.output_size:
            x = self.resample(x)

        if self.trainable_add:
            x = x * self.gate(self.mask) * 2.0

        output = self.norm(x)
        return output


class AddNorm(nn.Module):
    def __init__(self, input_size: int, skip_size: int = None, trainable_add: bool = True):
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.skip_size = skip_size or input_size

        if self.input_size != self.skip_size:
            self.resample = TimeDistributedInterpolation(self.input_size, batch_first=True, trainable=False)

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.input_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.input_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        if self.input_size != self.skip_size:
            skip = self.resample(skip)

        if self.trainable_add:
            skip = skip * self.gate(self.mask) * 2.0

        #logging.info(f"x:{x.shape}, skip:{skip.shape}")
        if skip.dim() == x.dim()-1:
            skip = skip[..., None, :,:]
            #logging.info(f"x:{x.shape}, skip:{skip.shape}")
        output = self.norm(x + skip)
        return output


class GateAddNorm(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = None,
        skip_size: int = None,
        trainable_add: bool = False,
        dropout: float = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size or input_size
        self.skip_size = skip_size or self.hidden_size
        self.dropout = dropout

        self.glu = GatedLinearUnit(self.input_size, hidden_size=self.hidden_size, dropout=self.dropout)
        self.add_norm = AddNorm(self.hidden_size, skip_size=self.skip_size, trainable_add=trainable_add)

    def forward(self, x, skip):
        output = self.glu(x)
        output = self.add_norm(output, skip)
        return output


class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: int = None,
        residual: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.residual = residual

        if self.input_size != self.output_size and not self.residual:
            residual_size = self.input_size
        else:
            residual_size = self.output_size

        if self.output_size != residual_size:
            self.resample_norm = ResampleNorm(residual_size, self.output_size)

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.elu = nn.ELU()

        if self.context_size is not None:
            self.context = nn.Linear(self.context_size, self.hidden_size, bias=False)

        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.init_weights()

        self.gate_norm = GateAddNorm(
            input_size=self.hidden_size,
            skip_size=self.output_size,
            hidden_size=self.output_size,
            dropout=self.dropout,
            trainable_add=False,
        )

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" in name:
                torch.nn.init.zeros_(p)
            elif "fc1" in name or "fc2" in name:
                torch.nn.init.kaiming_normal_(p, a=0, mode="fan_in", nonlinearity="leaky_relu")
            elif "context" in name:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x, context=None, residual=None):
        if residual is None:
            residual = x

        if self.input_size != self.output_size and not self.residual:
            residual = self.resample_norm(residual)

        x = self.fc1(x)
        if context is not None:
            #logging.info(f"self.context:{self.context}, context:{context.shape}, x:{x.shape}")
            context = self.context(context)
            if context.dim() == x.dim()-1:
                context = context[..., None, :, :]
            #logging.info(f"context:{context.shape}, x:{x.shape}")
            x = x + context
        x = self.elu(x)
        x = self.fc2(x)
        x = self.gate_norm(x, residual)
        return x


class VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        input_sizes: Dict[str, int],
        hidden_size: int,
        input_embedding_flags: Dict[str, bool] = {},
        dropout: float = 0.1,
        context_size: int = None,
        single_variable_grns: Dict[str, GatedResidualNetwork] = {},
        prescalers: Dict[str, nn.Linear] = {},
        reduce: bool = True
    ):
        """
        Calcualte weights for ``num_inputs`` variables  which are each of size ``input_size``
        """
        super().__init__()
        self.reduce = reduce
        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.input_embedding_flags = input_embedding_flags
        self.dropout = dropout
        self.context_size = context_size
        #logging.info(f"input_size_total:{self.input_size_total}")
        if self.num_inputs >= 1:
            if self.context_size is not None:
                self.flattened_grn = GatedResidualNetwork(
                    self.input_size_total,
                    min(self.hidden_size, self.num_inputs),
                    self.num_inputs,
                    self.dropout,
                    self.context_size,
                    residual=False,
                )
            else:
                self.flattened_grn = GatedResidualNetwork(
                    self.input_size_total,
                    min(self.hidden_size, self.num_inputs),
                    self.num_inputs,
                    self.dropout,
                    residual=False,
                )

        self.single_variable_grns = nn.ModuleDict()
        self.prescalers = nn.ModuleDict()
        #logging.info(f"self.input_embedding_flags:{self.input_embedding_flags}")
        for name, input_size in self.input_sizes.items():
            if name in single_variable_grns:
                self.single_variable_grns[name] = single_variable_grns[name]
            elif self.input_embedding_flags.get(name, False):
                self.single_variable_grns[name] = ResampleNorm(input_size, self.hidden_size)
            else:
                self.single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, self.hidden_size),
                    output_size=self.hidden_size,
                    dropout=self.dropout,
                )
            if name in prescalers:  # reals need to be first scaled up
                self.prescalers[name] = prescalers[name]
            elif not self.input_embedding_flags.get(name, False):
                #logging.info(f"adding prescalers, {name}")
                self.prescalers[name] = nn.Linear(1, input_size)

        self.softmax = nn.Softmax(dim=-1)

    @property
    def input_size_total(self):
        return sum(size if name in self.input_embedding_flags else size for name, size in self.input_sizes.items())

    @property
    def num_inputs(self):
        return len(self.input_sizes)

    def forward(self, x: Dict[str, torch.Tensor], context: torch.Tensor = None):
        if self.num_inputs >= 1:
            # transform single variables
            var_outputs = []
            weight_inputs = []
            for name in self.input_sizes.keys():
                # select embedding belonging to a single input
                variable_embedding = x[name]
                if name in self.prescalers:
                    variable_embedding = self.prescalers[name](variable_embedding)
                #logging.info(f"variable_embedding:{variable_embedding.shape}")
                weight_inputs.append(variable_embedding)
                var_outputs.append(self.single_variable_grns[name](variable_embedding))
            var_outputs = torch.stack(var_outputs, dim=-1)
            # logging.info(f"x:{x['relative_time_idx'].shape}")
            #logging.info(f"context:{context.shape}")
            #logging.info(f"var_outputs:{var_outputs.shape}")
            # calculate variable weights
            flat_embedding = torch.cat(weight_inputs, dim=-1)
            #logging.info(f"flat_embedding:{flat_embedding.shape}")
            sparse_weights = self.flattened_grn(flat_embedding, context)
            sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)
            #logging.info(f"sparse_weights:{sparse_weights.shape}")
            outputs = var_outputs * sparse_weights
            #logging.info(f"outputs before sum:{outputs.shape}")
            if self.reduce:
                outputs = outputs.sum(dim=-1)
            #logging.info(f"outputs after sum:{outputs.shape}")
        else:  # for one input, do not perform variable selection but just encoding
            name = next(iter(self.single_variable_grns.keys()))
            variable_embedding = x[name]
            if name in self.prescalers:
                variable_embedding = self.prescalers[name](variable_embedding)
            outputs = self.single_variable_grns[name](variable_embedding)  # fast forward if only one variable
            #logging.info(f"single variable: outputs:{outputs.shape}")
            # outputs: bs x n_variables, hidden_size
            if outputs.ndim == 3:  # -> batch size, time, hidden size, n_variables
                sparse_weights = torch.ones(outputs.size(0), outputs.size(1), 1, 1, device=outputs.device)  #
            else:  # ndim == 2 -> batch size, hidden size, n_variables
                sparse_weights = torch.ones(outputs.size(0), 1, 1, device=outputs.device)
        #logging.info(f"outputs:{outputs.shape}, sparse_weights:{sparse_weights.shape}")
        return outputs, sparse_weights


class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=160):
        super().__init__()
        assert d_model % 2 == 0, "model dimension has to be multiple of 2 (encode sin(pos) and cos(pos))"
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(0)
            pe = self.pe[:, :seq_len].view(seq_len, 1, self.d_model)
            x = x + pe
            return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, win_size, dropout: float = None, scale: bool = True):
        super(ScaledDotProductAttention, self).__init__()
        self.window_size = win_size
        self.distances = torch.zeros((win_size, win_size)).cuda()
        for i in range(win_size):
            for j in range(win_size):
                self.distances[i][j] = abs(i - j)
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, sigma, mask=None):
        #logging.info(f"queries:{q.shape}, key:{k.shape}, values:{v.shape}, sigma:{sigma.shape}, mask:{mask.shape}")
        # queries:torch.Size([512, 3, 4, 5]),
        # key:torch.Size([512, 6, 4, 5]),
        # values:torch.Size([512, 6, 4, 5]),
        # sigma:torch.Size([512, 3, 4])
        B, L, H, E = q.shape
        _, S, _, D = v.shape

        attn = torch.einsum("blhe,bshe->bhls", q, k)
        #logging.info(f"attn after einsum:{attn.shape}")
        #if self.mask:
        #    if attn_mask is None:
        #        attn_mask = TriangularCausalMask(B, L, device=queries.device)
        #    scores.masked_fill_(attn_mask.mask, -np.inf)
        #attn = scale * scores
        
        #attn = torch.bmm(q, k.permute(0, 2, 1))  # query-key overlap
        if self.scale:
            dimension = torch.as_tensor(k.size(-1), dtype=attn.dtype, device=attn.device).sqrt()
            attn = attn / dimension

        sigma = sigma.transpose(1, 2)  # B L H ->  B H L [ 512, 4, 3]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, self.window_size)  # B H L L
        mask = mask.unsqueeze(1).repeat(1, H, 1, 1) # B H L S
        #logging.info(f"sigma after unsqueeze:{sigma.shape}")
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))
        #logging.info(f"prior after unsqueeze: {prior.shape}")
        #logging.info(f"attn.shape:{attn.shape}")
        if mask is not None:
            #logging.info(f"mask.shape:{mask.shape}")
            _MASKING_VALUE = -1e+9 if attn.dtype == torch.float32 else -1e+4
            attn = attn.masked_fill(mask, _MASKING_VALUE)

        attn = self.softmax(attn)
        #logging.info(f"attn:{attn.shape}")
        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.einsum("bhls,bshd->blhd", attn, v)
        #output = torch.bmm(attn, v) # only works for dim=3
        return output.contiguous(), attn, prior, sigma


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, windows_size, n_head: int, d_model: int, dropout: float = 0.0):
        super(InterpretableMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        #logging.info(f"d_k:{self.d_k}, d_model:{d_model}, n_head:{n_head}")
        self.dropout = nn.Dropout(p=dropout)
        self.query_projection = nn.Linear(d_model, self.d_k * n_head)
        self.key_projection = nn.Linear(d_model, self.d_k * n_head)
        self.value_projection = nn.Linear(d_model, self.d_v * n_head)
        self.sigma_projection = nn.Linear(d_model, self.n_head)
        
        self.attention = ScaledDotProductAttention(win_size=windows_size)
        self.out_projection = nn.Linear(self.d_v * n_head, d_model)
        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, q, k, v, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        #[batch_size, window_size, d_model]
        #logging.info(f"q:{q.shape}, k:{k.shape}, v:{v.shape}")
        B, L, H = q.shape
        _, S, _ = k.shape
        H = self.n_head
        x = q
        queries = self.query_projection(q).view(B, L, H, -1)
        keys = self.key_projection(k).view(B, S, H, -1)
        values = self.value_projection(v).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior, sigma = self.attention(
            queries,
            keys,
            values,
            sigma,
            mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma

