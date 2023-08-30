from typing import Dict, List

import traceback
import logging
from pytorch_forecasting.metrics import MultiLoss

import torch
from torch import nn
from torchmetrics import Metric as LightningMetric


class MultiLossWithUncertaintyWeight(MultiLoss):
    def __init__(self, metrics: List[LightningMetric], head_index_map: Dict[str, int]):
        super(MultiLossWithUncertaintyWeight, self).__init__(metrics)
        self.task_num = len(metrics)
        self.log_vars = nn.Parameter(torch.zeros((self.task_num)))
        self.head_index_map = head_index_map

    @torch.jit.unused
    def forward(self, y_pred: torch.Tensor, y_actual: Dict[str, torch.Tensor], **kwargs):
        """
        Calculate composite metric

        Args:
            y_pred: network output
            y_actual: actual values
            **kwargs: arguments to update function

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        results = 0
        #logging.info(f"kwargs:{kwargs}")
        #logging.info(f"y_pred:{y_pred}")
        #logging.info(f"y_actual:{y_actual}")
        for idx, metric in enumerate(self.metrics):
            try:
                # logging.info(f"metric:{metric}"
                key = self.head_index_map[idx]
                #logging.info(f"y_pred[key]:{y_pred[key]}")
                #logging.info(f"y_actual[0][key]:{y_actual[0][key]}")
                #logging.info(f"compute key:{key}, idx:{idx}, metric:{metric}")
                res = metric(
                    y_pred[key],
                    (y_actual[0][key], y_actual[1]),
                    **{
                        name: value
                        for name, value in kwargs.items()
                    },
                )

            except TypeError:  # silently update without kwargs if not supported
                res = metric(y_pred[key], (y_actual[0][key], y_actual[1]))
            precision = torch.exp(-self.log_vars[idx])
            res = precision * res + self.log_vars[idx]
            results += res
        return results
