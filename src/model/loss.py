from typing import List

from pytorch_forecasting.metrics import MultiLoss

import torch
from torch import nn
from torchmetrics import Metric as LightningMetric

class MultiLossWithUncertaintyWeight(MultiLoss):
    def __init__(self, metrics: List[LightningMetric]):
        super(MultiLossWithUncertaintyWeight, self).__init__(metrics)
        self.task_num = len(metrics)
        self.log_vars = nn.Parameter(torch.zeros((self.task_num)))

    @torch.jit.unused
    def forward(self, y_pred: torch.Tensor, y_actual: torch.Tensor, **kwargs):
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
        for idx, metric in enumerate(self.metrics):
            try:
                res = metric(
                    y_pred[idx],
                    (y_actual[0][idx], y_actual[1]),
                    **{
                        name: value[idx] if isinstance(value, (list, tuple)) else value
                        for name, value in kwargs.items()
                    },
                )

            except TypeError:  # silently update without kwargs if not supported
                res = metric(y_pred[idx], (y_actual[0][idx], y_actual[1]))
            precision = torch.exp(-self.log_vars[idx])
            res = precision*res + self.log_vars[idx]
            results += res
        return results
