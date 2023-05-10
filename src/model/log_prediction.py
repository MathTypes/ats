import io
from PIL import Image
from datetime import datetime
import logging
from pytorch_lightning.callbacks import Callback
import numpy as np
import wandb
import torch
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import PIL
import inference
PIL.Image.MAX_IMAGE_PIXELS = None
import timeseries_utils
import timeseries_dataset
from torch.utils.data import DataLoader


class LogPredictionsCallback(Callback):
    def __init__(self, wandb_logger, X_val, window_size, step_size, enc_seq_len,
                 dec_seq_len, output_sequence_length, num_samples=2048):
        '''method used to define our model parameters'''
        super().__init__()
        self.wandb_logger = wandb_logger
        self.X_val = X_val
        self.criterion = torch.nn.L1Loss(reduction="none").to('cuda')
        self.val_indices = timeseries_utils.get_indices_entire_sequence(
            data=self.X_val.numpy(), 
            window_size=window_size, 
            step_size=step_size)
        self.val_wrapper = timeseries_dataset.TransformerDataset(
            data=self.X_val,
            indices=self.val_indices,
            enc_seq_len=enc_seq_len,
            dec_seq_len=dec_seq_len,
            target_seq_len=output_sequence_length
            )
        #self.val_loader = DataLoader(val_wrapper, batch_size=20, pin_memory=True, num_workers=8)

    def topk_by_sort(input, k, axis=None, ascending=True):
        if not ascending:
            input *= -1
        ind = np.argsort(input, axis=axis)
        ind = np.take(ind, np.arange(k), axis=axis)
        if not ascending:
            input *= -1
        val = np.take_along_axis(input, ind, axis=axis) 
        return ind, val

    def on_validation_epoch_end(self, trainer, pl_module):
        #wandb.init()
        for i, batch in enumerate(self.val_wrapper):
            src, _, tgt_y = batch
            logging.info(f"tgt_y:{tgt_y}")
            if pl_module.batch_first == False:
                shape_before = src.shape
                src = src.permute(1, 0, 2)

                shape_before = tgt_y.shape
                tgt_y = tgt_y.permute(1, 0, 2)
                
            prediction = inference.run_encoder_decoder_inference(
                model=pl_module.to('cuda'), 
                src=src, 
                forecast_window=pl_module.forecast_window,
                batch_size=src.shape[1]
                ).to('cuda')
            metrics = pl_module.to('cuda').compute_loss(tgt_y, prediction)
            metrics = torch.sum(metrics)
