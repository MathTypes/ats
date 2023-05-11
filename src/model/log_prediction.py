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
            data=self.X_val[-1024:].numpy(), 
            window_size=window_size, 
            step_size=step_size)
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.output_sequence_length = output_sequence_length
        #self.val_loader = DataLoader(val_wrapper, batch_size=20, pin_memory=True, num_workers=8)

    def topk_by_sort(self, input, k, axis=None, ascending=True):
        if not ascending:
            input *= -1
        ind = np.argsort(input, axis=axis)
        ind = np.take(ind, np.arange(k), axis=axis)
        if not ascending:
            input *= -1
        val = np.take_along_axis(input, ind, axis=axis) 
        return ind, val

    def compute_loss(self, y_hat, y):
        loss = self.criterion(y_hat, y)
        return loss

    def on_validation_epoch_end(self, trainer, pl_module):
        src_vec = []
        tgt_y_vec = []
        time_vec = []
        self.val_wrapper = timeseries_dataset.TransformerDataset(
            data=self.X_val[-1024:],
            indices=self.val_indices,
            enc_seq_len=self.enc_seq_len,
            dec_seq_len=self.dec_seq_len,
            target_seq_len=self.output_sequence_length
            )
        for i, batch in enumerate(self.val_wrapper):
            src, _, tgt_y = batch
            time = src[:,5]
            src = src[:,:5]
            tgt_y = tgt_y[:,:5]
            src_vec.append(src)
            time_vec.append(time)
            tgt_y_vec.append(tgt_y)
            if i > 256:
                break
            if i % 16 == 0:
                logging.info(f"logging prediction:{i}, {i%16}, len:{len(src_vec)}")
            if len(src_vec) == 64:
                src = torch.from_numpy(np.stack(src_vec))
                tgt_y = torch.from_numpy(np.stack(tgt_y_vec))
                times = torch.from_numpy(np.stack(time_vec)).squeeze(-1)
                if pl_module.batch_first == False:
                    shape_before = src.shape
                    src = src.permute(1, 0, 2)
                    times = times.permute(1, 0)
                    if tgt_y.dim() == 3:
                        tgt_y = tgt_y.permute(1, 0, 2)
                    else:
                        tgt_y = tgt_y.permute(1, 0)
                src = src.to('cuda')
                tgt_y = tgt_y[:,:,3].to('cuda')
                
                prediction = inference.run_encoder_decoder_inference(
                    model=pl_module.to('cuda'), 
                    src=src, 
                    forecast_window=pl_module.forecast_window,
                    batch_size=src.shape[1]
                    ).squeeze().to('cuda')
                
                logging.info(f"prediction:{prediction.shape}")
                logging.info(f"tgt_y:{tgt_y.shape}")
                loss = self.compute_loss(prediction, tgt_y)
                logging.info(f"loss:{loss.shape}")
                top_ind, top_loss = self.topk_by_sort(loss.to("cpu").numpy(), 10)
                for ind in top_ind:
                    fig = plt.figure()
                    fig.set_figwidth(40)
                    x = src[:,ind,:].cpu()
                    pred = prediction[:,ind,:].cpu()
                    y = tgt_y[:,ind,:].cpu()
                    time = times[:ind].cpu()
                    logging.info(f"time:{time.shape}")
                    close = x[:,3]
                    pred_close = pred[:].squeeze(-1)
                    y_close = y[:,3].squeeze(-1)
                    ax1 = fig.add_subplot(1, 1, 1)
                    ax1.plot(np.arange(close.shape[0]), close, label='Training data')
                    ax1.plot(np.arange(close.shape[0]-1, close.shape[0]+pred.shape[0]), np.concatenate(([close[-1]], pred_close)), label='Prediction', color="red")
                    ax1.plot(np.arange(close.shape[0]-1, close.shape[0]+pred.shape[0]), np.concatenate(([close[-1]], y_close)), label='Groud Truth', color="purple")
                    now = datetime.fromtimestamp(int(time.numpy()[-1])).strftime("%y-%m-%d %H:%M")
                    ax1.set_xlabel(f'{now}')
                    ax1.set_ylabel('y')
                    self.wandb_logger.log_image(f"chart-{now}", images=[fig])
                src_vec.clear()
                tgt_y_vec.clear()
