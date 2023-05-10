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
        src_vec = []
        tgt_y_vec = []
        for i, batch in enumerate(self.val_wrapper):
            src, _, tgt_y = batch
            src_vec.append(src)
            tgt_y_vec.append(tgt_y)
            if i > 1024:
                break
            if i % 256 == 0:
                logging.info(f"logging prediction:{i}, {i%256}, len:{len(src_vec)}")
            if len(src_vec) == 16:
                src = torch.from_numpy(np.stack(src_vec))
                tgt_y = torch.from_numpy(np.stack(tgt_y_vec))
                if pl_module.batch_first == False:
                    shape_before = src.shape
                    src = src.permute(1, 0, 2)
                    if tgt_y.dim() == 3:
                        tgt_y = tgt_y.permute(1, 0, 2)
                    else:
                        tgt_y = tgt_y.permute(1, 0)
                #logging.info(f"after src:{src.shape}")
                #logging.info(f"after tgt_y:{tgt_y.shape}")
                src = src.to('cuda')
                tgt_y = tgt_y.to('cuda')
                
                prediction = inference.run_encoder_decoder_inference(
                    model=pl_module.to('cuda'), 
                    src=src, 
                    forecast_window=pl_module.forecast_window,
                    batch_size=src.shape[1]
                    ).to('cuda')
                logging.info(f"src:{src.shape}")
                logging.info(f"tgt_y:{tgt_y.shape}")
                logging.info(f"prediction:{prediction.shape}")
                fig_cnt = src.shape[1] // 4
                for i in range(0, fig_cnt):
                    fig = plt.figure()
                    fig.set_figwidth(40)
                    for j in range(0, 4):
                        x = src[:,i*4+j,:].cpu()
                        pred = prediction[:,i*4+j,:].cpu()
                        y = tgt_y[:,i*4+j,:].cpu()
                        logging.info(f"x:{x.shape}")
                        logging.info(f"pred:{pred.shape}")
                        logging.info(f"y:{y.shape}")
                        #times = self.val_times[i*4+j].cpu()
                        #open = x[0]
                        #high = x[1]
                        #low = x[2]
                        close = x[:,3]
                        pred_close = pred[:,3]
                        #pred_close = pred
                        y_close = y[:,3]
                        #y_close = y
                        logging.info(f"close:{close}")
                        ax1 = fig.add_subplot(1, 4, j+1)
                        ax1.plot(np.arange(close.shape[0]), close, label='Training data')
                        ax1.plot(np.arange(close.shape[0]-1, close.shape[0]+pred.shape[0]), np.concatenate(([close[-1]], pred_close)), label='Prediction', color="red")
                        ax1.plot(np.arange(close.shape[0]-1, close.shape[0]+pred.shape[0]), np.concatenate(([close[-1]], y_close)), label='Groud Truth', color="purple")
                        #now = datetime.fromtimestamp(times.numpy()[-1])
                        #ax1.set_xlabel(f'{now.strftime("%y-%m-%d %H:%M")}')
                        ax1.set_ylabel('y')
                    self.wandb_logger.log_image(f"chart-{i}", images=[fig])
                        #logging.info(f"prediction:{prediction.shape}")
                        #metrics = pl_module.to('cuda').compute_loss(tgt_y, prediction)
                        #metrics = torch.sum(metrics)
                src_vec.clear()
                tgt_y_vec.clear()
