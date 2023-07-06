import datetime
import logging

import numpy as np
# find optimal learning rate
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet, PatchTstTransformer, PatchTstTftTransformer, PatchTftSupervised
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, MAPE, MASE, MAPCSE, RMSE, SMAPE, PoissonLoss, QuantileLoss
from timeseries_transformer import TimeSeriesTFT
from plotly.subplots import make_subplots
import wandb
import torch

from data_module import TransformerDataModule, LSTMDataModule, TimeSeriesDataModule
from models import (
    AttentionEmbeddingLSTM
)
import model_utils
from utils import Pipeline

torch.manual_seed(0)
np.random.seed(0)

target_col_name = ["OpenPct", "HighPct", "LowPct", "ClosePct", "VolumePct"]

## Params
dim_val = 512
n_heads = 8
n_decoder_layers = 4
n_encoder_layers = 4
enc_seq_len = 153 # length of input given to encoder
batch_first = False
forecast_window = 24
# Define input variables 
exogenous_vars = [] # should contain strings. Each string must correspond to a column name
input_variables = target_col_name + exogenous_vars
input_size = len(input_variables)

torch.set_float32_matmul_precision('medium')

class TFTPipeline(Pipeline):
    def __init__(self, dataset="sine_wave", device=None):
        super().__init__(device)
        self.dataset = dataset

    def create_model(self):
        self.data_module = TransformerDataModule("stock_returns",
                                                 output_sequence_length=forecast_window,
                                                 batch_size=2048)
        X_train = self.data_module.X_train
        dev = "cuda"

        logging.info(f"X_train:{X_train.shape}, dev:{dev}")
        self.model = TimeSeriesTFT(
            input_size=5,
            dim_val=16,
            dec_seq_len=enc_seq_len,
            batch_first=batch_first,
            forecast_window=forecast_window,
            num_predicted_features=1,
            device=dev
        ).float().to(dev)


class AttentionEmbeddingLSTMPipeline(Pipeline):
    def __init__(self, dataset="sine_wave", device=None):
        super().__init__(device)
        self.dataset = dataset

    def create_model(self):
        self.data_module = LSTMDataModule("stock_returns", batch_size=32, n_past=48, n_future=12)
        X_train = self.data_module.X_train
        y_train = self.data_module.y_train
        features = X_train.shape[1]
        mini_batch = X_train.shape[2]
        logging.info(f"features:{features}")
        logging.info(f"mini_batch:{mini_batch}")
        logging.info(f"y_train:{y_train.shape}")
        linear_channel = 1
        model = AttentionEmbeddingLSTM(
            input_features=features,
            linear_channel=linear_channel,
            period_channel=(mini_batch - linear_channel),
            input_size=mini_batch,
            out_size=y_train.shape[-1],
            out_values = 1,
            hidden_size=4
        )
        self.model = model.to(self.device, non_blocking=True)


class TimeSeriesPipeline(Pipeline):
    def __init__(self, dataset="fut", device=None, config=None):
        super().__init__(device)
        self.dataset = dataset
        self.config = config

    def create_model(self):
        self.heads, self.targets = model_utils.get_heads_and_targets(self.config)
        self.data_module = model_utils.get_data_module(self.config, self.targets)
        loss_per_head = model_utils.create_loss_per_head(self.heads, self.device, self.config.model.prediction_length)
        self.model = model_utils.get_nhits_model(self.config, self.data_module, loss_per_head["returns_prediction"]["loss"])
        self.model = self.model.to(self.device, non_blocking=True)

    def tune_model(self, study_name):
        #self.data_module = nhits.get_data_module(self.config)
        #self.model = nhits.get_model(self.config, self.data_module)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        #self.model = self.model.to(self.device, non_blocking=True)
        nhits.run_tune(self.config, study_name)

    def test_model(self):
        #self.data_module = nhits.get_data_module(self.config)
        #self.model = nhits.get_model(self.config, self.data_module)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        #self.model = self.model.to(self.device, non_blocking=True)
        #nhits.run_tune(config, study_name)
        pass

class TemporalFusionTransformerPipeline(Pipeline):
    def __init__(self, dataset="fut", device=None, config=None):
        super().__init__(device)
        self.dataset = dataset
        self.config = config

    def create_model(self):
        self.data_module = model_utils.get_data_module(self.config)
        self.model = model_utils.get_tft_model(self.config, self.data_module)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        self.model = self.model.to(self.device, non_blocking=True)

    def tune_model(self, study_name, config):
        #self.data_module = nhits.get_data_module(self.config)
        #self.model = nhits.get_model(self.config, self.data_module)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        #self.model = self.model.to(self.device, non_blocking=True)
        #nhits.run_tune(config, study_name)
        pass

class PatchTstTransformerPipeline(Pipeline):
    def __init__(self, dataset="fut", device=None, config=None):
        super().__init__(device)
        self.dataset = dataset
        self.config = config

    def create_model(self):
        self.data_module = model_utils.get_data_module(self.config)
        self.model = model_utils.get_patch_tst_model(self.config, self.data_module)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        self.model = self.model.to(self.device, non_blocking=True)

    def tune_model(self, study_name):
        #self.data_module = nhits.get_data_module(self.config)
        #self.model = nhits.get_model(self.config, self.data_module)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        #self.model = self.model.to(self.device, non_blocking=True)
        #nhits.run_tune(config, study_name)
        pass

class PatchTstTftPipeline(Pipeline):
    def __init__(self, dataset="fut", device=None, config=None):
        super().__init__(device)
        self.dataset = dataset
        self.config = config

    def create_model(self):
        self.data_module = model_utils.get_data_module(self.config)
        self.model = model_utils.get_patch_tst_tft_model(self.config, self.data_module)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        self.model = self.model.to(self.device, non_blocking=True)

    def tune_model(self, study_name, config):
        #self.data_module = nhits.get_data_module(self.config)
        #self.model = nhits.get_model(self.config, self.data_module)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        #self.model = self.model.to(self.device, non_blocking=True)
        #nhits.run_tune(config, study_name)
        pass
    

class PatchTftSupervisedPipeline(Pipeline):
    def __init__(self, dataset="fut", device=None, config=None):
        super().__init__(device)
        self.dataset = dataset
        self.config = config

    def create_model(self, checkpoint):
        self.heads, self.targets = model_utils.get_heads_and_targets(self.config)
        logging.info(f"head:{self.heads}, targets:{self.targets}")
        self.data_module = model_utils.get_data_module(self.config, self.targets)
        self.model = model_utils.get_patch_tft_supervised_model(self.config, self.data_module, self.heads)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        if checkpoint:
            self.model = self.model.load_from_checkpoint(checkpoint)
        self.model = self.model.to(self.device, non_blocking=True)

    def tune_model(self, study_name):
        #self.data_module = nhits.get_data_module(self.config)
        #self.model = nhits.get_model(self.config, self.data_module)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        #self.model = self.model.to(self.device, non_blocking=True)
        #nhits.run_tune(config, study_name)
        pass

    def test_model(self):
        #self.data_module = nhits.get_data_module(self.config)
        #self.model = nhits.get_model(self.config, self.data_module)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        #self.model = self.model.to(self.device, non_blocking=True)
        #nhits.run_tune(config, study_name)
        pass


    def eval_model(self):
        #self.data_module = nhits.get_data_module(self.config)
        # calcualte metric by which to display
        wandb_logger = WandbLogger(project='ATS', log_model=True)        
        trainer_kwargs = {'logger':wandb_logger}
        logging.info(f"rows:{len(self.data_module.eval_data)}")
        predictions = self.model.predict(self.data_module.val_dataloader(),
                                         mode=["raw", "prediction"],
                                         return_x=True, return_index=True,
                                         return_y=True, trainer_kwargs=trainer_kwargs)
        logging.info(f"predictions:{predictions.output.shape}, {predictions.y[0].shape}")
        logging.info(f"predictions:{predictions}")
        mean_losses = SMAPE(reduction="none")(predictions.output, predictions.y).mean(1)
        indices = mean_losses.argsort(descending=True)  # sort losses
        matched_eval_data = self.data_module.eval_data
        x = predictions.x
        y_close = predictions.y[0]
        y_close_cum_sum = torch.cumsum(y_close, dim=-1)
        logging.info(f"x:{x}")
        column_names = ["ticker", "time", "time_idx", "day_of_week", "hour_of_day", "year", "month", "day_of_month", "price_img",
                        "act_close_pct_max", "act_close_pct_min", "close_back_cumsum", "time_str",
                        "pred_time_idx", "pred_close_pct_max", "pred_close_pct_min", "img", "error_max", "error_min", "rmse", "mae"]
        data_table = wandb.Table(columns=column_names, allow_mixed_types=True)
        day_of_week_map = ["Mon", "Tue", "Wen", "Thu", "Fri", "Sat", "Sun"]
        for idx in range(10):  # plot 10 examples
            idx=indices[idx]
            context_length = len(x["encoder_target"][idx])
            prediction_length = len(x["decoder_time_idx"][idx])
            decoder_time_idx = int(x["decoder_time_idx"][idx][0].cpu().detach().numpy())
            train_data_row = matched_eval_data[matched_eval_data.time_idx == decoder_time_idx].iloc[0]
            dm = train_data_row["time"]
            dm_str = datetime.datetime.strftime(dm, "%Y%m%d-%H%M%S")
            train_data_rows = matched_eval_data[(matched_eval_data.time_idx>=decoder_time_idx-context_length) &
                                                (matched_eval_data.time_idx<decoder_time_idx+prediction_length)]
            x_time = train_data_rows["time"]
            #logging.info(f"xtime:{x_time}")
            fig = make_subplots(rows=2, cols=3, specs=[[{"secondary_y": True}, {"secondary_y": True},
                                                        {"secondary_y": True}],
                                                       [{"secondary_y": True}, {"secondary_y": True},
                                                        {"secondary_y": True}]], )
            fig.update_layout(autosize=False, width=1500, height=800,)
            fig.update_xaxes(
                rangebreaks=[
                    dict(bounds=["sat", "mon"]), #hide weekends
                    dict(bounds=[17, 4], pattern="hour"), #hide hours outside of 4am-5pm
                ],
            )
            prediction_date_time = train_data_row["ticker"] + " " + dm_str + " " + day_of_week_map[train_data_row["day_of_week"]]
            fig.update_layout(title=prediction_date_time, font=dict(size=20))
            self.model.plot_prediction(
                predictions.x,
                predictions.output,
                idx=idx,
                add_loss_to_title=SMAPE(quantiles=self.model.loss.quantiles),
                ax=fig, row=1, col=1, draw_mode="pred_cum", x_time=x_time
            )

            img_bytes = fig.to_image(format="png") # kaleido library
            im = PIL.Image.open(BytesIO(img_bytes))
            pred_img = wandb.Image(im)
            
            y_hat = y_hats[idx]
            y_hat_cum = y_hats_cum[idx]
            img = wandb.Image(im)
            fig = go.Figure(data=go.Ohlc(x=train_data_rows['time'],
                    open=train_data_rows['open'],
                    high=train_data_rows['high'],
                    low=train_data_rows['low'],
                    close=train_data_rows['close']))
            # add a bar at prediction time
            fig.update(layout_xaxis_rangeslider_visible=False)
            prediction_date_time = train_data_row["ticker"] + " " + dm_str + " " + day_of_week_map[train_data_row["day_of_week"]]
            fig.update_layout(title=prediction_date_time, font=dict(size=20))
            fig.update_xaxes(
                rangebreaks=[
                      dict(bounds=["sat", "mon"]), #hide weekends
                      dict(bounds=[17, 4], pattern="hour"), #hide hours outside of 4am-5pm
                  ],
            )
            img_bytes = fig.to_image(format="png") # kaleido library
            im = PIL.Image.open(BytesIO(img_bytes))
            img = wandb.Image(im)
            if torch.max(y_close_cum_sum_row) > 0.10:
                logging.info(f"bad row:{train_data_row}, idx:{idx}, index:{index}")
                logging.info(f"y_close_cum_sum_row:{y_close_cum_sum_row}")
                logging.info(f"y_close:{y_close[idx]}")
                logging.info(f"train_data_rows:{train_data_rows[-32:]}")
            base = 0
            data_table.add_data(
                train_data_row["ticker"], # 0 ticker
                dm, # 1 time
                train_data_row["time_idx"], # 2 time_idx
                train_data_row["day_of_week"], # 3 day of week
                train_data_row["hour_of_day"], # 4 hour of day
                train_data_row["year"], # 5 year
                train_data_row["month"], # 6 month
                train_data_row["day_of_month"], # 7 day_of_month
                wandb.Image(im),  # 8 image
                #np.argmax(label, axis=-1)
                torch.max(y_close_cum_sum_row) - base, # 9 max
                torch.min(y_close_cum_sum_row) - base, # 10 min
                base, # 11 close_back_cusum
                dm_str, # 12
                decoder_time_idx, torch.max(y_hat_cum), torch.min(y_hat_cum), pred_img, 0, 0
              )
        data_artifact = wandb.Artifact(f"run_{wandb.run.id}_pred", type="evaluation")
        data_artifact.add(data_table, "eval_data")

        # Calling `use_artifact` uploads the data to W&B.
        assert wandb.run is not None
        wandb.run.use_artifact(data_artifact)
        data_artifact.wait()
        
