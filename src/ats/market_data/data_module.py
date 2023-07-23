# import pytorch_lightning as pl
import lightning.pytorch as pl
import torch
import logging
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from pytorch_forecasting.data.encoders import (
    NaNLabelEncoder,
)
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder

from ats.market_data.datasets import (
    generate_stock_tokens,
    generate_stock_returns,
)
from ats.market_data import timeseries_dataset
from ats.market_data import timeseries_utils

eval_batch_size = 10

class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(
        self, config, train_data, eval_data, test_data, target, simulation_mode=False
    ):
        super().__init__()
        logging.info(f"train_data:{train_data.describe()}")
        train_data = train_data.fillna(-1)
        eval_data = eval_data.fillna(-1)
        self.train_data = train_data.dropna()
        self.eval_data = eval_data.dropna()
        self.test_data = test_data

        if self.train_data.empty:
            # TODO: get rid of the hack to fake train data during
            # eval and build_search mode. In both case, we do
            # not have train data and do not want data_module
            # to cry
            self.train_data = self.eval_data
        logging.info(f"train_data:{train_data.describe()}")
        logging.info(f"train_data:{train_data.iloc[-5:]}")
        logging.info(f"target:{target} {type(target)}")
        context_length = config.model.context_length
        prediction_length = config.model.prediction_length
        # target_normalizer = None
        target_normalizer = "auto"
        # target_normalizer=GroupNormalizer(
        #    groups=["ticker"], transformation="softplus"
        # ),  # use softplus and normalize by group
        # if isinstance(target, (typing.Set, typing.List)):
        #    normalizer_list = [
        #        EncoderNormalizer(transformation="relu") for i in range(len(target))
        #    ]
        #    target_normalizer = MultiNormalizer(normalizer_list)
        time_varying_known_reals = config.features.time_varying_known_reals
        if OmegaConf.is_list(time_varying_known_reals):
            time_varying_known_reals = OmegaConf.to_object(time_varying_known_reals)
        time_varying_unknown_reals = config.features.time_varying_unknown_reals
        if OmegaConf.is_list(time_varying_unknown_reals):
            time_varying_unknown_reals = OmegaConf.to_object(time_varying_unknown_reals)
        logging.info(f"train_data:{len(self.train_data)}")
        self.training = TimeSeriesDataSet(
            self.train_data,
            time_idx="time_idx",
            target=target,
            group_ids=["ticker"],
            min_encoder_length=context_length,  # keep encoder length long (as it is in the validation set)
            max_encoder_length=context_length,
            min_prediction_length=prediction_length,
            max_prediction_length=prediction_length,
            allow_missing_timesteps=True,
            target_normalizer=target_normalizer,
            lags=config.features.lags,
            static_categoricals=config.features.static_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            categorical_encoders={
                "ticker": NaNLabelEncoder().fit(self.train_data.ticker)
            },
            # categorical_encoders={"ticker": GroupNormalizer().fit(self.train_data.ticker)},
            add_relative_time_idx=config.features.add_relative_time_idx,
        )
        # create dataloaders for model
        self.batch_size = config.model.train_batch_size  # set this between 32 to 128
        # Need batch_size 1 to get example level metrics
        self.eval_batch_size = config.model.eval_batch_size
        # create validation set (predict=True) which means to predict the last max_prediction_length points in time
        # for each series
        eval_data_size = (
            int(len(self.eval_data) / self.eval_batch_size) * self.eval_batch_size
        )
        self.eval_data = self.eval_data[:eval_data_size]
        logging.info(f"eval_data_size:{len(self.eval_data)}")
        self.validation = TimeSeriesDataSet.from_dataset(self.training, self.eval_data)
        logging.info(f"test_data_size:{len(self.test_data)}")
        self.test = TimeSeriesDataSet.from_dataset(
            self.training, self.test_data, simulation_mode=simulation_mode
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        train_dataloader = self.training.to_dataloader(
            train=True,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=True,
            drop_last=True,
        )
        return train_dataloader

    def val_dataloader(self):
        # logging.info(f"val_dataloader_batch:{self.eval_batch_size}")
        # train = True is the hack to randomly sample from time series from different ticker.
        logging.info(f"eval_batch_size:{self.eval_batch_size}")
        val_dataloader = self.validation.to_dataloader(
            train=True,
            batch_size=self.eval_batch_size,
            num_workers=20,
            # batch_sampler="synchronized",
            pin_memory=True,
            drop_last=False,
        )
        return val_dataloader

    def test_dataloader(self):
        # Here use same validation as we use simulation for test.
        test_dataloader = self.validation.to_dataloader(
            train=False,
            batch_size=self.eval_batch_size,
            num_workers=4,
            batch_sampler=None,
            pin_memory=True,
            drop_last=True,
        )
        return test_dataloader
