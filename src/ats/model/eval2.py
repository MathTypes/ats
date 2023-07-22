import logging
import warnings

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

import warnings

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from pytorch_forecasting.metrics import MultivariateNormalDistributionLoss
from pytorch_forecasting.data import GroupNormalizer

from util import logging_utils

from math import ceil


def week_of_month(dt):
    """Returns the week of the month for the specified date."""
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()
    return int(ceil(adjusted_dom / 7.0))


if __name__ == "__main__":
    logging_utils.init_logging()

    data = pd.read_parquet("data/token/FUT/30min/ES", engine="fastparquet")
    data["Time"] = data.index
    data["ticker"] = "ES"
    data["volume"] = data["Volume"]
    data["close"] = data["Close"]
    data["Time"] = data["Time"].apply(lambda x: x.timestamp()).astype(np.float32)
    logging.info(f"data:{data.head()}")

    data["date"] = data.index
    # add time index
    data.insert(0, "time_idx", range(0, len(data)))
    # data["time_idx"] = data['date'].apply(lambda x:int(x.timestamp()))
    # data["time_idx"] -= data["time_idx"].min()

    # add additional features
    data["month"] = data.date.dt.month.astype(str).astype(
        "category"
    )  # categories have be strings
    data["series"] = data.apply(lambda x: x.ticker + "_" + x.month, axis=1)
    data["log_volume"] = np.log(data.volume + 1e-8)
    data["avg_volume_by_ticker"] = data.groupby(
        ["time_idx", "ticker"], observed=True
    ).volume.transform("mean")
    data["hour_of_day"] = (
        data["date"].apply(lambda x: x.hour).astype(str).astype("category")
    )
    data["day_of_week"] = data.index.dayofweek.astype(str).astype("category")
    data["day_of_month"] = data.index.day.astype(str).astype("category")
    data["week_of_month"] = (
        data["date"].apply(week_of_month).astype(str).astype("category")
    )
    data["week_of_year"] = data.index.isocalendar().week.astype(str).astype("category")
    logging.info(f"data:{data.head()}")
    logging.info(f"data:{data.describe()}")

    # we want to encode special days as one variable and thus need to first reverse one-hot encoding
    special_days = [
        "easter_day",
        "good_friday",
        "new_year",
        "christmas",
        "labor_day",
        "independence_day",
        "revolution_day_memorial",
        "regional_games",
        "fifa_u_17_world_cup",
        "football_gold_cup",
        "beer_capital",
        "music_fest",
    ]
    # data[special_days] = data[special_days].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")
    data.sample(10, random_state=521)

    max_encoder_length = 60
    max_prediction_length = 20
    val_idx = max(int(len(data) * 0.7), len(data) - 2048 * 16)
    tst_idx = max(int(len(data) * 0.8), len(data) - 2048)
    training_cutoff = val_idx
    train_data = data[:val_idx]
    context_length = max_encoder_length
    prediction_length = max_prediction_length

    logging.info(f"train_data:{train_data.head()}, training_cutoff={training_cutoff}")
    training = TimeSeriesDataSet(
        train_data,
        time_idx="time_idx",
        target="close",
        # categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
        group_ids=["series"],
        static_categoricals=["ticker"],
        static_reals=[],
        allow_missing_timesteps=True,
        variable_groups={},  # group of categorical variables can be treated as one variable
        time_varying_known_categoricals=[
            "month",
            "hour_of_day",
            "day_of_week",
            "week_of_month",
        ],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["close"],
        max_encoder_length=context_length,
        max_prediction_length=prediction_length,
        target_normalizer=GroupNormalizer(
            groups=["series"], transformation="softplus"
        ),  # use softplus and normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    validation = TimeSeriesDataSet.from_dataset(
        training, data, predict=True, stop_randomization=True
    )

    # create dataloaders for model
    batch_size = 128  # set this between 32 to 128
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
    )

    # configure network and trainer
    pl.seed_everything(42)

    net = DeepAR.from_dataset(
        training,
        learning_rate=3e-2,
        log_interval=10,
        log_val_interval=1,
        hidden_size=30,
        rnn_layers=2,
        loss=MultivariateNormalDistributionLoss(rank=30),
        optimizer="Adam",
    )

    # load the best model according to the validation loss
    # (given that we use early stopping, this is not necessarily the last epoch)
    best_model_path = (
        "lightning_logs/lightning_logs/version_5/checkpoints/epoch=17-step=900.ckpt"
    )
    best_model = DeepAR.load_from_checkpoint(best_model_path)
    logging.info(f"best_model_path:{best_model_path}")

    # calcualte mean absolute error on validation set
    # predictions = best_model.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu"))
    # MAE()(predictions.output, predictions.y)

    # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
    raw_predictions = net.predict(
        val_dataloader,
        mode="raw",
        return_x=True,
        n_samples=100,
        trainer_kwargs=dict(accelerator="cpu"),
    )

    # ticker = validation.x_to_index(raw_predictions.x)["ticker"]
    logging.info(f"raw_predictions.x:{raw_predictions.x}")
    logging.info(f"raw_predictions.output:{raw_predictions.output}")
    for idx in range(10):  # plot 10 examples
        fig = best_model.plot_prediction(
            raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True
        )
        print(f"fig:{fig}")
        filename = "/tmp/file.png"
        fig.savefig(filename)
        img = mpimg.imread(filename)
        # plt.imshow()
        imgplot = plt.imshow(img)
        # plt.suptitle(f"ticker: {ticker.iloc[idx]}")
        plt.show()
