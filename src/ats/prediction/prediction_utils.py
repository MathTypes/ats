import logging

from pytorch_forecasting.utils import create_mask, detach, to_list


def predict(model, new_prediction_data, wandb_logger):
    # logging.info(f"new_prediction_data:{new_prediction_data}")
    #logging.info(f"index:{train_dataset.index.iloc[-5:]}")
    trainer_kwargs = {"logger": wandb_logger}
    # logging.info(f"trainer_kwargs:{trainer_kwargs}")
    new_raw_predictions = model.predict(
        new_prediction_data,
        mode="raw",
        return_x=True,
        batch_size=1,
        trainer_kwargs=trainer_kwargs,
    )
    if isinstance(new_raw_predictions, (list)) and len(new_raw_predictions) < 1:
        logging.info(f"no prediction")
        return None, None
    prediction_kwargs = {}
    y_hats = to_list(
        model.to_prediction(new_raw_predictions.output, **prediction_kwargs)
    )
    prediction, position = y_hats
    logging.info(f"prediction:{prediction}")
    y_quantiles = to_list(
        model.to_quantiles(new_raw_predictions.output, **quantiles_kwargs)
    )[0]
    logging.info(f"y_quantiles:{y_quantiles}")
    # logging.info(f"y_hats:{y_hats}")
    # logging.info(f"y:{new_raw_predictions.y}")
    return prediction, y_quantiles
