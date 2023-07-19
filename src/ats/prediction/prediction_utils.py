import logging

from pytorch_forecasting.utils import create_mask, detach, to_list


def predict(model, new_prediction_data, wandb_logger):
    # logging.info(f"new_prediction_data:{new_prediction_data}")
    #logging.info(f"index:{train_dataset.index.iloc[-5:]}")
    trainer_kwargs = {"logger": wandb_logger}
    # logging.info(f"trainer_kwargs:{trainer_kwargs}")
    model = model.to("cuda:0")
    device = model.device
    new_raw_predictions = model.predict(
        new_prediction_data,
        mode="raw",
        return_x=True,
        batch_size=1,
        trainer_kwargs=trainer_kwargs,
    )
    logging.info(f"new_raw_predictions:{new_raw_predictions}")
    if isinstance(new_raw_predictions, (list)) and len(new_raw_predictions) < 1:
        logging.info(f"no prediction")
        return None, None
    prediction_kwargs = {}
    output = new_raw_predictions.output
    output = {
        key: [v.to(device) for v in val]
        if isinstance(val, list)
        else val.to(device)
        for key, val in output.items()
    }
    y_hats = to_list(
        model.to_prediction(output, **prediction_kwargs)
    )
    prediction, position = y_hats
    logging.info(f"prediction:{prediction}")
    y_quantiles = to_list(
        model.to_quantiles(output, **quantiles_kwargs)
    )[0]
    del output
    del new_raw_predictions
    logging.info(f"y_quantiles:{y_quantiles}")
    # logging.info(f"y_hats:{y_hats}")
    # logging.info(f"y:{new_raw_predictions.y}")
    return prediction, y_quantiles
