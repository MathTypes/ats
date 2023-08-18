import plotly.graph_objects as go
from pytorch_forecasting.utils import detach, to_list
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ats.model import viz_utils

def create_market_image(pred_input, pred_output):
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [
                {"secondary_y": True},
                {"secondary_y": True},
            ],
            [
                {"secondary_y": True},
                {"secondary_y": True},
            ],
        ],
    )
    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
        yaxis=dict(
            side="right",
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # hide weekends
        ],
    )
    fig.update_layout(title=pred_input.prediction_date_time, font=dict(size=20))
    y_max, y_min, y_hat_max, y_hat_min = prediction_utils.loss_stats(pred_output)
    viz_utils.add_market_viz(fig, pred_input)
    decoder_time_idx = pred_input.decoder_time_idx
    # logging.info(f"after add_market_viz")
    output_file = f"{self.image_root_path}/{decoder_time_idx}_{pred_input.prediction_date_time}_{y_max}_{y_min}_{y_hat_max}_{y_hat_min}.market.png"
    output_file = output_file.replace(" ", "_")
    try:
        fig.write_image(output_file)
        logging.info(f"generate market_image {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"can not generate {output_file}, {e}")
        return None


def create_image(self, pred_input, pred_output):
    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [
                {"secondary_y": True},
                {"secondary_y": True},
                {"secondary_y": True},
            ],
            [
                {"secondary_y": True},
                {"secondary_y": True},
                {"secondary_y": True},
            ],
        ],
    )
    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
        yaxis=dict(
            side="right",
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # hide weekends
        ],
    )
    fig.update_layout(title=pred_input.prediction_date_time, font=dict(size=20))
    # viz_utils.add_market_viz(fig, pred_input)
    viz_utils.add_model_prediction(fig, self.model, pred_input, pred_output)
    viz_utils.add_model_interpretation(fig, self.model, pred_input, pred_output)
    decoder_time_idx = pred_input.decoder_time_idx
    viz_utils.add_market_viz(fig, pred_input)
    y_max, y_min, y_hat_max, y_hat_min = prediction_utils.loss_stats(pred_output)
    output_file = f"{self.image_root_path}/{decoder_time_idx}_{pred_input.prediction_date_time}_{y_max}_{y_min}_{y_hat_max}_{y_hat_min}.png"
    output_file = output_file.replace(" ", "_")
    try:
        fig.write_image(output_file)
        logging.info(f"generate image {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"can not generate {output_file}, {e}")
        return None
