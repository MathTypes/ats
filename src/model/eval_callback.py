from typing import Any, Dict, Optional, Set
from wandb.keras import WandbEvalCallback
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

class WandbClfEvalCallback(WandbEvalCallback, Callback):
    def __init__(
        self, validloader, data_table_columns, pred_table_columns, num_samples=100
    ):
        super().__init__(data_table_columns, pred_table_columns)
        self.val_data = next(iter(validloader))

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Save checkpoint on train batch end if we meet the criteria for `every_n_train_steps`"""
        super().on_train_batch_end(batch)

    def add_ground_truth(self, logs=None):
        for idx, (x, y) in enumerate(self.val_data):
            self.data_table.add_data(
                idx,
                x["time"],
                x["close_pct"],
                #wandb.Image(image),
                #np.argmax(label, axis=-1)
                y[0]
            )

    def add_model_predictions(self, epoch, logs=None):
        # Get predictions
        preds = self._inference()
        table_idxs = self.data_table_ref.get_index()

        for idx in table_idxs:
            pred = preds[idx]
            self.pred_table.add_data(
                epoch,
                self.data_table_ref.data[idx][0],
                self.data_table_ref.data[idx][1],
                self.data_table_ref.data[idx][2],
                pred
            )

    def _inference(self):
      preds = []
      for x, y in self.val_data:
          logging.info(f"x:{x}")
          logging.info(f"y:{y}")
          pred = self.model(x)
          logging.info(f"pred:{pred}")
          argmax_pred = tf.argmax(pred, axis=-1).numpy()[0]
          preds.append(argmax_pred)

      return preds
     
