from __future__ import annotations

import argparse
from argparse import Namespace
import logging
import os
from pathlib import Path
from pprint import pprint
from typing import Any

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig
import pytorch_metric_learning.utils.logging_presets as logging_presets
import torch
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from ats.app.env_mgr import EnvMgr
from ats.market_data.market_data_mgr import MarketDataMgr

from vss.metrics.consts import (
    DATALOADER_WORKERS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATALOADER_NUM_WORKERS,
    DEFAULT_EMBEDDER_LAYERS,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_SAMPLER_M,
    DEFAULT_SPLIT,
    DEFAULT_WEIGHT_DECAY,
    SIZE,
)
from vss.metrics.nets import get_trunk_embedder
from vss.metrics.utils import (
    DatasetCombined,
    get_transformation_with_size,
    save_training_meta,
)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> Any:
    logging.info(f"Current working directory : {os.getcwd()}")
    logging.info(f"Orig working directory    : {get_original_cwd()}")
    logs_dir = Path(cfg.job.log_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()
    run_id = None
    env_mgr = EnvMgr(cfg, run_id)
    md_mgr = MarketDataMgr(env_mgr)
    # get neural net models and custom dataset for easier manipulation during training setup
    transformation = get_transformation_with_size(cfg.job.input_size)
    dataset = DatasetCombined.get_dataset(
        md_mgr,
        cfg.job.search_metric_meta,
        cfg.job.search_metric_data_dir,
        split=DEFAULT_SPLIT,
        transformation={"train": transformation, "test": transformation},
    )
    trunk, embedder = get_trunk_embedder(md_mgr,
                                         #cfg, cfg.job.search_metric_trunk_model,
                                         DEFAULT_EMBEDDER_LAYERS)

    # Set optimizers
    trunk_optimizer = torch.optim.Adam(
        trunk.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY
    )
    embedder_optimizer = torch.optim.Adam(
        embedder.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY
    )

    # TODO: add feature of choosing those params from cli/config
    # set pml specific losses, miners, samplers
    loss = losses.CircleLoss()
    miner = miners.MultiSimilarityMiner(epsilon=0.1)
    sampler = samplers.MPerClassSampler(
        dataset.y_train,
        m=DEFAULT_SAMPLER_M,
        length_before_new_iter=len(dataset.train_dataset),
    )

    # package above stuff into dictionaries compatible with pml
    models = {"trunk": trunk, "embedder": embedder}
    optimizers = {
        "trunk_optimizer": trunk_optimizer,
        "embedder_optimizer": embedder_optimizer,
    }
    loss_funcs = {"metric_loss": loss}
    mining_funcs = {"tuple_miner": miner}
    dataset_dict = {"val": dataset.test_dataset}

    # create hooks for training
    model_dir = str((logs_dir / "models").absolute())
    csv_dir = str((logs_dir / "logs").absolute())
    tensorboard_dir = str((logs_dir / "training_logs").absolute())

    # set up hooks for saving models and logging
    record_keeper, _, _ = logging_presets.get_record_keeper(csv_dir, tensorboard_dir)
    hooks = logging_presets.get_hook_container(record_keeper)

    # Create the tester and add it to end of epoch hook
    collate_fn = None
    if cfg.job.search_metric_trunk_model in ["patch_tft"]:
        collate_fn=dataset.train_dataset._collate_fn
    tester = testers.GlobalEmbeddingSpaceTester(
        end_of_testing_hook=hooks.end_of_testing_hook,
        # visualizer=umap.UMAP(),
        # visualizer_hook=visualizer_hook,
        dataloader_num_workers=DATALOADER_WORKERS,
        accuracy_calculator=AccuracyCalculator(k="max_bin_count"),
    )
    end_of_epoch_hook = hooks.end_of_epoch_hook(tester, dataset_dict, model_dir, test_collate_fn=collate_fn)

    trainer = trainers.MetricLossOnly(
        models,
        optimizers,
        DEFAULT_BATCH_SIZE,
        loss_funcs,
        mining_funcs,
        dataset.train_dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        dataloader_num_workers=DEFAULT_DATALOADER_NUM_WORKERS,
        end_of_iteration_hook=hooks.end_of_iteration_hook,
        end_of_epoch_hook=end_of_epoch_hook,
    )
    pprint(vars(cfg))
    save_training_meta(cfg, path=logs_dir)
    trainer.train(num_epochs=cfg.job.epochs)


if __name__ == "__main__":
    main()
