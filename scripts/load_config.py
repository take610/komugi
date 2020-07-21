import torch


import pathlib
import dataclasses
import yaml
import inspect
from configs.config_dataclass import Config


def load_config(path):
    path = pathlib.Path(path)
    config = Config.load(path)
    return config


class TrainGlobalConfig:
    id = 0
    seed = 42
    n_splits = 5
    num_workers = 4
    batch_size = 16
    n_epochs = 25
    lr = 0.001

    # ==========
    verbose = True
    verbose_step = 1
    # ==========

    # ==========
    step_scheduler = False
    validation_scheduler = True

    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode="min",
        factor=0.5,
        patience=1,
        verbose=False,
        threshold=0.0001,
        threshold_mode="abs",
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08
    )
