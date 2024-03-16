from pixr.properties import (NB_EPOCHS, TRAIN, VALIDATION, SCHEDULER, REDUCELRONPLATEAU,
                             MODEL, ARCHITECTURE, ID, NAME, SCHEDULER_CONFIGURATION, OPTIMIZER, PARAMS, LR,
                             LOSS, LOSS_MSE, DATALOADER, BATCH_SIZE)
from typing import Tuple


def model_configurations(config, model_preset="UNet") -> dict:
    if model_preset == "UNet":
        config[MODEL] = {
            ARCHITECTURE: dict(
                width=64,
                enc_blk_nums=[1, 1, 1, 28],
                middle_blk_num=1,
                dec_blk_nums=[1, 1, 1, 1],
            ),
            NAME: model_preset
        }
    else:
        raise ValueError(f"Unknown model preset {model_preset}")


def presets_experiments(
    exp: int,
    b: int = 16,
    n: int = 50,
    model_preset: str = "UNet"
) -> dict:
    config = {
        ID: exp,
        NAME: f"{exp:04d}",
        NB_EPOCHS: n
    }
    config[OPTIMIZER] = {
        NAME: "Adam",
        PARAMS: {
            LR: 1e-3
        }
    }
    config[DATALOADER] = {
        BATCH_SIZE: {
            TRAIN: b,
            VALIDATION: b
        },
    }
    model_configurations(config, model_preset=model_preset)
    config[SCHEDULER] = REDUCELRONPLATEAU
    config[SCHEDULER_CONFIGURATION] = {
        "factor": 0.8,
        "patience": 5
    }
    config[LOSS] = LOSS_MSE
    return config


def get_experiment_from_id(exp: int):
    if exp == 1:
        return presets_experiments(exp, b=4, n=200, model_preset="UNet")
