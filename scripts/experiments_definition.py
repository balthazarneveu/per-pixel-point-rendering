from pixr.properties import (NB_EPOCHS, TRAIN, VALIDATION, SCHEDULER, REDUCELRONPLATEAU,
                             MODEL, ARCHITECTURE, ID, NAME, SCHEDULER_CONFIGURATION, OPTIMIZER, PARAMS, LR,
                             LOSS, LOSS_MSE, DATALOADER, BATCH_SIZE, SCENE, NB_POINTS, SEED, PSEUDO_COLOR_DIMENSION, SCALE_LIST)
from pixr.synthesis.world_simulation import STAIRCASE
from typing import List


def model_configurations(config, model_preset="UNet") -> dict:
    if model_preset == "UNet":
        config[MODEL] = {
            ARCHITECTURE: dict(
                in_channels=config[PSEUDO_COLOR_DIMENSION],
                width=64,
                enc_blk_nums=[1, 1, 1, 28],
                middle_blk_num=1,
                dec_blk_nums=[1, 1, 1, 1],
            ),
            NAME: model_preset
        }
    elif model_preset == "Bypass":
        config[MODEL] = {
            ARCHITECTURE: dict(
                in_channels=config[PSEUDO_COLOR_DIMENSION],
                n_scales=len(config[SCALE_LIST])
            ),
            NAME: model_preset
        }
    else:
        raise ValueError(f"Unknown model preset {model_preset}")


def presets_experiments(
    exp: int,
    b: int = 16,
    n: int = 50,
    model_preset: str = "UNet",
    scene: str = STAIRCASE,
    nb_points: int = 20000,
    seed: int = 42,
    pseudo_color_dimension: int = 3,
    scale_list: List[int] = [0, 1, 2, 3],
    lr: float = 1e-3
) -> dict:
    config = {
        ID: exp,
        NAME: f"{exp:04d}",
        NB_EPOCHS: n
    }
    config[OPTIMIZER] = {
        NAME: "Adam",
        PARAMS: {
            LR: lr
        }
    }
    config[DATALOADER] = {
        BATCH_SIZE: {
            TRAIN: b,
            VALIDATION: b
        },
    }
    config[PSEUDO_COLOR_DIMENSION] = pseudo_color_dimension
    config[SCALE_LIST] = scale_list
    model_configurations(config, model_preset=model_preset)
    config[SCHEDULER] = REDUCELRONPLATEAU
    config[SCHEDULER_CONFIGURATION] = {
        "factor": 0.8,
        "patience": 5
    }
    config[LOSS] = LOSS_MSE
    config[SCENE] = scene
    config[NB_POINTS] = nb_points
    config[SEED] = seed
    return config


def get_experiment_from_id(exp: int):
    if exp == 1:
        return presets_experiments(exp, b=4, n=200, model_preset="UNet", scene=STAIRCASE)
    elif exp == 2:
        return presets_experiments(exp, b=4, n=200, model_preset="UNet", scene=STAIRCASE, pseudo_color_dimension=8)
    elif exp == 3:
        return presets_experiments(exp, b=4, n=50, model_preset="Bypass", scene=STAIRCASE, pseudo_color_dimension=5, lr=0.3)
    elif exp == 4:
        return presets_experiments(exp, b=4, n=200, model_preset="Bypass", scene=STAIRCASE, pseudo_color_dimension=3, lr=0.3)
    elif exp == 5:
        return presets_experiments(exp, b=4, n=50, model_preset="UNet", scene=STAIRCASE, pseudo_color_dimension=3, lr=0.01)
    else:
        raise NameError(f"Experiment {exp} not found!")
