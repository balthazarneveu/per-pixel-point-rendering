from pixr.properties import (NB_EPOCHS, TRAIN, VALIDATION, SCHEDULER, REDUCELRONPLATEAU,
                             MODEL, ARCHITECTURE, ID, NAME, SCHEDULER_CONFIGURATION, OPTIMIZER, PARAMS, LR,
                             LOSS, LOSS_MSE, DATALOADER, BATCH_SIZE, SCENE, NB_POINTS, SEED, PSEUDO_COLOR_DIMENSION,
                             SCALE_LIST, RATIO_TRAIN, MULTISCALE_SUPERVISION)
from pixr.synthesis.world_simulation import STAIRCASE
from typing import List


def model_configurations(config, model_preset="UNet", k_size=1, depth=2, h_dim=8) -> dict:
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
    elif model_preset == "StackedConvolutions":
        config[MODEL] = {
            ARCHITECTURE: dict(
                in_channels=config[PSEUDO_COLOR_DIMENSION],
                n_scales=len(config[SCALE_LIST]),
                k_size=k_size,
                depth=depth,
                h_dim=h_dim
            ),
            NAME: model_preset
        }
    elif model_preset == "Bypass":
        config[MODEL] = {
            ARCHITECTURE: dict(
                in_channels=config[PSEUDO_COLOR_DIMENSION],
                n_scales=len(config[SCALE_LIST]),
                k_size=k_size
            ),
            NAME: model_preset
        }
    elif model_preset == "TrueBypass":
        assert config[PSEUDO_COLOR_DIMENSION] == 3, f"TrueBypass requires in_channels == out_channels"
        config[MODEL] = {
            ARCHITECTURE: dict(
                in_channels=config[PSEUDO_COLOR_DIMENSION],
                n_scales=len(config[SCALE_LIST]),
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
    lr: float = 1e-3,
    k_size: int = 3,
    ratio_train: float = 0.8,
    depth: int = 2,
    h_dim: int = 8,
    ms_supervision: bool = True
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
    config[MULTISCALE_SUPERVISION] = ms_supervision
    config[RATIO_TRAIN] = ratio_train
    config[PSEUDO_COLOR_DIMENSION] = pseudo_color_dimension
    config[SCALE_LIST] = scale_list
    model_configurations(config, model_preset=model_preset, k_size=k_size, depth=depth, h_dim=h_dim)
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
    if exp == 0:
        conf = presets_experiments(exp, b=4, n=100, model_preset="TrueBypass",
                                   scene=STAIRCASE, pseudo_color_dimension=3, lr=0.3, k_size=1)
    if exp == 1:
        conf = presets_experiments(exp, b=32, n=100, model_preset="TrueBypass",
                                   scene=STAIRCASE, pseudo_color_dimension=3, lr=0.3, k_size=1)
    if exp == 2:
        conf = presets_experiments(exp, b=32, n=100, model_preset="TrueBypass",
                                   scene=STAIRCASE, pseudo_color_dimension=3, lr=0.3, k_size=1, ratio_train=0.98)  # Chekc
    if exp == 3:
        conf = presets_experiments(exp, b=32, n=100, model_preset="TrueBypass",
                                   scene="volleyball", pseudo_color_dimension=3, lr=0.3, k_size=1, ratio_train=0.98)  # Backface culling disabled!
    if exp == 4:
        conf = presets_experiments(exp, b=32, n=100, model_preset="TrueBypass",
                                   scene="volleyball", pseudo_color_dimension=3, lr=0.3, k_size=1, ratio_train=0.98)  # Optimize  with backface culling
    if exp == 5:
        conf = presets_experiments(exp, b=32, n=100, model_preset="TrueBypass",
                                   scene="volleyball", pseudo_color_dimension=3, lr=0.3, k_size=1, ratio_train=0.98)  # Optimize  with backface culling + 12 views!
    if exp == 6:
        conf = presets_experiments(exp, b=32, n=100, model_preset="StackedConvolutions",
                                   scene="volleyball", pseudo_color_dimension=3, lr=0.001, k_size=5, ratio_train=0.98)
    if exp == 7:
        conf = presets_experiments(exp, b=32, n=100, model_preset="Bypass",
                                   scene="volleyball", pseudo_color_dimension=3, lr=0.01, k_size=5, ratio_train=0.98)

    if exp == 10:
        conf = presets_experiments(exp, b=32, n=100, model_preset="TrueBypass",
                                   scene="volleyball", pseudo_color_dimension=3, lr=0.3, k_size=1, ratio_train=0.98)
    elif exp == 11:
        conf = presets_experiments(exp, b=32, n=100, model_preset="StackedConvolutions",
                                   scene="volleyball", pseudo_color_dimension=3, lr=0.1, k_size=5, ratio_train=0.98)
    elif exp == 12:  # 29.5db
        conf = presets_experiments(exp, b=16, n=100, model_preset="StackedConvolutions",
                                   scene="volleyball", pseudo_color_dimension=8, lr=0.01, k_size=5, ratio_train=0.98,
                                   depth=2, h_dim=8)
    elif exp == 13:  # 29.2db
        conf = presets_experiments(exp, b=8, n=100, model_preset="StackedConvolutions",
                                   scene="volleyball", pseudo_color_dimension=8, lr=0.01, k_size=5, ratio_train=0.98,
                                   depth=4, h_dim=8)
    elif exp == 14:  # 31.5db
        conf = presets_experiments(exp, b=8, n=100, model_preset="StackedConvolutions",
                                   scene="volleyball", pseudo_color_dimension=8, lr=0.001, k_size=5, ratio_train=0.98,
                                   depth=4, h_dim=8)
    elif exp == 15:  # 31.6db
        conf = presets_experiments(exp, b=8, n=400, model_preset="StackedConvolutions",
                                   scene="volleyball", pseudo_color_dimension=8, lr=0.005, k_size=5, ratio_train=0.98,
                                   depth=4, h_dim=8)
    # CHAIR IS TOO SMALL - alpha used in depth test probably hinders the learning process
    elif exp == 20:
        conf = presets_experiments(exp, b=32, n=100, model_preset="TrueBypass",
                                   scene="chair", pseudo_color_dimension=3, lr=0.3, k_size=1, ratio_train=0.98)
    elif exp == 21:
        conf = presets_experiments(exp, b=8, n=100, model_preset="StackedConvolutions",
                                   scene="chair", pseudo_color_dimension=8, lr=0.005, k_size=5, ratio_train=0.98,
                                   depth=4, h_dim=8)
    elif exp == 30:  # 20db
        conf = presets_experiments(exp, b=32, n=100, model_preset="TrueBypass",
                                   scene="material_balls", pseudo_color_dimension=3, lr=0.3, k_size=1, ratio_train=0.98)
    elif exp == 31:
        conf = presets_experiments(exp, b=8, n=100, model_preset="StackedConvolutions",
                                   scene="material_balls", pseudo_color_dimension=8, lr=0.005, k_size=5, ratio_train=0.98,
                                   depth=4, h_dim=8)
    elif exp == 32:
        conf = presets_experiments(exp, b=4, n=2000, model_preset="StackedConvolutions",
                                   scene="material_balls", pseudo_color_dimension=8, lr=0.005, k_size=7, ratio_train=0.98,
                                   depth=4, h_dim=8)
    elif exp == 40:  # 20.5db
        conf = presets_experiments(exp, b=32, n=100, model_preset="TrueBypass",
                                   scene="ficus", pseudo_color_dimension=3, lr=0.3, k_size=1, ratio_train=0.98)
    elif exp == 41:  # 22.7db
        conf = presets_experiments(exp, b=32, n=100, model_preset="TrueBypass",
                                   scene="ficus", pseudo_color_dimension=3, lr=0.3, k_size=1, ratio_train=0.98,
                                   nb_points=100000)
    elif exp == 42:  # ??
        conf = presets_experiments(exp, b=8, n=100, model_preset="Bypass",
                                   scene="ficus", pseudo_color_dimension=8, lr=0.01, k_size=5, ratio_train=0.98,
                                   nb_points=100000)
    elif exp == 43:
        conf = presets_experiments(exp, b=8, n=100, model_preset="StackedConvolutions",
                                   scene="ficus", pseudo_color_dimension=8, lr=0.01, k_size=3, ratio_train=0.98,
                                   depth=2, h_dim=8,
                                   nb_points=20000)
    elif exp == 44:
        conf = presets_experiments(exp, b=8, n=100, model_preset="StackedConvolutions",
                                   scene="ficus", pseudo_color_dimension=8, lr=0.01, k_size=3, ratio_train=0.98,
                                   depth=2, h_dim=8,
                                   nb_points=100000)
    elif exp == 45:
        conf = presets_experiments(exp, b=8, n=100, model_preset="StackedConvolutions",
                                   scene="ficus", pseudo_color_dimension=8, lr=0.01, k_size=3, ratio_train=0.98,
                                   depth=4, h_dim=8,
                                   nb_points=100000)
    elif exp == 46:
        conf = presets_experiments(exp, b=8, n=300, model_preset="StackedConvolutions",
                                   scene="ficus", pseudo_color_dimension=8, lr=0.01, k_size=3, ratio_train=0.98,
                                   depth=8, h_dim=8,
                                   nb_points=100000)

    elif exp == 500:  # 16.7db
        conf = presets_experiments(exp, b=32, n=100, model_preset="TrueBypass",
                                   scene="old_chair", pseudo_color_dimension=3, lr=0.3, k_size=1, ratio_train=0.98,
                                   nb_points=100000,
                                   ms_supervision=False)
    elif exp == 50:  # 16.7db
        conf = presets_experiments(exp, b=32, n=100, model_preset="TrueBypass",
                                   scene="old_chair", pseudo_color_dimension=3, lr=0.3, k_size=1, ratio_train=0.98,
                                   nb_points=100000)
    elif exp == 51:  # 21.2db ??
        conf = presets_experiments(exp, b=8, n=100, model_preset="Bypass",
                                   scene="old_chair", pseudo_color_dimension=8, lr=0.01, k_size=5, ratio_train=0.98,
                                   nb_points=100000)
    elif exp == 52:  # 28.6db -> not bad
        conf = presets_experiments(exp, b=8, n=100, model_preset="StackedConvolutions",
                                   scene="old_chair", pseudo_color_dimension=8, lr=0.01, k_size=3, ratio_train=0.98,
                                   depth=2, h_dim=8,
                                   nb_points=100000)
    elif exp == 53:  # Redo 52 - longer - 27.7db
        conf = presets_experiments(exp, b=8, n=600, model_preset="StackedConvolutions",
                                   scene="old_chair", pseudo_color_dimension=8, lr=0.01, k_size=3, ratio_train=0.98,
                                   depth=2, h_dim=8,
                                   nb_points=100000)
    elif exp == 54:  # Redo 52 - longer + slower LR - 29.1db
        conf = presets_experiments(exp, b=8, n=600, model_preset="StackedConvolutions",
                                   scene="old_chair", pseudo_color_dimension=8, lr=0.001, k_size=3, ratio_train=0.98,
                                   depth=2, h_dim=8,
                                   nb_points=100000)
    elif exp == 55:  # 29.9db
        conf = presets_experiments(exp, b=8, n=600, model_preset="StackedConvolutions",
                                   scene="old_chair", pseudo_color_dimension=8, lr=0.001, k_size=3, ratio_train=0.98,
                                   depth=4, h_dim=8,
                                   nb_points=100000)
    elif exp == 56:
        conf = presets_experiments(exp, b=8, n=100, model_preset="Bypass",
                                   scene="old_chair", pseudo_color_dimension=8, lr=0.01,
                                   k_size=1,
                                   ratio_train=0.98,
                                   nb_points=100000)
    elif exp == 57:
        conf = presets_experiments(exp, b=8, n=100, model_preset="Bypass",
                                   scene="old_chair", pseudo_color_dimension=8, lr=0.3,
                                   k_size=1,
                                   ratio_train=0.98,
                                   nb_points=100000)
    elif exp == 58:  # 21.5db - 400k points!
        conf = presets_experiments(exp, b=8, n=100, model_preset="TrueBypass",
                                   scene="old_chair", pseudo_color_dimension=3, lr=0.3,
                                   k_size=1,
                                   ratio_train=0.98,
                                   nb_points=400000)
    elif exp == 59:  # 23dB - 400k points -> no MS supervision
        conf = presets_experiments(exp, b=8, n=100, model_preset="TrueBypass",
                                   scene="old_chair", pseudo_color_dimension=3, lr=0.3,
                                   k_size=1,
                                   ratio_train=0.98,
                                   nb_points=400000,
                                   ms_supervision=False)
    elif exp == 60:  # 22.8 - 400k points + LR 0.01 -> no MS supervision
        conf = presets_experiments(exp, b=8, n=100, model_preset="TrueBypass",
                                   scene="old_chair", pseudo_color_dimension=3, lr=0.01,
                                   k_size=1,
                                   ratio_train=0.98,
                                   nb_points=400000,
                                   ms_supervision=False)
    elif exp == 61:  # 25db - 800k points + LR 0.01 -> no MS supervision
        conf = presets_experiments(exp, b=8, n=100, model_preset="TrueBypass",
                                   scene="old_chair", pseudo_color_dimension=3, lr=0.01,
                                   k_size=1,
                                   ratio_train=0.98,
                                   nb_points=800000,
                                   ms_supervision=False)
    elif exp == 62:  # 25db - 800k points + LR 0.01 -> no MS supervision
        conf = presets_experiments(exp, b=8, n=100, model_preset="TrueBypass",
                                   scene="old_chair", pseudo_color_dimension=3, lr=0.01,
                                   k_size=1,
                                   ratio_train=0.98,
                                   nb_points=100000,
                                   ms_supervision=False)

    elif exp == 63:  # 19.6??
        conf = presets_experiments(exp, b=8, n=100, model_preset="StackedConvolutions",
                                   scene="old_chair", pseudo_color_dimension=8, lr=0.001, k_size=3, ratio_train=0.98,
                                   depth=2, h_dim=8,
                                   nb_points=100000,
                                   ms_supervision=False)
    elif exp == 70:  # 26.5dB
        conf = presets_experiments(exp, b=8, n=100, model_preset="TrueBypass",
                                   scene="material_balls", pseudo_color_dimension=3, lr=0.01,
                                   k_size=1,
                                   ratio_train=0.98,
                                   nb_points=800000,
                                   ms_supervision=False)
    print(conf)
    return conf
