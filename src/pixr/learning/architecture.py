from pixr.properties import MODEL, NAME, N_PARAMS, ARCHITECTURE
from pixr.learning.unet import UNet
import torch


def load_architecture(config: dict) -> torch.nn.Module:
    conf_model = config[MODEL][ARCHITECTURE]
    if config[MODEL][NAME] == UNet.__name__:
        model = UNet(**conf_model)
    else:
        raise ValueError(f"Unknown model {config[MODEL][NAME]}")
    config[MODEL][N_PARAMS] = model.count_parameters()
    config[MODEL]["receptive_field"] = model.receptive_field()
    return model
