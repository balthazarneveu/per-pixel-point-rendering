from pixr.properties import DEVICE, OPTIMIZER, PARAMS
from pixr.learning.architecture import load_architecture
from typing import Tuple
import torch


def get_training_content(
    config: dict,
    training_mode: bool = False,
    extra_params=[],
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, dict]:
    model = load_architecture(config)
    optimizer = None
    if training_mode:
        optimizer = torch.optim.Adam(list(model.parameters()) + extra_params, **config[OPTIMIZER][PARAMS])
    return model, optimizer
