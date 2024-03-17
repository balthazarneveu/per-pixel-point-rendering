from pixr.properties import MODEL, NAME, N_PARAMS, ARCHITECTURE
from pixr.learning.unet import UNet
from pixr.learning.bypass import Bypass, TrueBypass
from pixr.learning.stacked_convolutions import StackedConvolutions
import torch


def load_architecture(config: dict) -> torch.nn.Module:
    conf_model = config[MODEL][ARCHITECTURE]
    if config[MODEL][NAME] == UNet.__name__:
        model = UNet(**conf_model)
    elif config[MODEL][NAME] == StackedConvolutions.__name__:
        model = StackedConvolutions(**conf_model)
    elif config[MODEL][NAME] == Bypass.__name__:
        model = Bypass(**conf_model)
    elif config[MODEL][NAME] == TrueBypass.__name__:
        model = TrueBypass(**conf_model)
    else:
        raise ValueError(f"Unknown model {config[MODEL][NAME]}")
    config[MODEL][N_PARAMS] = model.count_parameters()
    config[MODEL]["receptive_field"] = model.receptive_field()
    return model
