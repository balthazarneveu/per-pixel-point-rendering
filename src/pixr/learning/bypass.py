from pixr.learning.base import BaseModel
import torch.nn as nn


class Bypass(BaseModel):
    def __init__(self, in_channels=3, out_channels=3, **kwargs):
        super().__init__()
        print(f"Pseudo color dimension {in_channels} -> {out_channels}")
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return x[..., :3, :, :]
