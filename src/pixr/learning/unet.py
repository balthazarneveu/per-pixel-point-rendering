from pixr.learning.base import BaseModel
import torch.nn as nn


class UNet(BaseModel):
    def __init__(self, in_channels=3, out_channels=3, k_size=3, encoder=[1, 1, 1, 1], decoder=[1, 1, 1, 1], thickness=4, **kwargs):
        super().__init__()
        print(f"Pseudo color dimension {in_channels} -> {out_channels}")
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, k_size, padding=k_size//2)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        assert x.shape[-3] == self.in_channels, f"Input shape {x.shape} does not match {self.in_channels}"
        x = self.conv1(x)
        return x
