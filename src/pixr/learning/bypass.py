from pixr.learning.base import BaseModel
import torch.nn as nn
import torch
from typing import List


class Bypass(BaseModel):
    def __init__(self, in_channels=3, out_channels=3, n_scales=4, **kwargs):
        super().__init__()
        print(f"Pseudo color dimension {in_channels} -> {out_channels}")
        self.in_channels = in_channels
        self.n_scales = n_scales
        self.combine_channels = torch.nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, 1) for _ in range(n_scales)]
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: List[torch.Tensor]):
        return [combine_channels(x[idx]) for idx, combine_channels in enumerate(self.combine_channels)]
