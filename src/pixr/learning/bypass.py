from pixr.learning.base import BaseModel
import torch.nn as nn
import torch
from typing import List


class Bypass(BaseModel):
    def __init__(self, in_channels=3, out_channels=3, n_scales=4, k_size=1, **kwargs):
        super().__init__()
        print(f"Pseudo color dimension {in_channels} -> {out_channels}")
        self.in_channels = in_channels
        self.n_scales = n_scales
        self.combine_channels = torch.nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, k_size, padding=k_size//2)
             for _ in range(n_scales)]
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: List[torch.Tensor]):
        return [combine_channels(x[idx]) for idx, combine_channels in enumerate(self.combine_channels)]


class TrueBypass(BaseModel):
    def __init__(self, in_channels=3, out_channels=3, n_scales=4, k_size=1, **kwargs):
        super().__init__()
        assert in_channels == out_channels, f"TrueBypass requires in_channels == out_channels"
        print(f"Pseudo color dimension {in_channels} -> {out_channels}")
        self.in_channels = in_channels
        self.n_scales = n_scales
        self.backgound_constant = torch.nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: List[torch.Tensor]):
        return [x[idx] + self.backgound_constant for idx in range(self.n_scales)]
