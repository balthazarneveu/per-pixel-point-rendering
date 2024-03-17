from pixr.learning.base import BaseModel, ConvolutionStage
from pixr.properties import SIGMOID, SIMPLE_GATE, RELU, LEAKY_RELU, IDENTITY
import torch.nn as nn
import torch
from typing import List


class StackedConvolutions(BaseModel):
    def __init__(self, in_channels=3, out_channels=3, n_scales=4, k_size=3, **kwargs):
        super().__init__()
        print(f"Pseudo color dimension {in_channels} -> {out_channels}")
        print(f"Number of scales {n_scales} , kernel size {k_size}")
        self.in_channels = in_channels
        self.n_scales = n_scales
        self.processing_stages = torch.nn.ModuleList(
            [ConvolutionStage(
                in_channels, out_channels,
                h_dim=8,  # *(2**sc),
                depth=2,
                k_size=k_size,
                activation=RELU,
                # last_activation=SIGMOID
                last_activation=IDENTITY
            )
                for sc in range(n_scales)]
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: List[torch.Tensor]):
        return [processing_stage(x[idx]) for idx, processing_stage in enumerate(self.processing_stages)]


if __name__ == "__main__":
    model = StackedConvolutions(in_channels=8, out_channels=3, n_scales=4)
    print(model)
    x = [torch.rand(1, 8, 256//(2**sc), 256//(2**sc)) for sc in range(4)]
    y = model(x)
    for idx, y_ in enumerate(y):
        print(f"Output {idx} shape {y_.shape}")
    print(model.count_parameters())
    print(model.receptive_field())
