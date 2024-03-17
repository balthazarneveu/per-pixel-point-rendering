import torch
from pixr.properties import LEAKY_RELU, RELU, SIMPLE_GATE, TANH, SIGMOID, IDENTITY, DEVICE
from typing import Optional, Tuple


class BaseModel(torch.nn.Module):
    """Base class for all restoration models with additional useful methods"""

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def receptive_field(
        self,
        channels: Optional[int] = 3,
        size: Optional[int] = 256,
        device: Optional[str] = None
    ) -> Tuple[int, int]:
        """Compute the receptive field of the model

        Returns:
            int: receptive field
        """
        if device is None:
            device = "cpu"
        channels = self.in_channels if hasattr(self, "in_channels") else channels
        n_scales = self.n_scales if hasattr(self, "n_scales") else 1
        input_tensor = [
            torch.ones(1, channels, size//(2**sc), size//(2**sc),
                       requires_grad=True).to(device) for sc in range(n_scales)
        ]
        out = self.forward(input_tensor)
        if isinstance(out, (list, tuple)):
            out = out[0]
        grad = torch.zeros_like(out)
        grad[..., out.shape[-2]//2, out.shape[-1]//2] = torch.nan  # set NaN gradient at the middle of the output
        out.backward(gradient=grad)
        self.zero_grad()
        receptive_field_mask = input_tensor[0].grad.isnan()[0, 0]
        receptive_field_indexes = torch.where(receptive_field_mask)
        # Count NaN in the input
        receptive_x = 1+receptive_field_indexes[-1].max() - receptive_field_indexes[-1].min()  # Horizontal x
        receptive_y = 1+receptive_field_indexes[-2].max() - receptive_field_indexes[-2].min()  # Vertical y
        return receptive_x.item(), receptive_y.item()


class SimpleGate(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


def get_non_linearity(activation: str):
    if activation == LEAKY_RELU:
        non_linearity = torch.nn.LeakyReLU()
    elif activation == RELU:
        non_linearity = torch.nn.ReLU()
    elif activation is None or activation == IDENTITY:
        non_linearity = torch.nn.Identity()
    elif activation == SIMPLE_GATE:
        non_linearity = SimpleGate()
    elif activation == SIGMOID:
        non_linearity = torch.nn.Sigmoid()
    elif activation == TANH:
        non_linearity = torch.nn.Tanh()
    else:
        raise ValueError(f"Unknown activation {activation}")
    return non_linearity


class BaseConvolutionBlock(torch.nn.Module):
    def __init__(self, ch_in, ch_out: int, k_size: int, activation="LeakyReLU", bias: bool = True, padding_mode="zeros") -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(ch_in, ch_out, k_size, padding=k_size//2, bias=bias, padding_mode=padding_mode)
        self.non_linearity = get_non_linearity(activation)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = self.conv(x_in)  # [N, ch_in, H, W] -> [N, ch_in+channels_extension, H, W]
        x = self.non_linearity(x)
        return x


class ConvolutionStage(torch.nn.Module):
    def __init__(
        self,  ch_in: int, ch_out: int,
        h_dim: int = None,
        k_size: int = 3,
        bias: bool = True,
        activation=LEAKY_RELU,
        last_activation=LEAKY_RELU,
        depth: int = 1,
    ) -> None:
        """Chain several convolutions together to create a processing stage
        for a single scale of a UNET
        or stacked convolutions
        """
        super().__init__()
        self.conv_stage = torch.nn.ModuleList()
        if h_dim is None:
            h_dim = max(ch_out, ch_in)
        for idx in range(depth):
            ch_inp = ch_in if idx == 0 else h_dim
            ch_outp = ch_out if idx == depth-1 else h_dim
            self.conv_stage.append(
                BaseConvolutionBlock(
                    ch_inp, ch_outp,
                    k_size=k_size,
                    activation=activation if idx < depth-1 else last_activation,
                    bias=bias,
                    # padding_mode="replicate"
                )
            )
        self.conv_stage = torch.nn.Sequential(*self.conv_stage)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        return self.conv_stage(x_in)
