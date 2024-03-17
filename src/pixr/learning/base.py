import torch
from pixr.properties import LEAKY_RELU, RELU, SIMPLE_GATE, DEVICE
from typing import Optional, Tuple


class SimpleGate(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


def get_non_linearity(activation: str):
    if activation == LEAKY_RELU:
        non_linearity = torch.nn.LeakyReLU()
    elif activation == RELU:
        non_linearity = torch.nn.ReLU()
    elif activation is None:
        non_linearity = torch.nn.Identity()
    elif activation == SIMPLE_GATE:
        non_linearity = SimpleGate()
    else:
        raise ValueError(f"Unknown activation {activation}")
    return non_linearity


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
            torch.ones(1, channels, size//(2**sc), size//(2**sc), requires_grad=True).to(device) for sc in range(n_scales)
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
