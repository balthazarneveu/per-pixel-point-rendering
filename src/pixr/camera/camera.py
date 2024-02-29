import torch
from interactive_pipe import interactive


@interactive(
    exposure=(1., [0., 5.]),
    gamma=(2.2, [1., 4.]),
)
def linear_rgb_to_srgb(img, exposure: float = 1., gamma: float = 2.2):
    img = exposure * img
    # @TODO: white balance and color matrixes should be applied here
    img = img.clip(0., 1.)  # saturate sensor
    sRGB = torch.where(img <= 0.0031308, 12.92 * img, 1.055 * torch.pow(img, 1.0 / gamma) - 0.055)
    return sRGB
