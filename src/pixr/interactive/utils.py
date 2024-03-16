import cv2
import numpy as np
import torch


def tensor_to_image(image: torch.Tensor) -> np.ndarray:
    image = image.cpu().numpy()
    return image


def rescale_image(image: np.ndarray, global_params={}) -> np.ndarray:
    scale = global_params.get('scale', 0)
    if scale > 0:
        factor = 2**scale
        image = cv2.resize(image, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
    return image
