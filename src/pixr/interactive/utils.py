import cv2
import numpy as np
import torch


def tensor_to_image(image: torch.Tensor) -> np.ndarray:
    if image is None:
        return None
    image = image.cpu().contiguous().numpy().clip(0, 1)
    return image


def rescale_image(image: np.ndarray, global_params={}) -> np.ndarray:
    if image is None:
        return None
    scale = global_params.get('scale', 0)
    if scale > 0:
        factor = 2**scale
        image = cv2.resize(image, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
    return image
