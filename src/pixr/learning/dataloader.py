import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from rstor.synthetic_data.dead_leaves_cpu import cpu_dead_leaves_chart
from rstor.synthetic_data.dead_leaves_gpu import gpu_dead_leaves_chart
from rstor.properties import DATALOADER, BATCH_SIZE, TRAIN, VALIDATION, LENGTH, CONFIG_DEAD_LEAVES, SIZE
import cv2
from skimage.filters import gaussian
import random
import numpy as np

from numba import cuda

from rstor.utils import DEFAULT_TORCH_FLOAT_TYPE


class ViewDataset(Dataset):
    def __init__(
        self,
        views: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
    ):
        self.length = views.shape[0]
        self.views = views
        self.camera_intrinsics = camera_intrinsics
        self.camera_extrinsics = camera_extrinsics

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.views[idx].permute(2, 0, 1), self.camera_intrinsics[idx], self.camera_extrinsics[idx]


def get_data_loader(config, training_material, valid_material):
    dl_train = ViewDataset(*training_material)
    dl_valid = ViewDataset(*valid_material)
    dl_dict = {
        TRAIN: DataLoader(
            dl_train,
            shuffle=False,
            batch_size=config[DATALOADER][BATCH_SIZE][TRAIN],
        ),
        VALIDATION: DataLoader(
            dl_valid,
            shuffle=False,
            batch_size=config[DATALOADER][BATCH_SIZE][VALIDATION]
        ),
    }
    return dl_dict
