import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from pixr.properties import DATALOADER, BATCH_SIZE, TRAIN, VALIDATION


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
