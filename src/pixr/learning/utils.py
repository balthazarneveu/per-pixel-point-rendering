from pixr.multiview.scenes_utils import load_views
from interactive_pipe.data_objects.image import Image
import torch
from pathlib import Path
from typing import List, Tuple
import random


def prepare_data(views_path_list: List[Path]):
    views = load_views(views_path_list)
    rendered_images = []
    camera_intrinsics = []
    camera_extrinsics = []
    for idx, view_dict in enumerate(views):
        img = Image.load_image(view_dict["path_to_image"])
        img = torch.from_numpy(img).float()  # .permute(2, 0, 1)
        rendered_images.append(img)
        camera_intrinsics_single, w, h = view_dict["camera_intrinsics"], view_dict["w"], view_dict["h"]
        camera_extrinsics_single = view_dict["camera_extrinsics"]
        camera_intrinsics.append(camera_intrinsics_single)
        camera_extrinsics.append(camera_extrinsics_single)
    rendered_images = torch.stack(rendered_images)
    camera_intrinsics = torch.stack(camera_intrinsics)
    camera_extrinsics = torch.stack(camera_extrinsics)
    rendered_images.requires_grad = False
    camera_intrinsics.requires_grad = False
    camera_extrinsics.requires_grad = False
    return rendered_images, camera_intrinsics, camera_extrinsics, w, h


def split_dataset(number_of_views: int, seed: int = 42, ratio_train: float = 0.8) -> Tuple[List[int], List[int]]:
    assert ratio_train > 0 and ratio_train < 1.
    random.seed(seed)
    indices = list(range(number_of_views))
    random.shuffle(indices)
    split = int(number_of_views * ratio_train)
    train_indices = indices[:split]
    val_indices = indices[split:]
    return train_indices, val_indices


def prepare_dataset(out_root: Path, name: str, seed: int = 42, ratio_train: float = 0.8):
    view_dir = out_root/f"{name}"
    views = sorted(list(view_dir.glob("view_*")))
    train_indices, valid_indices = split_dataset(len(views))
    views_train = [views[i] for i in train_indices]
    views_valid = [views[i] for i in valid_indices]
    rendered_view_train, camera_intrinsics_train, camera_extrinsics_train, w, h = prepare_data(views_train)
    rendered_view_valid, camera_intrinsics_valid, camera_extrinsics_valid, w, h = prepare_data(views_valid)
    # all_rendered_images.requires_grad = False
    # rendered_images = all_rendered_images
    return (rendered_view_train, camera_intrinsics_train, camera_extrinsics_train), (rendered_view_valid, camera_intrinsics_valid, camera_extrinsics_valid), (w, h)
