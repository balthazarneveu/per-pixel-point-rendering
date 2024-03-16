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
    return rendered_images, camera_intrinsics, camera_extrinsics, w, h


def split_dataset(rendered_images_size: int, seed: int = 42, ratio_train: float = 0.8) -> Tuple[List[int], List[int]]:
    assert ratio_train > 0 and ratio_train < 1.
    random.seed(seed)
    indices = list(range(rendered_images_size))
    random.shuffle(indices)
    split = int(rendered_images_size * ratio_train)
    train_indices = indices[:split]
    val_indices = indices[split:]
    return train_indices, val_indices


def prepare_dataset(out_root: Path, name: str,):
    view_dir = out_root/f"{name}"
    views = sorted(list(view_dir.glob("view_*")))
    all_rendered_images, camera_intrinsics, camera_extrinsics, w, h = prepare_data(views)
    all_rendered_images.requires_grad = False
    # Limit the amount of views for now! -> split train and validation here!
    # rendered_images = all_rendered_images[:5]
    rendered_images = all_rendered_images
