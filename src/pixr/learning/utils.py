from pixr.multiview.scenes_utils import load_views
from pixr.synthesis.extract_point_cloud import pick_point_cloud_from_triangles
from interactive_pipe.data_objects.image import Image
from pixr.synthesis.normals import extract_normals
from pixr.synthesis.world_simulation import generate_simulated_world
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


def get_point_cloud(name, num_samples: int = 20000, seed: int = 42):
    wc_triangles, triangle_colors = generate_simulated_world(scene_mode=name)
    wc_normals = extract_normals(wc_triangles)
    wc_points, points_colors, wc_normals = pick_point_cloud_from_triangles(
        wc_triangles,
        triangle_colors,
        wc_normals,
        num_samples=num_samples,
        seed=seed
    )
    return wc_points, wc_normals, points_colors


def preare_image_dataset(out_root: Path, name: str, seed: int = 42, ratio_train: float = 0.8):
    view_dir = out_root/f"{name}"
    views = sorted(list(view_dir.glob("view_*")))
    train_indices, valid_indices = split_dataset(len(views), seed, ratio_train)
    views_train = [views[i] for i in train_indices]
    views_valid = [views[i] for i in valid_indices]
    rendered_view_train, camera_intrinsics_train, camera_extrinsics_train, w, h = prepare_data(views_train)
    rendered_view_valid, camera_intrinsics_valid, camera_extrinsics_valid, w, h = prepare_data(views_valid)
    return (
        rendered_view_train, camera_intrinsics_train, camera_extrinsics_train), (
        rendered_view_valid, camera_intrinsics_valid, camera_extrinsics_valid), (
        w, h
    )


def prepare_dataset(
    out_root: Path, name: str,
    seed: int = 42, ratio_train: float = 0.8,
    num_samples=20000
):
    training_material, validation_material, size = preare_image_dataset(out_root, name, seed, ratio_train)
    wc_points, wc_normals, _ = get_point_cloud(name, num_samples, seed=seed)
    return training_material, validation_material, size, (wc_points, wc_normals)


def save_model(model, points, normals, colors, output_path):
    torch.save({
        "model": model.state_dict(),
        "point_cloud": points,
        "normals": normals,
        "colors": colors,
    },
        output_path
    )


def load_model(path):
    model_dic = torch.load(path)
    model_state_dict = model_dic["model"]
    points = model_dic["point_cloud"]
    normals = model_dic["normals"]
    colors = model_dic["colors"]
    return model_state_dict, points, normals, colors
