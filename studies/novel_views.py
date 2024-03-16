from pathlib import Path
import torch
from differentiate_points_splat import forward_chain_not_parametric
from pixr.camera.camera_geometry import set_camera_parameters, get_camera_extrinsics, get_camera_intrinsics
from interactive_pipe.data_objects.image import Image


def load_colored_point_cloud_from_files(splat_scene_path):
    scene_dict = torch.load(splat_scene_path)
    wc_points = scene_dict["point_cloud"]
    wc_normals = scene_dict["normals"]
    color_pred = scene_dict["colors"]
    wc_points.requires_grad = False
    wc_normals.requires_grad = False
    color_pred.requires_grad = False
    return wc_points, wc_normals, color_pred


def main_static_version(splat_scene_path):
    wc_points, wc_normals, color_pred = load_colored_point_cloud_from_files(splat_scene_path)
    scale = 0
    set_camera_parameters(z=13.)
    yaw, pitch, roll, cam_pos = set_camera_parameters(trans_z=13.741)
    camera_extrinsics = get_camera_extrinsics(yaw, pitch, roll, cam_pos)
    camera_intrinsics, w, h = get_camera_intrinsics()
    img_pred = forward_chain_not_parametric(
        wc_points, wc_normals, camera_extrinsics, camera_intrinsics, color_pred, w, h, scale=scale, no_grad=True)
    Image(img_pred.detach().cpu().numpy()).show()


if __name__ == '__main__':
    splat_scene_path = Path(
        "/Data/code/per-pixel-point-rendering/__output/staircase_splat_differentiate_points/checkpoint_00040.pt")
    assert splat_scene_path.exists()
    main_static_version(splat_scene_path)
