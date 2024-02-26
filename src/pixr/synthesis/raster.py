import torch
from typing import List, Tuple
from interactive_pipe import interactive, interactive_pipeline
from interactive_pipe.data_objects.curves import Curve, SingleCurve


def get_camera_intrinsics(w: int = 640, h: int = 480) -> torch.Tensor:

    cx, cy = w / 2., h / 2.
    focal_length = 1000.
    fx, fy = focal_length, focal_length
    return torch.Tensor(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ]
    ), w, h


def euler_to_rot(yaw=0., pitch=0., roll=0.):
    tensor_0 = torch.zeros(1)
    tensor_1 = torch.ones(1)
    RX = torch.stack([
        torch.stack([tensor_1, tensor_0, tensor_0]),
        torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
        torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3, 3)

    RY = torch.stack([
        torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
        torch.stack([tensor_0, tensor_1, tensor_0]),
        torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).reshape(3, 3)

    RZ = torch.stack([
        torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
        torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
        torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)
    R = torch.mm(torch.mm(RZ, RY), RX)
    return R


def get_camera_extrinsics(yaw: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor, cam_pos: torch.Tensor) -> torch.Tensor:
    cam_rot = euler_to_rot(yaw, pitch, roll)
    camera_ext = torch.cat([cam_rot, cam_pos], dim=1)
    return camera_ext


def project_3d_to_2d(wc_triangles: torch.Tensor, camera_intrinsics: torch.Tensor, camera_extrinsics: torch.Tensor) -> torch.Tensor:
    cc_triangles = torch.matmul(camera_extrinsics, wc_triangles)
    cc_triangles[:, 1, :] *= -1.  # flip y axis to get a image-like coordinate system
    cc_triangles = torch.matmul(camera_intrinsics, cc_triangles)
    cc_triangles /= cc_triangles[:, -1:, :]  # pinhole model! normalize by distance
    cc_triangles = cc_triangles.numpy()
    return cc_triangles
