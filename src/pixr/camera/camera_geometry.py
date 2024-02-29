import torch
from typing import Tuple


def set_camera_parameters(yaw_deg=0., pitch_deg=0., roll_deg=0., trans_x=0., trans_y=0., trans_z=0.) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    yaw = torch.deg2rad(torch.Tensor([yaw_deg]))
    pitch = torch.deg2rad(torch.Tensor([pitch_deg]))
    roll = torch.deg2rad(torch.Tensor([roll_deg]))
    cam_pos = torch.stack([torch.Tensor([trans_x]), torch.Tensor([trans_y]), torch.Tensor([trans_z])])
    return yaw, pitch, roll, cam_pos


def get_camera_intrinsics(w: int = 640, h: int = 480) -> Tuple[torch.Tensor, int, int]:
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
        torch.stack([tensor_0, torch.cos(pitch), -torch.sin(pitch)]),
        torch.stack([tensor_0, torch.sin(pitch), torch.cos(pitch)])]).reshape(3, 3)

    RY = torch.stack([
        torch.stack([torch.cos(yaw), tensor_0, torch.sin(yaw)]),
        torch.stack([tensor_0, tensor_1, tensor_0]),
        torch.stack([-torch.sin(yaw), tensor_0, torch.cos(yaw)])]).reshape(3, 3)

    RZ = torch.stack([
        torch.stack([torch.cos(roll), -torch.sin(roll), tensor_0]),
        torch.stack([torch.sin(roll), torch.cos(roll), tensor_0]),
        torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)
    R = torch.mm(torch.mm(RZ, RY), RX)
    return R


def get_camera_extrinsics(yaw: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor, cam_pos: torch.Tensor) -> torch.Tensor:
    cam_rot = euler_to_rot(yaw, pitch, roll)
    camera_ext = torch.cat([cam_rot, cam_pos], dim=1)
    return camera_ext
