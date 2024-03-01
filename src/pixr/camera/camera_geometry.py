import torch
from typing import Tuple


def set_camera_parameters(yaw_deg=0., pitch_deg=0., roll_deg=0., trans_x=0., trans_y=0., trans_z=0.) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    yaw = torch.deg2rad(torch.Tensor([yaw_deg]))
    pitch = torch.deg2rad(torch.Tensor([pitch_deg]))
    roll = torch.deg2rad(torch.Tensor([roll_deg]))
    cam_pos = torch.stack([torch.Tensor([trans_x]), torch.Tensor([trans_y]), torch.Tensor([trans_z])])
    return yaw, pitch, roll, cam_pos


def get_camera_intrinsics(w: int = 640, h: int = 480, focal_length_pix: float = 1000.) -> Tuple[torch.Tensor, int, int]:
    """
    Get camera intrinsics matrix and image dimensions.

    Args:
        w (int): Width of the image in pixels. Default is 640.
        h (int): Height of the image in pixels. Default is 480.
        focal_length_pix (float): Focal length of the camera in pixels. Default is 1000.

    Returns:
        Tuple[torch.Tensor, int, int]: A tuple containing the camera intrinsics matrix, width, and height of the image.
    """
    cx, cy = w / 2., h / 2.
    fx, fy = focal_length_pix, focal_length_pix
    return torch.Tensor(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ]
    ), w, h


def euler_to_rot(
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    roll: torch.Tensor
) -> torch.Tensor:
    """
    Converts Euler angles to a rotation matrix.

    Args:
        yaw (torch.Tensor): Yaw angle in radians.
        pitch (torch.Tensor): Pitch angle in radians.
        roll (torch.Tensor): Roll angle in radians.

    Returns:
        torch.Tensor: Rotation matrix.

    """
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


def get_camera_extrinsics(
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    roll: torch.Tensor,
    cam_pos: torch.Tensor  # translation
) -> torch.Tensor:
    cam_rot = euler_to_rot(yaw, pitch, roll)
    camera_ext = torch.cat([cam_rot, cam_pos], dim=1)
    return camera_ext
