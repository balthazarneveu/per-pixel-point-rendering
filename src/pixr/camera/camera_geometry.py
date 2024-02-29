import torch
from typing import Tuple


def set_camera_parameters(yaw_deg=0., pitch_deg=0., roll_deg=0., trans_x=0., trans_y=0., trans_z=0.) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    yaw = torch.deg2rad(torch.Tensor([yaw_deg]))
    pitch = torch.deg2rad(torch.Tensor([pitch_deg]))
    roll = torch.deg2rad(torch.Tensor([roll_deg]))
    cam_pos = torch.stack([torch.Tensor([trans_x]), torch.Tensor([trans_y]), torch.Tensor([trans_z])])
    return yaw, pitch, roll, cam_pos
