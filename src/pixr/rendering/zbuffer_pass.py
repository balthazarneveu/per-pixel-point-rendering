import torch
from typing import Tuple


def zbuffer_pass(
    points,  # [M, d=xy]
    depths,  # [M, d=1]
    w, h,
    camera_intrinsics_inverse,  # [3, 3]
    cc_normals,  # [M, d=3, 1]
    scale_factor,
    normal_culling_flag=True
) -> Tuple[torch.Tensor, torch.Tensor]:
    # points.shape=torch.Size([100, 3]), depths.shape=torch.Size([100])
    # cc_normals.shape=torch.Size([100, 3, 1]),
    # colors.shape=torch.Size([100, 3]),  z_buffer_flat.shape=torch.Size([307201])
    z_buffer_flat = torch.full((1 + h * w,), float('inf'), device=points.device, dtype=torch.float32)
    x, y = torch.round(points[:, 0]/scale_factor).long(), torch.round(points[:, 1]/scale_factor).long()
    idx = 1 + y*w+x
    # -- Bounds Test --
    idx *= (x >= 0) * (x < w) * (y >= 0) * (y < h)  # force to 0 if out of bounds -> then remove 0
    # -- In front of camera --
    idx[depths < 0] = 0
    # -- Normal culling! --
    if normal_culling_flag:
        beam_direction = torch.matmul(camera_intrinsics_inverse, points.t()).t()
        # https://github.com/pytorch/pytorch/issues/18027 batched dot product (a*b).sum
        mask_culling = (cc_normals.squeeze(-1)*beam_direction).sum(axis=-1) <= 0
        idx[mask_culling] = 0
    z_buffer_flat = z_buffer_flat.scatter_reduce(0, idx, depths, reduce="amin", include_self=True)
    return z_buffer_flat, idx


