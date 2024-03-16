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


def aggregate_colors_fuzzy_depth_test(
    depths,  # [M, d=1]
    colors,  # [M, rgb or pseudo-colors]
    z_buffer_flat,  # [1 + h * w]
    flat_idx,  # [M] (-> int h*w+1) : point_to_flat_image_coordinate_mapping
    w: int,
    h: int,
):
    # z-buffer flat [?, 10m, 3m, inf, 50cm, -10m ,....] <-> dropped first element + image [h,w]

    #               point 0          point1
    # flat_idx = [645 <-> (0, 5),  1280 <-> (1, 0) , ...... ]
    # flat_idx [M] (-> int h*w+1)

    # flat_idx = zbuffer_pass(points, depths, w, h, camera_intrinsics_inverse,
    #                         cc_normals, scale_factor, normal_culling_flag=normal_culling_flag, early_return=True)
    # Per point minimum depth scaled by 1.01, if point depth is <= , then it's a valid point

    min_depth = z_buffer_flat[flat_idx]
    mask = depths <= z_buffer_flat[flat_idx]  # fuzzy depth test
    flat_idx[~mask] = 0  # discarded points are mapped to the dropped element index.
    flat_colors = torch.zeros((1 + h * w, colors.shape[-1]), device=colors.device, dtype=torch.float32)
    for ch in range(colors.shape[-1]):
        flat_colors[..., ch] = flat_colors[..., ch].scatter_reduce(0, flat_idx, colors[..., ch], reduce="mean",
                                                                   include_self=False)
    new_colors = flat_colors[1:, :].reshape(h, w, 3)
    print(f"{depths.shape=:}\n , {colors.shape=:}\n{z_buffer_flat.shape=:}")
    print(f"{flat_idx.shape=:}\n {mask.shape=:}\n {min_depth.shape=:}\n {flat_colors.shape=:}\n {new_colors.shape=:}")
    return new_colors


def zbuffer_pass_for_loop(points, depths, w, h, camera_intrinsics_inverse, cc_normals, scale_factor,
                          normal_culling_flag=True):
    z_buffer = torch.full((h, w), float('inf'))
    for batch_idx in range(points.shape[0]):
        point = points[batch_idx, :]
        x, y = torch.round(point[0]/scale_factor).long(), torch.round(point[1]/scale_factor).long()
        # -- Bounds Test --
        if not (0 <= x < w and 0 <= y < h):
            continue
        # -- In front of camera --
        depth = depths[batch_idx]
        if depth < 0:
            continue
        # -- Normal culling! --
        if normal_culling_flag:
            normal = cc_normals[batch_idx] if cc_normals is not None else None
            beam_direction = torch.matmul(camera_intrinsics_inverse, point)
            if normal is not None and torch.dot(normal.squeeze(-1), beam_direction) <= 0:
                continue
        # -- Z-buffering! --
        buffered_z = z_buffer[y, x]
        if depth <= buffered_z:
            z_buffer[y, x] = depth
        else:
            continue
    return z_buffer
