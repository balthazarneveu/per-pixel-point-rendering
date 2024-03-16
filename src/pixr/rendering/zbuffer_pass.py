import torch


def zbuffer_pass(points, depths, w, h, camera_intrinsics_inverse, cc_normals, scale_factor,
                 normal_culling_flag=True):
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
    z_buffer = z_buffer_flat[1:].reshape(h, w)
    return z_buffer


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


# def aggregate_colors_fuzzy_depth_test(points, depths, w, h, camera_intrinsics_inverse, cc_normals, scale_factor
#                                       z_buffer):
