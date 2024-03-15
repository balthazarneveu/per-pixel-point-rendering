import torch


def zbuffer_pass(points, depths, w, h, camera_intrinsics, cc_normals, scale_factor):
    z_buffer_flat = torch.full((1 + h * w,), float('inf'), device=points.device, dtype=torch.float32)
    x, y = torch.round(points[:, 0]/scale_factor).long(), torch.round(points[:, 1]/scale_factor).long()
    idx = 1 + y*w+x
    idx *= (x >= 0) * (x < w) * (y >= 0) * (y < h)  # force to 0 if out of bounds -> then remove 0
    z_buffer_flat = z_buffer_flat.scatter_reduce(0, idx, depths, reduce="amin", include_self=True)
    z_buffer = z_buffer_flat[1:].reshape(h, w)
    if False:
        valid_mask = z_buffer < float('inf')
        z_buffer[~valid_mask] = 0.
        z_buffer = z_buffer.clip(0, 1)
    return z_buffer
