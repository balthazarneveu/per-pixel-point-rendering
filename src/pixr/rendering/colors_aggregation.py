import torch


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
    # The per-channel for loop can be avoided with the code below by repeating the indexes on the colors axis..
    # but this does not seem to be faster
    for ch in range(colors.shape[-1]):
        flat_colors[..., ch] = flat_colors[..., ch].scatter_reduce(0, flat_idx, colors[..., ch], reduce="mean",
                                                                   include_self=False)
    if False:  # Avoid for loop
        flat_colors = flat_colors.scatter_reduce(
            0, flat_idx.unsqueeze(-1).repeat(1, colors.shape[-1]),
            colors,
            reduce="mean", include_self=False)

    new_colors = flat_colors[1:, :].reshape(h, w, 3)
    print(f"{depths.shape=:}\n , {colors.shape=:}\n{z_buffer_flat.shape=:}")
    print(f"{flat_idx.shape=:}\n {mask.shape=:}\n {min_depth.shape=:}\n {flat_colors.shape=:}\n {new_colors.shape=:}")
    return new_colors
