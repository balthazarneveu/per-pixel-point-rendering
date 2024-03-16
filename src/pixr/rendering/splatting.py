import torch
from typing import Optional
from pixr.rendering.zbuffer_pass import zbuffer_pass, zbuffer_pass_for_loop, aggregate_colors_fuzzy_depth_test

# @TODO: WARNING: channel last! - not compatible with usual N, C, H, W
# @TODO: do not set the image to zero, needs to be initialized outside
# @TODO: z-buffer here / depths tests here
# @TODO: get rid of the primitive dimension!
# @TODO: multiscale
# @torch.compile()


def splat_points(
    cc_points: torch.Tensor,
    colors: torch.Tensor,
    depths: torch.Tensor,
    w_full: int,
    h_full: int,
    camera_intrinsics: torch.Tensor,
    cc_normals: Optional[torch.Tensor],
    debug: Optional[bool] = False,
    no_grad: Optional[bool] = True,
    z_buffer_flag: Optional[bool] = True,
    scale: Optional[int] = 0,
    normal_culling_flag: Optional[bool] = True,
    fuzzy_depth_test: Optional[float] = 0.01,
    for_loop_zbuffer: Optional[bool] = False,
    global_params: Optional[dict] = {}
) -> torch.Tensor:
    """
    Splat the colors of the vertices onto an image.

    Args:
        cc_points (torch.Tensor): Tensor of shape (batch_size, num_vertices, 2)
        representing the point cloud projected in camera space.
        colors (torch.Tensor): Tensor of shape (batch_size, num_vertices, 3) representing the colors at the vertices.
        depths (torch.Tensor): Tensor of shape (batch_size, 1, prim, ) representing the depths of the triangles.
        w (int): Width of the output image.
        h (int): Height of the output image.
        debug (Optional[bool], optional): If True, visualize the splatting process with larger points.
        Defaults to False.
        no_grad (Optional[bool], optional): If True, disable gradient computation during splatting.
        Defaults to False.

    Returns:
        torch.Tensor: Tensor of shape (H, W, C=3rgb) representing the splatted image.
    """
    scale_factor = 2.**scale
    global_params['scale'] = scale
    # Create an empty image with shape (h, w, 3)
    w, h = int(w_full/scale_factor), int(h_full/scale_factor)
    image = torch.zeros((h, w, 3))
    image = image.to(cc_points.device)
    # Get the number of vertices
    num_vertices = cc_points.shape[1]
    camera_intrinsics_inverse = torch.linalg.inv(camera_intrinsics)
    camera_intrinsics_inverse = camera_intrinsics_inverse.to(cc_points.device)
    # Extract the colors at the vertices
    vertex_colors = colors[:, :num_vertices, :]
    weights = torch.zeros((h, w), device=cc_points.device)
    weights.requires_grad = False
    with torch.no_grad() if no_grad else torch.enable_grad():
        if z_buffer_flag:
            with torch.no_grad():
                if fuzzy_depth_test > 0 and z_buffer_flag:
                    if not for_loop_zbuffer:
                        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> No for loop zbuffer")
                        z_buffer_flat, point_to_flat_image_coordinate_mapping = zbuffer_pass(
                            cc_points[:, :, 0],
                            depths[..., 0, 0],
                            w, h,
                            camera_intrinsics_inverse,
                            cc_normals,
                            scale_factor=scale_factor,
                            normal_culling_flag=normal_culling_flag
                        )
                        z_buffer_flat = (1+fuzzy_depth_test) * z_buffer_flat
                        z_buffer = z_buffer_flat[1:].reshape(h, w)
                    else:
                        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FOR!!! loop zbuffer")
                        z_buffer = zbuffer_pass_for_loop(
                            cc_points[:, :, 0],
                            depths[..., 0, 0],
                            w, h,
                            camera_intrinsics_inverse,
                            cc_normals,
                            scale_factor=scale_factor,
                            normal_culling_flag=normal_culling_flag
                        )
                    # if debug:
                    #     valid_mask = z_buffer < float('inf')
                    #     z_buffer[~valid_mask] = 0.
                    #     z_buffer = (1./z_buffer).clip(0, 1)
                    #     return z_buffer
                    z_buffer = (1+fuzzy_depth_test) * z_buffer
                else:
                    z_buffer = torch.full((h, w), float('inf'))
        # Perform splatting of vertex colors
    if not for_loop_zbuffer:
        z_buffer_flat = (1+fuzzy_depth_test) * z_buffer_flat
        image = aggregate_colors_fuzzy_depth_test(
            depths[..., 0, 0],
            colors[:, 0, :],
            z_buffer_flat,
            point_to_flat_image_coordinate_mapping,
            w, h
        )
        return image
    else:
        for batch_idx in range(cc_points.shape[0]):
            point = cc_points[batch_idx, :, 0]
            color = vertex_colors[batch_idx, 0, :]

            x, y = torch.round(point[0]/scale_factor).long(), torch.round(point[1]/scale_factor).long()
            # -- Bounds Test --
            if not (0 <= x < w and 0 <= y < h):
                continue

            # -- In front of camera --
            depth = depths[batch_idx, :, 0]

            if depth < 0:
                # if debug:
                #     image[y, x] = torch.tensor([0, 1, 1])  # FORCED CYAN FOR DEBUG!
                continue
            # -- Normal culling! --
            if normal_culling_flag:
                normal = cc_normals[batch_idx] if cc_normals is not None else None
                beam_direction = torch.matmul(camera_intrinsics_inverse, point)
                if normal is not None and torch.dot(normal.squeeze(-1), beam_direction) <= 0:
                    # if debug:
                    #     image[y, x] = torch.tensor([1, 1, 0])  # FORCE RED FOR DEBUG
                    continue

            # -- Z-buffering! --
            if z_buffer_flag:
                buffered_z = z_buffer[y, x]
                if depth <= buffered_z:
                    if fuzzy_depth_test == 0:
                        # Update Z-buffer only in the case of the fuzzy depth test
                        z_buffer[y, x] = depth
                else:
                    continue
            if fuzzy_depth_test == 0:
                image[y, x, :] = color
                weights[y, x] = 1.
            else:
                image[y, x, :] += color
                weights[y, x] += 1.
    background = 0.275
    weights = weights.unsqueeze(-1)
    mask = (weights == 0) * 1.
    image = (1-mask) * (image/(weights+1e-8)) + mask * background  # .repeat(1, 1, 3))
    # Return the splatted image
    # return image
    if debug:
        return weights.repeat(1, 1, 3)/5.
    else:
        return image
