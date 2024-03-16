import torch
from typing import Optional
from pixr.rendering.zbuffer_pass import zbuffer_pass
from pixr.rendering.colors_aggregation import aggregate_colors_fuzzy_depth_test
# @TODO: WARNING: channel last! - not compatible with usual N, C, H, W


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
    w, h = int(w_full/scale_factor), int(h_full/scale_factor)
    image = torch.zeros((h, w, 3))
    image = image.to(cc_points.device)
    camera_intrinsics_inverse = torch.linalg.inv(camera_intrinsics)
    camera_intrinsics_inverse = camera_intrinsics_inverse.to(cc_points.device)
    weights = torch.zeros((h, w), device=cc_points.device)
    weights.requires_grad = False
    with torch.no_grad() if no_grad else torch.enable_grad():
        with torch.no_grad():
            z_buffer_flat, point_to_flat_image_coordinate_mapping = zbuffer_pass(
                cc_points[:, :, 0],
                depths[..., 0, 0],
                w, h,
                camera_intrinsics_inverse,
                cc_normals,
                scale_factor=scale_factor,
                normal_culling_flag=normal_culling_flag
            )
            if not z_buffer_flag:
                z_buffer_flat = torch.full((1 + h * w,), float('inf'), device=cc_points.device, dtype=torch.float32)
    z_buffer_flat = (1+fuzzy_depth_test) * z_buffer_flat
    image = aggregate_colors_fuzzy_depth_test(
        depths[..., 0, 0],
        colors[:, 0, :],
        z_buffer_flat,
        point_to_flat_image_coordinate_mapping,
        w, h
    )
    if debug:
        z_buffer = z_buffer_flat[1:].reshape(h, w)
        return (z_buffer != float("inf"))*(z_buffer/20.).clip(0., 1.)
    return image
