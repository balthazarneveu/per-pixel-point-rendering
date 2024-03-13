import torch
from typing import Optional


# @TODO: refactor this function with the one in shader.py
def point_in_triangle(x: int, y: int, triangle: torch.Tensor) -> bool:
    # Get the vertices of the triangle
    v0, v1, v2 = triangle

    # Calculate the barycentric coordinates of the point (x, y) with respect to the triangle
    barycentric_coords = barycentric_coordinates(x, y, v0, v1, v2)

    # Check if the barycentric coordinates are inside the triangle
    return 0 <= barycentric_coords[0] <= 1 and 0 <= barycentric_coords[1] <= 1 and 0 <= barycentric_coords[2] <= 1


def barycentric_coordinates(x: int, y: int, v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    # Calculate the area of the triangle
    triangle_area = 0.5 * (-v1[1] * v2[0] + v0[1] * (-v1[0] + v2[0]) + v0[0] * (v1[1] - v2[1]) + v1[0] * v2[1])

    # Calculate the barycentric coordinates
    alpha = (0.5 * (-v1[1] * v2[0] + v0[1] * (-v1[0] + v2[0]) + v0[0] * (v1[1] - v2[1]) + v1[0] * v2[1]) -
             0.5 * (-v1[1] * x + v0[1] * (-v1[0] + x) + v0[0] * (v1[1] - y) + v1[0] * y)) / triangle_area
    beta = (0.5 * (v2[1] * x - v0[1] * v2[0] + v0[0] * (-v2[1] + y) - v2[0] * y) / triangle_area)
    gamma = 1 - alpha - beta

    return torch.tensor([alpha, beta, gamma])


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
    w: int,
    h: int,
    camera_intrinsics: torch.Tensor,
    cc_normals: Optional[torch.Tensor],
    debug: Optional[bool] = False,
    no_grad: Optional[bool] = True
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
    # Create an empty image with shape (h, w, 3)
    image = torch.zeros((h, w, 3))
    # Get the number of vertices
    num_vertices = cc_points.shape[1]
    camera_intrinsics_inverse = torch.linalg.inv(camera_intrinsics)
    camera_intrinsics_inverse = camera_intrinsics_inverse.to(cc_points.device)
    # Extract the colors at the vertices
    vertex_colors = colors[:, :num_vertices, :]
    with torch.no_grad() if no_grad else torch.enable_grad():
        # Perform splatting of vertex colors
        for batch_idx in range(cc_points.shape[0]):
            triangle = cc_points[batch_idx]  # Not triangles!
            color = vertex_colors[batch_idx]
            normal = cc_normals[batch_idx] if cc_normals is not None else None
            for node_idx in range(triangle.shape[1]):  # We shall not loop over this dimension!
                x, y = torch.round(triangle[0, node_idx]).long(), torch.round(triangle[1, node_idx]).long()
                # -- Bounds Test --
                if not (0 <= x < w and 0 <= y < h):
                    continue

                # -- In front of camera --
                depth = depths[batch_idx, :, node_idx]

                if depth < 0:
                    if debug:
                        image[y, x] = torch.tensor([0, 1, 1])  # FORCED CYAN FOR DEBUG!
                    continue

                # -- Normal culling! --
                beam_direction = torch.matmul(camera_intrinsics_inverse, triangle[:, node_idx])
                if normal is not None and torch.dot(normal.squeeze(-1), beam_direction) <= 0:
                    if debug:
                        image[y, x] = torch.tensor([1, 1, 0])  # FORCE RED FOR DEBUG
                    continue
                image[y, x] = color[node_idx]

                # if 0 <= x < w and 0 <= y < h:
                #     if debug:
                #         # DEBUG!
                #         for u in range(-4, 4):
                #             for v in range(-4, 4):
                #                 if 0 <= x+u < w and 0 <= y+v < h:
                #                     image[y+v, x+u] = color[node_idx]
                #     image[y, x] = color[node_idx]
    # Return the splatted image
    return image
