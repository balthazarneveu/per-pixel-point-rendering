import torch
from tqdm import tqdm


def barycentric_coord_broadcast(p, a, b, c):
    """
    Compute barycentric coordinates of point p with respect to triangle (a, b, c).
    """
    # Expand a, b, c to match the shape of p for broadcasting
    a = a.unsqueeze(0).unsqueeze(0)  # Shape becomes [1, 1, 2]
    b = b.unsqueeze(0).unsqueeze(0)
    c = c.unsqueeze(0).unsqueeze(0)

    v0 = b - a
    v1 = c - a
    v2 = p - a

    # Compute dot products using einsum for batched operations
    d00 = torch.einsum('ijk,ijk->ij', v0, v0)
    d01 = torch.einsum('ijk,ijk->ij', v0, v1)
    d11 = torch.einsum('ijk,ijk->ij', v1, v1)
    d20 = torch.einsum('ijk,ijk->ij', v2, v0)
    d21 = torch.einsum('ijk,ijk->ij', v2, v1)

    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return u, v, w


# @torch.compile()
def shade_screen_space_sequential(
    cc_triangles: torch.Tensor,
    colors: torch.Tensor,
    depths: torch.Tensor,
    width: int, height: int,
    show_depth: bool = False,
    no_grad: bool = True,
    debug: bool = False,
    limit: int = -1
) -> torch.Tensor:
    with torch.no_grad() if no_grad else torch.enable_grad():
        # Create an empty image with shape (h, w, 3)
        image = torch.zeros((height, width, 3), device=cc_triangles.device)
        depth_buffer = torch.full((height, width), float('inf'), device=cc_triangles.device, dtype=torch.float32)
        # Get the number of vertices
        num_vertices = cc_triangles.shape[1]
        # Extract the colors at the vertices
        vertex_colors = colors[:, :num_vertices, :]

        for batch_idx in tqdm(range(cc_triangles.shape[0])):
            if limit > 0 and batch_idx > limit:
                break
            depth_values = depths[batch_idx, 0, :]
            triangle = cc_triangles[batch_idx][:2, :]
            color = vertex_colors[batch_idx]
            # -- 1A get triangle bounding box--
            (bb_x_min, bb_y_min), _ = torch.min(triangle, axis=-1)
            (bb_x_max, bb_y_max), _ = torch.max(triangle, axis=-1)
            bb_x_min, bb_y_min = torch.floor(bb_x_min).int(), torch.floor(bb_y_min).int()
            bb_x_max, bb_y_max = torch.ceil(bb_x_max).int(), torch.ceil(bb_y_max).int()

            # -- 1B intersect bounding box with screen boundaries --
            bb_x_min, bb_y_min = max(min(bb_x_min, width - 1), 0), max(min(bb_y_min, height - 1), 0)
            bb_x_max, bb_y_max = max(min(bb_x_max, width - 1), 0), max(min(bb_y_max, height - 1), 0)
            if bb_x_min >= bb_x_max or bb_y_min >= bb_y_max:
                if debug:
                    print("x", bb_x_min, bb_x_max, triangle)
                    print("y", bb_y_min, bb_y_max, triangle)
                continue  # Skip may be due ton infinity values

            # -- 2. Backface culling --
            x0, y0 = triangle[:, 0]
            x1, y1 = triangle[:, 1]
            x2, y2 = triangle[:, 2]
            # Calculate the signed area of the triangle for normal culling
            signed_area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))

            if signed_area <= 0:  # Skip rendering if the triangle is not counterclockwise (facing away)
                continue

            # -- 3. Rasterize the triangle--
            # Create a grid for the bounding box
            x_range = torch.arange(bb_x_min, bb_x_max + 1, dtype=torch.float32, device=cc_triangles.device)
            y_range = torch.arange(bb_y_min, bb_y_max + 1, dtype=torch.float32, device=cc_triangles.device)

            grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='xy', )
            grid_points = torch.stack([grid_x, grid_y], dim=-1)  # Shape: [num_rows, num_cols, 2]

            # Compute barycentric coordinates for the grid points
            tr = triangle  # Shape: [1, 1, 2, 3]
            u, v, w = barycentric_coord_broadcast(grid_points, tr[..., 0], tr[..., 1], tr[..., 2])
            # Interpolate depth for each point in the grid
            interpolated_depth = u * depth_values[0] + v * depth_values[1] + w * depth_values[2]

            # # Find mask where points are inside the triangle
            # -- 4. Mask defines the points inside the triangle and not behing the camera! --
            mask = (u >= 0) & (v >= 0) & (w >= 0) & (interpolated_depth > 0)

            # # Interpolate color for each point in the grid
            interpolated_color = u.unsqueeze(-1) * color[0] + v.unsqueeze(-1) * color[1] + w.unsqueeze(-1) * color[2]

            # # Apply color to image where mask is True, considering the offset of the bounding box
            closer_depth_mask = (interpolated_depth < depth_buffer[bb_y_min:bb_y_max+1, bb_x_min:bb_x_max+1]) & mask
            if not show_depth:
                image[bb_y_min:bb_y_max+1, bb_x_min:bb_x_max+1][closer_depth_mask] = interpolated_color[
                    closer_depth_mask]
            else:
                # Debug : visualize inverse depth
                image[bb_y_min:bb_y_max+1, bb_x_min:bb_x_max+1][closer_depth_mask] = 5. / \
                    interpolated_depth[closer_depth_mask].unsqueeze(-1)
            depth_buffer[bb_y_min:bb_y_max+1, bb_x_min:bb_x_max +
                         1][closer_depth_mask] = interpolated_depth[closer_depth_mask]

    return image
