import torch
from tqdm import tqdm


def batched_barycentric_coord(p: torch.Tensor, triangles):
    """_summary_

    Args:
        p (torch.Tensor): _description_
        triangles (_type_): _description_

    Returns:
        _type_: _description_
    """
    triangles = triangles.unsqueeze(0)  # .repeat(p.shape[0], 1, 1, 1)
    p = p.squeeze(-1)
    p = p.unsqueeze(1)
    print("triangles", triangles.shape, "points", p.shape)
    a = triangles[..., 0]
    b = triangles[..., 1]
    c = triangles[..., 2]
    print("triangle edge", a.shape, "points", p.shape)
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


def shade_screen_space(
        cc_triangles: torch.Tensor,
        colors: torch.Tensor,
        depths: torch.Tensor,
        width: int, height: int,
        show_depth: bool = False,
        no_grad: bool = True,
        debug: bool = False,
        for_loop: bool = False
) -> torch.Tensor:
    if for_loop:
        return shade_screen_space_for_loop(cc_triangles, colors, depths, width, height, show_depth, no_grad, debug)
    else:
        return shade_screen_space_no_for_loop(cc_triangles, colors, depths, width, height, show_depth, no_grad, debug)


def shade_screen_space_no_for_loop(
        cc_triangles: torch.Tensor,
        colors: torch.Tensor,
        depths: torch.Tensor,
        width: int, height: int,
        show_depth: bool = False,
        no_grad: bool = True,
        debug: bool = False
) -> torch.Tensor:
    with torch.no_grad() if no_grad else torch.enable_grad():
        # Create an empty image with shape (h, w, 3)
        image = torch.zeros((height, width, 3), device=cc_triangles.device)
        depth_buffer = torch.full((height, width), float('inf'), device=cc_triangles.device, dtype=torch.float32)
        depth_buffer = depth_buffer.view(-1, 1)
        # Get the number of vertices
        num_vertices = cc_triangles.shape[1]
        # Extract the colors at the vertices
        vertex_colors = colors[:, :num_vertices, :]

        batch_size = 64
        num_batches = (cc_triangles.shape[0] + batch_size - 1) // batch_size
        x_range = torch.arange(0, image.shape[1], dtype=torch.float32, device=cc_triangles.device)
        y_range = torch.arange(0, image.shape[0], dtype=torch.float32, device=cc_triangles.device)
        grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='xy', )
        grid_points = torch.stack([grid_x, grid_y], dim=-1)  # Shape: [num_rows, num_cols, 2]
        grid_points = grid_points.view(-1, 2)
        for batch_idx in tqdm(range(num_batches)):
            # Compute start and end indices for the current batch
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, cc_triangles.shape[0])

            # Select the current batch of triangles, depths, and colors
            batch_triangles = cc_triangles[start_idx:end_idx, :2, :]
            batch_depths = depths[start_idx:end_idx]
            batch_vertex_colors = vertex_colors[start_idx:end_idx]
            if debug:
                print(batch_triangles.shape, batch_depths.shape, batch_vertex_colors.shape)
            bb_min, _ = torch.min(batch_triangles, axis=-1)
            bb_max, _ = torch.max(batch_triangles, axis=-1)
            bb_x_min, bb_y_min = bb_min[..., 0, ...], bb_min[..., 1, ...]
            bb_x_max, bb_y_max = bb_max[..., 0, ...], bb_max[..., 1, ...]
            bb_x_min, bb_y_min = torch.floor(bb_x_min).int(), torch.floor(bb_y_min).int()
            bb_x_max, bb_y_max = torch.ceil(bb_x_max).int(), torch.ceil(bb_y_max).int()
            if debug:
                print(grid_points.shape, batch_triangles.shape)
            u, v, w = batched_barycentric_coord(grid_points, batch_triangles)
            if debug:
                print(u.shape, v.shape, w.shape)
            mask = (u >= 0) & (v >= 0) & (w >= 0)
            if debug:
                print(u.shape, batch_vertex_colors.shape)
            interpolated_color = u.unsqueeze(-1) * batch_vertex_colors[:, 0] + v.unsqueeze(-1) * \
                batch_vertex_colors[:, 1] + w.unsqueeze(-1) * batch_vertex_colors[:, 2]

            if True:
                if debug:
                    print("triangle depth", batch_depths.shape)
                interpolated_depth = u.unsqueeze(-1) * batch_depths[..., 0] + v.unsqueeze(-1) * \
                    batch_depths[..., 1] + w.unsqueeze(-1) * batch_depths[..., 2]
                print("interpolated depth", interpolated_depth.shape)
                interpolated_depth[~mask] = float('inf')
                closest_depth, closest_depth_idx = torch.min(interpolated_depth, dim=1)
                closer_mask = closest_depth < depth_buffer

                depth_buffer[closer_mask] = closest_depth[closer_mask]
                interpolated_color[~mask] = 0.
                if debug:
                    print("interpolated_color", interpolated_color.shape, "closest_depth_idx", closest_depth_idx.shape)

                closest_depth_idx = closest_depth_idx.unsqueeze(-1).expand(-1, -1, 3)
                interpolated_color = torch.gather(interpolated_color, 1, closest_depth_idx)
                interpolated_color = interpolated_color.squeeze(1)
                if debug:
                    print("interpolated_color!", interpolated_color.shape, closer_mask.shape)
                interpolated_color *= closer_mask

                image += interpolated_color.view(image.shape)

                if False:
                    # visualize inverse depth
                    image += (0.1/closest_depth).repeat(1, 3).view(image.shape)

            else:

                interpolated_color[~mask] = 0.
                interpolated_color = interpolated_color.sum(axis=1)
                if debug:
                    print(interpolated_color.shape, mask.shape)
                image += interpolated_color.view(image.shape)
            # No Z buffer!
    # if show_depth:
    # Debug : visualize inverse depth
    if show_depth:
        image = 0.5 / depth_buffer.repeat(1, 3).view(image.shape)
        image = (image - image.min()) / (image.max() - image.min())
    return image


def shade_screen_space_for_loop(
        cc_triangles: torch.Tensor,
        colors: torch.Tensor,
        depths: torch.Tensor,
        width: int, height: int,
        show_depth: bool = False,
        no_grad: bool = True,
        debug: bool = False
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
            depth_values = depths[batch_idx, 0, :]
            triangle = cc_triangles[batch_idx][:2, :]
            color = vertex_colors[batch_idx]
            (bb_x_min, bb_y_min), _ = torch.min(triangle, axis=-1)
            (bb_x_max, bb_y_max), _ = torch.max(triangle, axis=-1)
            bb_x_min, bb_y_min = torch.floor(bb_x_min).int(), torch.floor(bb_y_min).int()
            bb_x_max, bb_y_max = torch.ceil(bb_x_max).int(), torch.ceil(bb_y_max).int()

            # Clamp values to be within image dimensions
            bb_x_min, bb_y_min = max(min(bb_x_min, width - 1), 0), max(min(bb_y_min, height - 1), 0)
            bb_x_max, bb_y_max = max(min(bb_x_max, width - 1), 0), max(min(bb_y_max, height - 1), 0)
            # if debug:
            #     print("x", bb_x_min, bb_x_max)
            #     print("y", bb_y_min, bb_y_max)
            if bb_x_min >= bb_x_max or bb_y_min >= bb_y_max:
                if debug:
                    print("x", bb_x_min, bb_x_max, triangle)
                    print("y", bb_y_min, bb_y_max, triangle)
                continue  # Skip may be due ton infinity values
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
            mask = (u >= 0) & (v >= 0) & (w >= 0)

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
