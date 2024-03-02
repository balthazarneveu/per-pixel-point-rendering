import torch
from tqdm import tqdm


def batched_barycentric_coord(p: torch.Tensor, triangles):
    triangles = triangles.unsqueeze(0)  # .repeat(p.shape[0], 1, 1, 1)
    p = p.squeeze(-1)
    p = p.unsqueeze(1)
    # print("triangles", triangles.shape, "points", p.shape)
    a = triangles[..., 0]
    b = triangles[..., 1]
    c = triangles[..., 2]
    # print("triangle edge", a.shape, "points", p.shape)
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


def shade_screen_space_parallel(
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
            if limit > 0 and batch_idx*batch_size > limit:
                break
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
            # interpolated_color *= closer_mask
            interpolated_color[~closer_mask.expand(-1, 3)] = 0.

            image += interpolated_color.view(image.shape)

            # No Z buffer!
    # Debug : visualize inverse depth
    if show_depth:
        image = 0.5 / depth_buffer.repeat(1, 3).view(image.shape)
        image = (image - image.min()) / (image.max() - image.min())
    return image
