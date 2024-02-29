import torch
from typing import Tuple
from interactive_pipe import interactive, interactive_pipeline
from interactive_pipe.data_objects.curves import Curve, SingleCurve
import numpy as np
from pixr.synthesis.forward_project import get_camera_intrinsics, get_camera_extrinsics, project_3d_to_2d
from pixr.camera.camera import linear_rgb_to_srgb

def set_camera_parameters(yaw_deg=0., pitch_deg=0., roll_deg=0., trans_x=0., trans_y=0., trans_z=0.) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    yaw = torch.deg2rad(torch.Tensor([yaw_deg]))
    pitch = torch.deg2rad(torch.Tensor([pitch_deg]))
    roll = torch.deg2rad(torch.Tensor([roll_deg]))
    cam_pos = torch.stack([torch.Tensor([trans_x]), torch.Tensor([trans_y]), torch.Tensor([trans_z])])
    return yaw, pitch, roll, cam_pos


def generate_3d_scene(z=5, delta_z=0.):
    # [N, 3, xyz]
    wc_triangles = torch.Tensor(
        [
            [
                [0., 0., z, 1.],
                [0., 1., z, 1.],
                [1., 1., z, 1.]
            ],
            [
                [-1., 0., z+delta_z, 1.],
                [2., 0., z+delta_z, 1.],
                [2., 1., z+delta_z, 1.]
            ]
        ]

    )
    wc_triangles = wc_triangles.permute(0, 2, 1)
    # colors_nodes = torch.Tensor(
    #     [
    #         [
    #             [1., 0., 0.],
    #             [0., 1., 0.],
    #             [0., 0., 1.]
    #         ],
    #         [
    #             [1., 1., 0.],
    #             [0., 1., 1.],
    #             [1., 0., 1.]
    #         ]
    #     ]
    # )
    colors_nodes = torch.Tensor(
        [
            [
                [0., 0.1, 1.],
                [0.1, 0., 1.],
                # [0., 0.3, 1.]
                [1., 1., 2.]
            ],
            [
                [1., 1., 0.],
                [1., 0., 0.],
                [1., 0.5, 0.3]
            ]
        ]
    )

    return wc_triangles, colors_nodes


def visualize_2d_scene(cc_triangles: torch.Tensor, w, h) -> Curve:
    cc_triangles = cc_triangles.numpy()
    t1 = SingleCurve(
        x=[cc_triangles[0, 0, idx] for idx in [0, 1, 2, 0]],
        y=[cc_triangles[0, 1, idx] for idx in [0, 1, 2, 0]],
        style="bo"
    )
    t2 = SingleCurve(
        x=[cc_triangles[1, 0, idx] for idx in [0, 1, 2, 0]],
        y=[cc_triangles[1, 1, idx] for idx in [0, 1, 2, 0]],
        style="ro"
    )
    corners = SingleCurve(
        x=[0, 0, w, w, 0],
        y=[0, h, h, 0, 0],
        style="k-"
    )
    center = SingleCurve(
        x=[w/2],
        y=[h/2],
        style="g+",
        markersize=10
    )
    img_scene = Curve(
        [t1, t2, center, corners],
        xlim=[0, w-1],
        ylim=[h-1, 0],
        grid=True,
        xlabel="x",
        ylabel="y",
        title="Projected points")
    return img_scene


def splat_points(cc_triangles: torch.Tensor, colors: torch.Tensor, w: int, h: int, debug=True) -> np.ndarray:
    # Create an empty image with shape (h, w, 3)
    image = torch.zeros((h, w, 3))
    # Get the number of vertices
    num_vertices = cc_triangles.shape[1]

    # Extract the colors at the vertices
    vertex_colors = colors[:, :num_vertices, :]

    # Perform splatting of vertex colors
    for batch_idx in range(cc_triangles.shape[0]):
        triangle = cc_triangles[batch_idx]
        color = vertex_colors[batch_idx]
        for node_idx in range(triangle.shape[1]):
            x, y = torch.round(triangle[0, node_idx]).long(), torch.round(triangle[1, node_idx]).long()
            if 0 <= x < w and 0 <= y < h:
                if debug:
                    # DEBUG!
                    for u in range(-4, 4):
                        for v in range(-4, 4):
                            if 0 <= x+u < w and 0 <= y+v < h:
                                image[y+v, x+u] = color[node_idx]
                image[y, x] = color[node_idx]
    # Return the splatted image
    return image


def tensor_to_image(image: torch.Tensor) -> np.ndarray:
    image = image.numpy()
    return image


def barycentric_coords(p, a, b, c):
    """
    Compute barycentric coordinates of point p with respect to triangle (a, b, c).
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = torch.dot(v0, v0)
    d01 = torch.dot(v0, v1)
    d11 = torch.dot(v1, v1)
    d20 = torch.dot(v2, v0)
    d21 = torch.dot(v2, v1)
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
    print(v0.shape, v1.shape, v2.shape)

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

# def barycentric_coords(p, a, b, c):
#     """
#     Compute barycentric coordinates of point p with respect to triangle (a, b, c).
#     """

#     v0 = b - a
#     v1 = c - a
#     v2 = p - a
#     print(v0.shape, v1.shape, v2.shape)
#     # Reshape v0 and v1 for matrix multiplication
#     v0 = v0.permute(2, 0, 1)  # Shape: [2, 1, 1]
#     v1 = v1.permute(2, 0, 1)  # Shape: [2, 1, 1]

#     # Use matmul for dot products, handling broadcasting
#     # Flatten v2 for matmul, then reshape to original grid shape for further calculations
#     d00 = torch.matmul(v0.permute(1, 2, 0), v0).squeeze()
#     d01 = torch.matmul(v0.permute(1, 2, 0), v1).squeeze()
#     d11 = torch.matmul(v1.permute(1, 2, 0), v1).squeeze()
#     d20 = torch.matmul(v2, v0).squeeze(-1)
#     d21 = torch.matmul(v2, v1).squeeze(-1)

#     denom = d00 * d11 - d01 * d01
#     v = (d11 * d20 - d01 * d21) / denom
#     w = (d00 * d21 - d01 * d20) / denom
#     u = 1.0 - v - w

#     return u, v, w


def fill_triangle(image, vertices, color):
    """
    Fill a triangle in the image with the given color.
    vertices: Coordinates of the triangle's vertices.
    color: Color to fill the triangle.
    """
    # Extract vertices
    a, b, c = vertices
    # Compute bounding box of the triangle
    x_min, y_min = torch.min(vertices, dim=-1)
    x_max, y_max = torch.max(vertices, dim=-1)
    print(x_min, y_min, x_max, y_max)
    # Iterate over the bounding box
    for x in range(int(x_min), int(x_max) + 1):
        for y in range(int(y_min), int(y_max) + 1):
            p = torch.tensor([x, y], dtype=torch.float32)
            u, v, w = barycentric_coords(p, a, b, c)
            if u >= 0 and v >= 0 and w >= 0:  # Point is inside the triangle
                # Interpolate color
                interpolated_color = u * color[0] + v * color[1] + w * color[2]
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    image[y, x] = interpolated_color


@interactive(
    show_depth=(False,)
)
def render_textures(cc_triangles: torch.Tensor, colors: torch.Tensor, depths: torch.Tensor, width: int, height: int, show_depth: bool = False) -> np.ndarray:
    # Create an empty image with shape (h, w, 3)
    image = torch.zeros((height, width, 3))
    depth_buffer = torch.full((height, width), float('inf'), dtype=torch.float32)
    # Get the number of vertices
    num_vertices = cc_triangles.shape[1]
    # Extract the colors at the vertices
    vertex_colors = colors[:, :num_vertices, :]
    # Perform splatting of vertex colors
    for batch_idx in range(cc_triangles.shape[0]):
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
        print("x", bb_x_min, bb_x_max)
        print("y", bb_y_min, bb_y_max)
        # Create a grid for the bounding box
        x_range = torch.arange(bb_x_min, bb_x_max + 1, dtype=torch.float32)
        y_range = torch.arange(bb_y_min, bb_y_max + 1, dtype=torch.float32)

        grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='xy')
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
            image[bb_y_min:bb_y_max+1, bb_x_min:bb_x_max+1][closer_depth_mask] = interpolated_color[closer_depth_mask]
        else:
            # Debug : visualize inverse depth
            image[bb_y_min:bb_y_max+1, bb_x_min:bb_x_max+1][closer_depth_mask] = 5. / \
                interpolated_depth[closer_depth_mask].unsqueeze(-1)
        depth_buffer[bb_y_min:bb_y_max+1, bb_x_min:bb_x_max +
                     1][closer_depth_mask] = interpolated_depth[closer_depth_mask]

    return image


def projection_pipeline():
    wc_triangles, colors = generate_3d_scene()
    yaw, pitch, roll, cam_pos = set_camera_parameters()
    camera_extrinsics = get_camera_extrinsics(yaw, pitch, roll, cam_pos)
    camera_intrinsics, w, h = get_camera_intrinsics()
    cc_triangles, depths = project_3d_to_2d(wc_triangles, camera_intrinsics, camera_extrinsics)
    # Screen space triangles.
    rendered_image = render_textures(cc_triangles, colors, depths, w, h)
    rendered_image = linear_rgb_to_srgb(rendered_image)
    rendered_image = tensor_to_image(rendered_image)
    # Let's splat the triangle nodes
    splatted_image = splat_points(cc_triangles, colors, w, h)
    splatted_image = tensor_to_image(splatted_image)
    img_scene = visualize_2d_scene(cc_triangles, w, h)
    return img_scene, splatted_image, rendered_image


def main():
    interactive(
        z=(10., (2., 100.)),
        delta_z=(0.01, (-5., 5.))
    )(generate_3d_scene)
    interactive(
        yaw_deg=(0., (-180., 180.)),
        pitch_deg=(0., (-180., 180.)),
        roll_deg=(0., (-180., 180.)),
        trans_x=(0., (-10., 10.)),
        trans_y=(0., (-10., 10.)),
        trans_z=(0., (-10., 10.))
    )(set_camera_parameters)
    interactive_pipeline(gui="mpl", size=(10, 10))(projection_pipeline)()


if __name__ == '__main__':
    main()
