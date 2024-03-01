import torch
from interactive_pipe import interactive, interactive_pipeline
from interactive_pipe.data_objects.curves import Curve, SingleCurve
import numpy as np
from pixr.camera.camera_geometry import get_camera_intrinsics, get_camera_extrinsics
from pixr.synthesis.forward_project import project_3d_to_2d
from pixr.camera.camera import linear_rgb_to_srgb
from pixr.synthesis.shader import shade_screen_space
from pixr.camera.camera_geometry import set_camera_parameters
from pixr.synthesis.world_simulation import generate_3d_scene_sample_triangles


def pick_point_cloud_from_triangles(
        wc_triangles: torch.Tensor,
        colors: torch.Tensor,
        num_samples: int = 100
) -> torch.Tensor:
    """Pick m samples from the triangles in the 3D scene.

    Args:
        wc_triangles (torch.Tensor): [N, 4=xyz1, 3] triangles
        colors (torch.Tensor): [N, 3, 3=rgb] colors at the vertices
        num_samples (int, optional): number of samples to pick. Defaults to 100.

    Returns:
        torch.Tensor: [m, 4=xyz1, 1] point cloud
        torch.Tensor: [m, 1, 3=rgb] point colors
    """
    print(colors.shape)
    # Calculate the vectors forming the sides of the triangles
    vec0 = wc_triangles[:, :3, 1] - wc_triangles[:, :3, 0]
    vec1 = wc_triangles[:, :3, 2] - wc_triangles[:, :3, 0]
    cross_product = torch.cross(vec0, vec1, dim=1)

    # Calculate the area of each triangle for weighted sampling (using cross product)
    areas = 0.5 * torch.norm(cross_product, dim=1)
    total_area = torch.sum(areas)
    probabilities = areas / total_area
    # Probabilities shall be 0.25, 0.75 for the 2 triangles we're considering
    # Sample triangle indices based on their area
    sampled_indices = torch.multinomial(probabilities, num_samples, replacement=True)

    # Sample points within the selected triangles
    sampled_points = torch.zeros(num_samples, 4, 1)  # Initialize tensor for sampled points
    sampled_colors = torch.zeros(num_samples, 1, 3)  # Initialize tensor for sampled colors
    for i, idx in enumerate(sampled_indices):
        # Barycentric coordinates for a random point within a triangle
        r1, r2 = torch.sqrt(torch.rand(1)), torch.rand(1)
        barycentric = torch.tensor([1 - r1, r1 * (1 - r2), r1 * r2])

        # Convert barycentric coordinates to Cartesian coordinates
        point = torch.matmul(wc_triangles[idx, :3, :], barycentric.unsqueeze(1))
        point_homogeneous = torch.cat((point, torch.tensor([[1.0]])))  # Convert to homogeneous coordinates

        sampled_points[i] = point_homogeneous

        # Interpolate colors based on barycentric coordinates
        # color = torch.einsum('ij,j->i', colors[idx], barycentric)  # Efficient matrix-vector multiplication
        color = torch.einsum('ij,j->i', colors[idx].T, barycentric)  # Efficient matrix-vector multiplication
        sampled_colors[i, 0, :] = color

    return sampled_points, sampled_colors


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


def splat_points(
    cc_triangles: torch.Tensor, colors: torch.Tensor,
    w: int,
    h: int,
    debug=False,
    no_grad: bool = False
) -> np.ndarray:
    # Create an empty image with shape (h, w, 3)
    image = torch.zeros((h, w, 3))
    # Get the number of vertices
    num_vertices = cc_triangles.shape[1]

    # Extract the colors at the vertices
    vertex_colors = colors[:, :num_vertices, :]
    with torch.no_grad() if no_grad else torch.enable_grad():
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


def projection_pipeline():
    wc_triangles, colors = generate_3d_scene_sample_triangles()
    wc_points, points_colors = pick_point_cloud_from_triangles(wc_triangles, colors)
    yaw, pitch, roll, cam_pos = set_camera_parameters()
    camera_extrinsics = get_camera_extrinsics(yaw, pitch, roll, cam_pos)
    camera_intrinsics, w, h = get_camera_intrinsics()
    cc_points, points_depths = project_3d_to_2d(wc_points, camera_intrinsics, camera_extrinsics)
    cc_triangles, triangles_depths = project_3d_to_2d(wc_triangles, camera_intrinsics, camera_extrinsics)
    # Screen space triangles.
    rendered_image = shade_screen_space(cc_triangles, colors, triangles_depths, w, h)
    rendered_image = linear_rgb_to_srgb(rendered_image)
    rendered_image = tensor_to_image(rendered_image)
    # Let's splat the triangle nodes
    splatted_image = splat_points(cc_points, points_colors, w, h)
    # splatted_image = splat_points(cc_triangles, colors, w, h)
    splatted_image = tensor_to_image(splatted_image)
    img_scene = visualize_2d_scene(cc_triangles, w, h)
    return img_scene, splatted_image, rendered_image


def main():
    import logging
    logging.basicConfig(level=logging.INFO)
    interactive(
        z=(10., (2., 100.)),
        delta_z=(0.01, (-5., 5.))
    )(generate_3d_scene_sample_triangles)
    interactive(
        yaw_deg=(0., (-180., 180.)),
        pitch_deg=(0., (-180., 180.)),
        roll_deg=(0., (-180., 180.)),
        trans_x=(0., (-10., 10.)),
        trans_y=(0., (-10., 10.)),
        trans_z=(0., (-10., 10.))
    )(set_camera_parameters)
    interactive(show_depth=(False,))(shade_screen_space)
    interactive_pipeline(
        gui="qt", cache=True,
        safe_input_buffer_deepcopy=False,
        size=(20, 10)
    )(projection_pipeline)()


if __name__ == '__main__':
    main()
