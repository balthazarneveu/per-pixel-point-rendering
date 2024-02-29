import torch
from typing import Tuple
from interactive_pipe import interactive, interactive_pipeline
from interactive_pipe.data_objects.curves import Curve, SingleCurve
import numpy as np
from pixr.synthesis.forward_project import get_camera_intrinsics, get_camera_extrinsics, project_3d_to_2d
from pixr.camera.camera import linear_rgb_to_srgb
from pixr.synthesis.shader import shade_screen_space

from pixr.camera.camera_geometry import set_camera_parameters


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


def projection_pipeline():
    wc_triangles, colors = generate_3d_scene()
    yaw, pitch, roll, cam_pos = set_camera_parameters()
    camera_extrinsics = get_camera_extrinsics(yaw, pitch, roll, cam_pos)
    camera_intrinsics, w, h = get_camera_intrinsics()
    cc_triangles, depths = project_3d_to_2d(wc_triangles, camera_intrinsics, camera_extrinsics)
    # Screen space triangles.
    rendered_image = shade_screen_space(cc_triangles, colors, depths, w, h)
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
    interactive(show_depth=(False,))(shade_screen_space)
    interactive_pipeline(gui="mpl", size=(10, 10))(projection_pipeline)()


if __name__ == '__main__':
    main()
