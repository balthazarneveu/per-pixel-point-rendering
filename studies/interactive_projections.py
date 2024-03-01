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
from pixr.synthesis.extract_point_cloud import pick_point_cloud_from_triangles
from pixr.rendering.splatting import splat_points


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
