import torch
import argparse
import sys
from interactive_pipe import interactive_pipeline
from interactive_pipe.data_objects.curves import Curve, SingleCurve
import numpy as np
from pixr.camera.camera_geometry import get_camera_intrinsics, get_camera_extrinsics
from pixr.synthesis.forward_project import project_3d_to_2d
from pixr.camera.camera import linear_rgb_to_srgb
from pixr.rasterizer.rasterizer import shade_screen_space
from pixr.camera.camera_geometry import set_camera_parameters
from pixr.synthesis.normals import extract_normals
from pixr.synthesis.world_simulation import generate_simulated_world
from pixr.synthesis.extract_point_cloud import pick_point_cloud_from_triangles
from interactive_plugins import define_default_sliders
from pixr.rendering.splatting import splat_points
import cv2


def visualize_2d_scene(cc_triangles: torch.Tensor, w, h) -> Curve:
    cc_triangles = cc_triangles.cpu().numpy()
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
    image = image.cpu().numpy()
    return image


def rescale_image(image: np.ndarray, global_params={}) -> np.ndarray:
    scale = global_params.get('scale', 0)
    if scale > 0:
        factor = 2**scale
        image = cv2.resize(image, (0, 0), fx=factor, fy=factor)
    return image


def projection_pipeline():
    wc_triangles, colors = generate_simulated_world()
    # wc_triangles, colors = generate_3d_scene_sample_from_mesh()
    wc_normals = extract_normals(wc_triangles)
    wc_points, points_colors, wc_normals = pick_point_cloud_from_triangles(wc_triangles, colors, wc_normals)
    yaw, pitch, roll, cam_pos = set_camera_parameters()
    camera_extrinsics = get_camera_extrinsics(yaw, pitch, roll, cam_pos)
    camera_intrinsics, w, h = get_camera_intrinsics()
    cc_points, points_depths, cc_normals = project_3d_to_2d(
        wc_points, camera_intrinsics, camera_extrinsics, wc_normals)
    cc_triangles, triangles_depths, _ = project_3d_to_2d(wc_triangles, camera_intrinsics, camera_extrinsics, None)
    # Screen space triangles.
    rendered_image = shade_screen_space(cc_triangles, colors, triangles_depths, w, h)
    rendered_image = linear_rgb_to_srgb(rendered_image)
    rendered_image = tensor_to_image(rendered_image)
    # Let's splat the triangle nodes
    splatted_image = splat_points(cc_points, points_colors, points_depths, w, h, camera_intrinsics, cc_normals)
    # splatted_image = splat_points(cc_triangles, colors, w, h)
    splatted_image = tensor_to_image(splatted_image)
    splatted_image = rescale_image(splatted_image)
    # img_scene = visualize_2d_scene(cc_triangles, w, h)
    # return img_scene, splatted_image, rendered_image

    return splatted_image, rendered_image


def splat_pipeline():
    wc_triangles, colors = generate_simulated_world()
    # wc_triangles, colors = generate_3d_scene_sample_from_mesh()
    wc_normals = extract_normals(wc_triangles)
    wc_points, points_colors, wc_normals = pick_point_cloud_from_triangles(wc_triangles, colors, wc_normals)
    yaw, pitch, roll, cam_pos = set_camera_parameters()
    camera_extrinsics = get_camera_extrinsics(yaw, pitch, roll, cam_pos)
    camera_intrinsics, w, h = get_camera_intrinsics()
    cc_points, points_depths, cc_normals = project_3d_to_2d(
        wc_points, camera_intrinsics, camera_extrinsics, wc_normals)

    # Let's splat the triangle nodes
    splatted_image = splat_points(cc_points, points_colors, points_depths, w, h, camera_intrinsics, cc_normals)
    splatted_image = tensor_to_image(splatted_image)

    return splatted_image


def main(argv):
    arg = argparse.ArgumentParser()
    arg.add_argument('-r', '--rasterizer', help="Rasterize the scene", action="store_true")
    args = arg.parse_args(argv)
    rasterizer_flag = args.rasterizer

    import logging
    logging.basicConfig(level=logging.INFO)
    define_default_sliders()

    interactive_pipeline(
        gui="qt",
        cache=True,
        safe_input_buffer_deepcopy=False,
        size=(20, 10)
    )(projection_pipeline if rasterizer_flag else splat_pipeline)()


if __name__ == '__main__':
    main(sys.argv[1:])
