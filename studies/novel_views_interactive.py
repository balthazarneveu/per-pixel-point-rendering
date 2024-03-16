from pathlib import Path
from pixr.camera.camera_geometry import get_camera_extrinsics, get_camera_intrinsics, set_camera_parameters_orbit_mode
from interactive_pipe.data_objects.image import Image
from pixr.rendering.splatting import splat_points
from interactive_plugins import define_default_sliders
from novel_views import load_colored_point_cloud_from_files
from pixr.synthesis.forward_project import project_3d_to_2d
from interactive_pipe import interactive_pipeline
from pixr.interactive.utils import tensor_to_image, rescale_image


def splat_pipeline_novel_view(wc_points, wc_normals, points_colors):
    # yaw, pitch, roll, cam_pos = set_camera_parameters()
    yaw, pitch, roll, cam_pos = set_camera_parameters_orbit_mode()
    camera_extrinsics = get_camera_extrinsics(yaw, pitch, roll, cam_pos)
    camera_intrinsics, w, h = get_camera_intrinsics()
    cc_points, points_depths, cc_normals = project_3d_to_2d(
        wc_points, camera_intrinsics, camera_extrinsics, wc_normals)

    # # Let's splat the triangle nodes
    splatted_image = splat_points(cc_points, points_colors, points_depths, w, h, camera_intrinsics, cc_normals)
    splatted_image = tensor_to_image(splatted_image)
    splatted_image = rescale_image(splatted_image)
    return splatted_image


def main_interactive_version(splat_scene_path):
    wc_points, wc_normals, color_pred = load_colored_point_cloud_from_files(splat_scene_path)
    define_default_sliders(orbit_mode=True)
    interactive_pipeline(
        gui="qt",
        cache=True,
        safe_input_buffer_deepcopy=False,
        size=(20, 10)
    )(splat_pipeline_novel_view)(wc_points, wc_normals, color_pred)


if __name__ == '__main__':
    splat_scene_path = Path(
        "/Data/code/per-pixel-point-rendering/__output/staircase_splat_differentiate_points/checkpoint_00200.pt")
    assert splat_scene_path.exists()
    main_interactive_version(splat_scene_path)
