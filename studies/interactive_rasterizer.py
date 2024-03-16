from interactive_pipe import interactive_pipeline
from pixr.camera.camera_geometry import get_camera_intrinsics, get_camera_extrinsics
from pixr.synthesis.forward_project import project_3d_to_2d
from pixr.camera.camera import linear_rgb_to_srgb
from pixr.rasterizer.rasterizer import shade_screen_space
from pixr.camera.camera_geometry import set_camera_parameters
from interactive_plugins import define_default_sliders
# from pixr.synthesis.world_from_mesh import generate_3d_scene_sample_from_mesh
from pixr.synthesis.world_simulation import generate_simulated_world
from pixr.interactive.utils import tensor_to_image


def projection_pipeline():
    wc_triangles, colors = generate_simulated_world()
    yaw, pitch, roll, cam_pos = set_camera_parameters()
    camera_extrinsics = get_camera_extrinsics(yaw, pitch, roll, cam_pos)
    camera_intrinsics, w, h = get_camera_intrinsics()
    cc_triangles, triangles_depths, cc_normals = project_3d_to_2d(
        wc_triangles, camera_intrinsics, camera_extrinsics, None)
    # Screen space triangles.
    rendered_image = shade_screen_space(cc_triangles, colors, triangles_depths, w, h)
    rendered_image = linear_rgb_to_srgb(rendered_image)
    rendered_image = tensor_to_image(rendered_image)
    return rendered_image


def main():
    import logging
    logging.basicConfig(level=logging.INFO)
    define_default_sliders()
    interactive_pipeline(
        gui="qt", cache=True,
        safe_input_buffer_deepcopy=False,
        size=(20, 10)
    )(projection_pipeline)()


if __name__ == '__main__':
    main()
