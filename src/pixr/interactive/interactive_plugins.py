from interactive_pipe import interactive
from pixr.camera.camera import linear_rgb_to_srgb
from pixr.camera.camera_geometry import set_camera_parameters, set_camera_parameters_orbit_mode
from pixr.synthesis.extract_point_cloud import pick_point_cloud_from_triangles
from pixr.rasterizer.rasterizer import shade_screen_space
from pixr.rendering.splatting import splat_points, ms_splatting
from pixr.synthesis.world_simulation import generate_simulated_world, ALL_SCENE_MODES, STAIRCASE
from pixr.properties import MESH_PATH


def define_default_sliders(orbit_mode=False, multiscale=None, max_samples=10000, default_num_samples=100, default_scene=STAIRCASE):
    interactive(
        exposure=(1., [0., 5.]),
        gamma=(2.2, [1., 4.]),
    )(linear_rgb_to_srgb)
    if orbit_mode:
        interactive(
            yaw_deg=(0., [-360., 360.]),
            pitch_deg=(0., [-360., 360.]),
            roll_deg=(0., [-180., 180.]),
            trans_x=(0., [-10., 10.]),
            trans_y=(0., [-10., 10.]),
            trans_z=(13.741, [-30., 50.])
        )(set_camera_parameters_orbit_mode)
    else:
        interactive(
            yaw_deg=(0., [-180., 180.]),
            pitch_deg=(0., [-180., 180.]),
            roll_deg=(0., [-180., 180.]),
            trans_x=(0., [-10., 10.]),
            trans_y=(0., [-10., 10.]),
            trans_z=(13.741, [-30., 50.])
        )(set_camera_parameters)

    interactive(
        num_samples=(default_num_samples, [100, max_samples])
    )(pick_point_cloud_from_triangles)
    interactive(
        z=(0., [-10., 10.]),
        delta_z=(2., [-5., 5.]),
        scene_mode=(default_scene,
                    ALL_SCENE_MODES +
                    [pth.stem for pth in MESH_PATH.glob("*.obj")]),
        normalize=(False,),
    )(generate_simulated_world)
    interactive(
        show_depth=(False,),
        # for_loop=(True,),
        # limit=(-1, [-1, 10000]),
    )(shade_screen_space)
    if multiscale is not None:
        interactive(
            scale=(0, [0, multiscale-1]),
            z_buffer_flag=(True,),
            normal_culling_flag=(True,),
            fuzzy_depth_test=(0.01, [0., 0.1]),
        )(ms_splatting)
    else:
        interactive(
            debug=(False,),
            z_buffer_flag=(True,),
            normal_culling_flag=(True,),
            fuzzy_depth_test=(0.01, [0., 0.1]),
            scale=(0, [0, 5]),
            # for_loop_zbuffer=(False,),
        )(splat_points)
