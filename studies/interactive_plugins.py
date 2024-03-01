from interactive_pipe import interactive
from pixr.camera.camera import linear_rgb_to_srgb
from pixr.camera.camera_geometry import set_camera_parameters
from pixr.synthesis.world_simulation import generate_3d_scene_sample_triangles
from pixr.synthesis.extract_point_cloud import pick_point_cloud_from_triangles
from pixr.synthesis.shader import shade_screen_space
from pixr.synthesis.world_from_mesh import generate_3d_scene_sample_from_mesh
from pixr.properties import MESH_PATH


def define_default_sliders():
    interactive(
        exposure=(1., [0., 5.]),
        gamma=(2.2, [1., 4.]),
    )(linear_rgb_to_srgb)
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
    interactive(
        num_samples=(100, [100, 10000])
    )(pick_point_cloud_from_triangles)
    interactive(
        mesh_name=("teapot", [pth.stem for pth in MESH_PATH.glob("*.obj")]),
    )(generate_3d_scene_sample_from_mesh)
    interactive(show_depth=(False,))(shade_screen_space)
