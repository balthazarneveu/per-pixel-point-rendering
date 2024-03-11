import torch
from typing import Tuple
from pixr.synthesis.world_from_triangles import generate_3d_scene_sample_test_triangles, generate_3d_staircase_scene, generate_rect
from pixr.synthesis.world_from_mesh import generate_3d_scene_sample_from_mesh

TEST_RECT = "test_rect"
TEST_TRIANGLES = "test_triangles"
STAIRCASE = "staircase"
ALL_SCENE_MODES = [TEST_RECT, TEST_TRIANGLES, STAIRCASE]


def generate_simulated_world(
    scene_mode=TEST_RECT,
    z: float = 0,
    delta_z: float = 2.,
    normalize: bool = False,
    invert_z_axis: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scene_mode == TEST_RECT:
        wc_triangles, colors_nodes = generate_rect(z=z, delta_z=delta_z)
    elif scene_mode == TEST_TRIANGLES:
        wc_triangles, colors_nodes = generate_3d_scene_sample_test_triangles(z=z, delta_z=delta_z)
    elif scene_mode == STAIRCASE:
        wc_triangles, colors_nodes = generate_3d_staircase_scene(num_steps=5, z=z, delta_z=delta_z)
    else:
        wc_triangles, colors_nodes = generate_3d_scene_sample_from_mesh(mesh_name=scene_mode, z=z, normalize=normalize)
    if invert_z_axis:
        # DISABLE WHEN EXPORTING TO OBJ FOR BLENDER!
        wc_triangles[..., 2, :] *= -1
    return wc_triangles, colors_nodes
