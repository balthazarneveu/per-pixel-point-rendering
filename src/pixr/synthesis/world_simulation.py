import torch
from typing import Tuple
from pixr.synthesis.world_from_triangles import generate_3d_scene_sample_test_triangles, generate_3d_staircase_scene
from pixr.synthesis.world_from_mesh import generate_3d_scene_sample_from_mesh


def generate_3d_scene_sample_triangles(
    z: float = 5,
    delta_z: float = 0.,
    scene_mode="test_triangles"
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scene_mode == "test_triangles":
        wc_triangles, colors_nodes = generate_3d_scene_sample_test_triangles(z=z, delta_z=delta_z)
    elif scene_mode == "staircase":
        wc_triangles, colors_nodes = generate_3d_staircase_scene(z=z)
    return wc_triangles, colors_nodes


def generate_simulated_world(
    scene_mode="test_triangles",
    z: float = 5,
    delta_z: float = 0.,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scene_mode == "test_triangles":
        wc_triangles, colors_nodes = generate_3d_scene_sample_test_triangles(z=z, delta_z=delta_z)
    elif scene_mode == "staircase":
        wc_triangles, colors_nodes = generate_3d_staircase_scene(z=z)
    else:
        wc_triangles, colors_nodes = generate_3d_scene_sample_from_mesh(mesh_name=scene_mode, z=z)
    return wc_triangles, colors_nodes
