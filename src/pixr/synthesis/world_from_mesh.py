import torch
import trimesh
from pathlib import Path
from typing import Tuple
from pixr.properties import DEVICE, MESH_PATH


def generate_3d_scene_sample_from_mesh(
        mesh_name: str = "teapot",
        mesh_path: Path = MESH_PATH,
        z: float = 5,
        device: str = DEVICE) -> Tuple[torch.Tensor, torch.Tensor]:
    # Load the mesh
    mesh = trimesh.load((MESH_PATH/mesh_name).with_suffix(".obj"), force='mesh')

    # Ensure the mesh is a Trimesh object
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("The loaded object is not a Trimesh instance.")

    # Extract vertices and faces from the mesh
    vertices = mesh.vertices
    faces = mesh.faces

    # Convert vertices to PyTorch tensor and add z-coordinate and homogenous coordinate
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32)
    vertices_tensor = torch.cat([vertices_tensor, torch.full((vertices_tensor.shape[0], 1), z)], dim=1)
    # vertices_tensor = torch.cat([vertices_tensor, torch.ones((vertices_tensor.shape[0], 1))],
    #                             dim=1)  # Add homogenous coordinate

    # Create tensor for triangle vertices in world coordinates
    wc_triangles = torch.stack([vertices_tensor[faces[:, i]] for i in range(3)], dim=1)

    # Generate random colors for each triangle
    colors_nodes = torch.rand(faces.shape[0], 3, 3)  # Each vertex of each triangle gets a color (RGB)
    wc_triangles = wc_triangles.permute(0, 2, 1)
    center = wc_triangles.mean(dim=(0, -1), keepdim=True)
    center[..., -1, :] = 0.
    wc_triangles -= center
    scale = wc_triangles[..., :3, :].std(dim=(0, 1, 2), keepdim=True)
    # print("center", center)
    # print("scale", scale)
    wc_triangles *= 0.3/scale  # 30 cm size
    wc_triangles[..., 2, :] += z
    wc_triangles = wc_triangles.to(device)
    colors_nodes = colors_nodes.to(device)
    return wc_triangles, colors_nodes
