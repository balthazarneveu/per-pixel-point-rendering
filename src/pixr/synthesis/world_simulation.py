import torch
from typing import Tuple


def generate_3d_scene_sample_triangles(z: float = 5, delta_z: float = 0.) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a 3D scene with N triangles in world coordinates and their colors.

    Args:
        z (float, optional): The z-coordinate of the triangles. Defaults to 5.
        delta_z (float, optional): The change in z-coordinate for the second set of triangles. Defaults to 0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the world-coordinate triangles and their colors.
    """
    # [N, 3, xyz1]
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
            ],
            [
                [0.5, 0., -z, 1.],
                [0.5, 1., -z, 1.],
                [1.5, 1., -z, 1.]
            ],
        ]
    )
    # [N, xyz1, 3=triangle]
    wc_triangles = wc_triangles.permute(0, 2, 1)
    # Note: colors go above 1. to mimick very bright colors and check HDR rendering.

    colors_nodes = torch.Tensor(
        [
            [
                [0., 0.1, 1.],
                [0.1, 0., 1.],
                [1., 1., 2.]
            ],
            [
                [1., 1., 0.],
                [1., 0., 0.],
                [1., 0.5, 0.3]
            ],
            [
                [1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 0.]
            ]
        ]
    )
    # [N, 3 triangle, 3=rgb]

    return wc_triangles, colors_nodes
