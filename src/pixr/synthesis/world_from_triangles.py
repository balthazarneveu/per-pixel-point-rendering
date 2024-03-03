import torch
from typing import Tuple
import numpy as np

# Function to generate rainbow colors


def rainbow_color(step):
    return [
        np.abs(np.sin(0.3 * step + 0)),
        np.abs(np.sin(0.3 * step + 2 * np.pi / 3)),
        np.abs(np.sin(0.3 * step + 4 * np.pi / 3))
    ]


def generate_rect(
    z: float = 0.,
    delta_z: float = 0.,
) -> Tuple[torch.Tensor, torch.Tensor]:
    wc_triangles = []
    colors_nodes = []

    current_z = z
    wc_triangles.extend([
        [
            [-1, -1, current_z+delta_z, 1.],
            [1, -1, current_z+delta_z, 1.],
            [1, 1, current_z+delta_z, 1.]
            # ],
        ][::-1],
        [
            [1, 1, current_z, 1.],
            [-1, 1, current_z, 1.],
            [-1, -1, current_z, 1.]
        ][::-1]
    ])

    for shft in range(2):  # Add the same color for the four triangles that form a step
        colors_nodes.append([rainbow_color(1+3*shft), rainbow_color(2+3*shft), rainbow_color(3+3*shft)])

    wc_triangles = torch.Tensor(wc_triangles)
    wc_triangles = wc_triangles.permute(0, 2, 1)  # Adjusting dimensions to match your format
    colors_nodes = torch.Tensor(colors_nodes)

    return wc_triangles, colors_nodes


def generate_3d_scene_sample_test_triangles(z: float = 0, delta_z: float = 0.) -> Tuple[torch.Tensor, torch.Tensor]:
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
            ][::-1],
            [
                [-1., 1., z+delta_z, 1.],
                [-1., 0., z+delta_z, 1.],
                [2., 1., z+delta_z, 1.]
            ],  # BACKFACING THE CAMERA!
            [
                [0.5, 0., -z - 10., 1.],
                [0.5, 1., -z - 10., 1.],
                [1.5, 1., -z - 10., 1.]
            ][::-1],  # BEHIND CAMERA!
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
            ][::-1],
            [
                [1., 1., 0.],
                [1., 0., 0.],
                [1., 0.5, 0.3]
            ][::-1],
            [
                [1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 0.]
            ]
        ]
    )
    # [N, 3 triangle, 3=rgb]

    return wc_triangles, colors_nodes


def generate_3d_staircase_scene(
    num_steps: int = 5,
    step_size: Tuple[float, float, float] = (0.5, 5, 0.5),
    z: float = 0.,
    delta_z: float = 0.,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a 3D scene with a staircase made of rectangles (each represented by two triangles)
    in world coordinates and their colors.

    Args:
        num_steps (int, optional): Number of steps in the staircase.
        step_size (Tuple[float, float, float], optional): Size of each step in x, y, and z directions.
        z (float, optional): The starting z-coordinate of the staircase.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing
        - world-coordinate triangles of the staircase
        - vertices colors.
    """
    step_x, step_y, step_z = step_size
    step_z = step_z + delta_z
    wc_triangles = []
    colors_nodes = []
    offset_y = 1.

    current_z = z
    for i in range(num_steps):
        color = np.array(rainbow_color(i))
        x_stair_end = i * step_x + step_x
        if True:
            # Original step
            wc_triangles.extend([
                [
                    [i * step_x, -step_y/2.+offset_y, current_z, 1.],
                    [x_stair_end, -step_y/2.+offset_y, current_z, 1.],
                    [i * step_x, step_y/2.+offset_y, current_z, 1.]
                ][::-1],
                [
                    [x_stair_end, -step_y/2.+offset_y, current_z, 1.],
                    [x_stair_end, step_y/2.+offset_y, current_z, 1.],
                    [i * step_x, step_y/2.+offset_y, current_z, 1.]
                ][::-1]
            ])

            for _ in range(2):  # Add the same color for the four triangles that form a step
                colors_nodes.append([color, color, color])

        # Perpendicular step
        wc_triangles.extend([
            [
                [x_stair_end, -step_y/2.+offset_y, current_z + step_z, 1.],
                [x_stair_end, -step_y/2.+offset_y, current_z, 1.],
                [x_stair_end, step_y/2.+offset_y, current_z + step_z, 1.]
            ],  # [::-1],
            [
                [x_stair_end, -step_y/2.+offset_y, current_z, 1.],
                [x_stair_end, step_y/2.+offset_y, current_z, 1.],
                [x_stair_end, step_y/2.+offset_y, current_z + step_z, 1.]
            ]  # [::-1]
        ])
        for _ in range(2):
            colors_nodes.append([1.-color/2., 1.-color/2., 1.-color/2.])

        current_z += delta_z

    wc_triangles = torch.Tensor(wc_triangles)
    wc_triangles = wc_triangles.permute(0, 2, 1)  # Adjusting dimensions to match your format
    colors_nodes = torch.Tensor(colors_nodes)

    return wc_triangles, colors_nodes
