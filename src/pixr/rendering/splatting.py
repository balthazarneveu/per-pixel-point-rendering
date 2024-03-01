import torch


def splat(cc_triangles: torch.Tensor, colors: torch.Tensor, w: int, h: int):
    # Create an empty image with shape (h, w, 3)
    image = torch.zeros((h, w, 3))

    # Iterate over each triangle
    for i in range(cc_triangles.shape[0]):
        triangle = cc_triangles[i]

        # Get the color of the triangle
        color = colors[i]

        # Iterate over each pixel in the triangle
        for x in range(w):
            for y in range(h):
                # Check if the pixel is inside the triangle
                if point_in_triangle(x, y, triangle):
                    # Set the pixel value to the color of the triangle
                    image[y, x] = color

    return image

# @TODO: refactor this function with the one in shader.py


def point_in_triangle(x: int, y: int, triangle: torch.Tensor) -> bool:
    # Get the vertices of the triangle
    v0, v1, v2 = triangle

    # Calculate the barycentric coordinates of the point (x, y) with respect to the triangle
    barycentric_coords = barycentric_coordinates(x, y, v0, v1, v2)

    # Check if the barycentric coordinates are inside the triangle
    return 0 <= barycentric_coords[0] <= 1 and 0 <= barycentric_coords[1] <= 1 and 0 <= barycentric_coords[2] <= 1


def barycentric_coordinates(x: int, y: int, v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    # Calculate the area of the triangle
    triangle_area = 0.5 * (-v1[1] * v2[0] + v0[1] * (-v1[0] + v2[0]) + v0[0] * (v1[1] - v2[1]) + v1[0] * v2[1])

    # Calculate the barycentric coordinates
    alpha = (0.5 * (-v1[1] * v2[0] + v0[1] * (-v1[0] + v2[0]) + v0[0] * (v1[1] - v2[1]) + v1[0] * v2[1]) -
             0.5 * (-v1[1] * x + v0[1] * (-v1[0] + x) + v0[0] * (v1[1] - y) + v1[0] * y)) / triangle_area
    beta = (0.5 * (v2[1] * x - v0[1] * v2[0] + v0[0] * (-v2[1] + y) - v2[0] * y) / triangle_area)
    gamma = 1 - alpha - beta

    return torch.tensor([alpha, beta, gamma])
