import torch


def pick_point_cloud_from_triangles(
        wc_triangles: torch.Tensor,
        colors: torch.Tensor,
        num_samples: int = 100
) -> torch.Tensor:
    """Pick m samples from the triangles in the 3D scene.

    Args:
        wc_triangles (torch.Tensor): [N, 4=xyz1, 3] triangles
        colors (torch.Tensor): [N, 3, 3=rgb] colors at the vertices
        num_samples (int, optional): number of samples to pick. Defaults to 100.

    Returns:
        torch.Tensor: [m, 4=xyz1, 1] point cloud
        torch.Tensor: [m, 1, 3=rgb] point colors
    """
    print(colors.shape)
    # Calculate the vectors forming the sides of the triangles
    vec0 = wc_triangles[:, :3, 1] - wc_triangles[:, :3, 0]
    vec1 = wc_triangles[:, :3, 2] - wc_triangles[:, :3, 0]
    cross_product = torch.cross(vec0, vec1, dim=1)

    # Calculate the area of each triangle for weighted sampling (using cross product)
    areas = 0.5 * torch.norm(cross_product, dim=1)
    total_area = torch.sum(areas)
    probabilities = areas / total_area
    # Probabilities shall be 0.25, 0.75 for the 2 triangles we're considering
    # Sample triangle indices based on their area
    sampled_indices = torch.multinomial(probabilities, num_samples, replacement=True)

    # Sample points within the selected triangles
    sampled_points = torch.zeros(num_samples, 4, 1, device=wc_triangles.device)  # Initialize tensor for sampled points
    sampled_colors = torch.zeros(num_samples, 1, 3, device=wc_triangles.device)  # Initialize tensor for sampled colors
    for i, idx in enumerate(sampled_indices):
        # Barycentric coordinates for a random point within a triangle
        r1, r2 = torch.sqrt(torch.rand(1)), torch.rand(1)
        barycentric = torch.tensor([1 - r1, r1 * (1 - r2), r1 * r2], device=wc_triangles.device)

        # Convert barycentric coordinates to Cartesian coordinates
        point = torch.matmul(wc_triangles[idx, :3, :], barycentric.unsqueeze(1))
        # Convert to homogeneous coordinates
        point_homogeneous = torch.cat((point, torch.tensor([[1.0]], device=wc_triangles.device)))

        sampled_points[i] = point_homogeneous

        # Interpolate colors based on barycentric coordinates
        # color = torch.einsum('ij,j->i', colors[idx], barycentric)  # Efficient matrix-vector multiplication
        color = torch.einsum('ij,j->i', colors[idx].T, barycentric)  # Efficient matrix-vector multiplication
        sampled_colors[i, 0, :] = color

    return sampled_points, sampled_colors
