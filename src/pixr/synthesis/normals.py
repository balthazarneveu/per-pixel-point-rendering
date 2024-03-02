import torch


def extract_normals(wc_triangles: torch.Tensor) -> torch.Tensor:
    """
    Extracts the normal vectors from the world-coordinate triangles.

    Args:
        wc_triangles (torch.Tensor): [N, 4=xyz1, 3] triangles in world coordinates

    Returns:
        torch.Tensor:  wc_normals (torch.Tensor): [N, 3] normals at the vertices
    """
    normal_vectors = torch.cross(wc_triangles[:, :-1, 1] - wc_triangles[:, :-1, 0],
                                 wc_triangles[:, :-1, 2] - wc_triangles[:, :-1, 0],
                                 dim=-1)
    normal_vectors /= torch.linalg.norm(normal_vectors, dim=1, keepdim=True)
    return normal_vectors
