import torch
from typing import Tuple


def project_3d_to_2d(
    wc_triangles: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    camera_extrinsics: torch.Tensor,
    no_grad: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    camera_extrinsics = camera_extrinsics.to(wc_triangles.device)
    camera_intrinsics = camera_intrinsics.to(wc_triangles.device)
    with torch.no_grad() if no_grad else torch.enable_grad():
        cc_triangles = torch.matmul(camera_extrinsics, wc_triangles)
        cc_triangles[:, 1, :] *= -1.  # flip y axis to get a image-like coordinate system
        cc_triangles = torch.matmul(camera_intrinsics, cc_triangles)
        depth = cc_triangles[:, -1:, :]
        cc_triangles = cc_triangles / cc_triangles[:, -1:, :]  # pinhole model! normalize by distance
    return cc_triangles, depth
