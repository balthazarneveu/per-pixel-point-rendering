import torch
from typing import List, Tuple
from interactive_pipe import interactive
from interactive_pipe.data_objects.curves import Curve, SingleCurve



def get_camera_intrinsics(w: int = 640, h: int = 480) -> torch.Tensor:

    cx, cy = w / 2., h / 2.
    focal_length = 1000.
    # camera_intrinsics[0, 0] = focal
    # camera_intrinsics[1, 1] = focal
    # camera_intrinsics[0, 2] = cx
    # camera_intrinsics[1, 2] = cy
    fx, fy = focal_length, focal_length
    return torch.Tensor(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ]
    ), w, h


def main():
    # [N, 3, xyz]
    z = 5
    wc_triangles = torch.Tensor(
        [
            [
                [0., 0., z, 1.],
                [0., 1., z, 1.],
                [1., 1., z, 1.]
            ],
            [
                [0., 0., z, 1.],
                [2., 0., z, 1.],
                [2., 1., z, 1.]
            ]
        ]

    )
    wc_triangles = wc_triangles.permute(0, 2, 1)
    print(wc_triangles.shape)
    camera = torch.zeros(3, 4)
    cam_rot = torch.eye(3)
    cam_pos = torch.zeros(3)
    camera[:3, :3] = cam_rot
    camera[:, -1] = cam_pos
    print(camera.shape, wc_triangles.shape)
    cc_triangles = torch.matmul(camera, wc_triangles)

    camera_intrinsics, w, h = get_camera_intrinsics()
    cc_triangles[:, 1, :] *= -1.  # flip y axis to get a image-like coordinate system
    cc_triangles = torch.matmul(camera_intrinsics, cc_triangles)
    cc_triangles /= cc_triangles[:, -1:, :]  # pinhole model! normalize by distance
    print(cc_triangles.shape)

    cc_triangles = cc_triangles.numpy()
    t1 = SingleCurve(
        x=[cc_triangles[0, 0, idx] for idx in [0, 1, 2, 0]],
        y=[cc_triangles[0, 1, idx] for idx in [0, 1, 2, 0]],
        style="bo"
    )
    t2 = SingleCurve(
        x=[cc_triangles[1, 0, idx] for idx in [0, 1, 2, 0]],
        y=[cc_triangles[1, 1, idx] for idx in [0, 1, 2, 0]],
        style="ro"
    )
    corners = SingleCurve(
        x=[0, 0, w, w, 0],
        y=[0, h, h, 0, 0],
        style="k-"
    )
    center = SingleCurve(
        x=[w/2],
        y=[h/2],
        style="g+",
        markersize=10
    )
    Curve([t1, t2, center, corners],
          xlim=[0, w-1],
          ylim=[h-1, 0],
          grid=True).show()


if __name__ == '__main__':
    main()
