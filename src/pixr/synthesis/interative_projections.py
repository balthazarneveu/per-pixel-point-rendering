import torch
from typing import List, Tuple
from interactive_pipe import interactive, interactive_pipeline
from interactive_pipe.data_objects.curves import Curve, SingleCurve

from pixr.synthesis.raster import get_camera_intrinsics, get_camera_extrinsics, project_3d_to_2d


@interactive(
    yaw_deg=(0., (-180., 180.)),
    pitch_deg=(0., (-180., 180.)),
    roll_deg=(0., (-180., 180.)),
    trans_x=(0., (-10., 10.)),
    trans_y=(0., (-10., 10.)),
    trans_z=(0., (-10., 10.))
)
def set_camera_parameters(yaw_deg=0., pitch_deg=0., roll_deg=0., trans_x=0., trans_y=0., trans_z=0.) -> torch.Tensor:
    yaw = torch.deg2rad(torch.Tensor([yaw_deg]))
    pitch = torch.deg2rad(torch.Tensor([pitch_deg]))
    roll = torch.deg2rad(torch.Tensor([roll_deg]))

    cam_pos = torch.stack([torch.Tensor([trans_x]), torch.Tensor([trans_y]), torch.Tensor([trans_z])])
    return yaw, pitch, roll, cam_pos


@interactive(
    z=(10., (2., 100.)),
    delta_z=(0., (-5., 5.))
)
def generate_3d_scene(z=5, delta_z=0.):
    # [N, 3, xyz]

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
            ]
        ]

    )
    wc_triangles = wc_triangles.permute(0, 2, 1)
    return wc_triangles


def visualize_2d_scene(cc_triangles: torch.Tensor, w, h) -> Curve:
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
    img_scene = Curve(
        [t1, t2, center, corners],
        xlim=[0, w-1],
        ylim=[h-1, 0],
        grid=True,
        xlabel="x",
        ylabel="y",
        title="Projected points")
    return img_scene


@interactive_pipeline(gui="mpl", size=(10, 10))
def projection_pipeline():
    wc_triangles = generate_3d_scene()
    yaw, pitch, roll, cam_pos = set_camera_parameters()
    camera_extrinsics = get_camera_extrinsics(yaw, pitch, roll, cam_pos)
    camera_intrinsics, w, h = get_camera_intrinsics()
    # camera_intrinsics, w, h = get_camera_intrinsics()
    cc_triangles = project_3d_to_2d(wc_triangles, camera_intrinsics, camera_extrinsics)
    img_scene = visualize_2d_scene(cc_triangles, w, h)
    return img_scene


def main():
    projection_pipeline()


if __name__ == '__main__':
    main()
