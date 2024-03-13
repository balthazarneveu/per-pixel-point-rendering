from pixr.camera.camera_geometry import get_camera_intrinsics, get_camera_extrinsics
from pixr.synthesis.forward_project import project_3d_to_2d
from pixr.synthesis.world_simulation import generate_simulated_world, TEST_TRIANGLES
from pixr.camera.camera_geometry import set_camera_parameters
from pixr.rendering.splatting import splat_points
from pixr.synthesis.extract_point_cloud import pick_point_cloud_from_triangles
import torch
from matplotlib import pyplot as plt
from typing import Optional
from pixr.properties import DEVICE
from pixr.synthesis.normals import extract_normals


def generate_world(
    num_samples: Optional[int] = None,
    device=DEVICE
):
    with torch.no_grad():
        triangles, colors = generate_simulated_world(z=5., delta_z=1., scene_mode=TEST_TRIANGLES)
        wc_normals = extract_normals(triangles)
        if num_samples is not None:
            point_cloud, colors, wc_normals = pick_point_cloud_from_triangles(
                triangles, colors, wc_normals, num_samples=num_samples)
        else:
            # point_cloud = triangles.view(-1, 1, 3)
            # @TODO: flatten  the primitive dimensions
            point_cloud = triangles
        point_cloud.requires_grad = False
        point_cloud = point_cloud.to(device)
        colors = colors.to(device)
    return point_cloud, colors, wc_normals


def generate_scene(
        point_cloud_3d,
        wc_normals,
        colors=None,
        scene_config={
            'yaw_deg': 8.861538461538487,
            'pitch_deg': 10.523076923076928,
            'roll_deg': -11.076923076923038,
            'trans_x': -1.53846153846154,
            'trans_y': 0.8,
            'trans_z': 13.,
            'w': 128,
            'h': 64,
            'focal_length_pix': 200.
        },
        device=DEVICE,
):
    with torch.no_grad():
        cam_int_gt, w, h = get_camera_intrinsics(
            w=scene_config['w'],
            h=scene_config['h'],
            focal_length_pix=scene_config['focal_length_pix']
        )
        yaw_gt, pitch_gt, roll_gt, cam_pos_gt = set_camera_parameters(**scene_config)
        cam_int_gt = cam_int_gt.to(device)
        cam_pos_gt = cam_pos_gt.to(device)
        wc_normals = wc_normals.to(device)
        yaw_gt = yaw_gt.to(device)
        pitch_gt = pitch_gt.to(device)
        roll_gt = roll_gt.to(device)
        proj_point_cloud_gt, splat_image_gt, cc_normals = forward_chain(
            point_cloud_3d,
            wc_normals,
            yaw_gt, pitch_gt, roll_gt, cam_pos_gt, cam_int_gt,
            colors=colors,
            w=w, h=h
        )
        proj_point_cloud_gt.requires_grad = False
        if colors is not None:
            splat_image_gt.requires_grad = False
    return proj_point_cloud_gt, cc_normals, yaw_gt, pitch_gt, roll_gt, cam_pos_gt, w, h, cam_int_gt, splat_image_gt


def check_gradient_descent_on_camera_coordinates(
    splat_flag=False,
    show=False,
    device=DEVICE,
    num_samples: Optional[int] = None
):
    # NO BACKPROP ALONG SPLATTING INVOLVED !
    # Fit camera parameters from 2D projections and known 3D scene (fixed point cloud)
    point_cloud_3d, colors, wc_normals = generate_world(num_samples=num_samples, device=device)
    (proj_point_cloud_gt, cc_normals, yaw_gt, pitch_gt, roll_gt, cam_pos_gt,
     w, h, cam_int_gt, splat_image_gt) = generate_scene(
        point_cloud_3d, wc_normals, colors=colors, device=device)
    proj_point_cloud_gt = proj_point_cloud_gt.to(device)
    if splat_image_gt is not None:
        splat_image_gt = splat_image_gt.clip(0., 1.)
    # if splat_flag:
    #     plt.imshow(splat_image_gt.detach().numpy())
    #     plt.show()
    with torch.autograd.set_detect_anomaly(True):
        yaw = torch.tensor([0.], requires_grad=True).to(device)
        pitch = torch.tensor([0.], requires_grad=True).to(device)
        roll = torch.tensor([0.], requires_grad=True).to(device)
        cam_pos = torch.tensor([0., 0., 14.], requires_grad=True).to(device).unsqueeze(-1)
        yaw = torch.nn.Parameter(yaw)
        pitch = torch.nn.Parameter(pitch)
        roll = torch.nn.Parameter(roll)
        cam_pos = torch.nn.Parameter(cam_pos)
        optimizer = torch.optim.Adam([yaw, pitch, roll, cam_pos], lr=0.3)
        cam_int_gt.requires_grad = False  # freeze the camera intrinsics
        for step in range(200+1):
            plot_step_flag = step % 100 == 0
            optimizer.zero_grad()
            proj_point_cloud, img_pred, _ = forward_chain(
                point_cloud_3d, wc_normals,
                yaw, pitch, roll,
                cam_pos,
                cam_int_gt,
                w=w, h=h,
                colors=colors if (splat_flag and plot_step_flag) else None
            )
            loss = torch.nn.functional.mse_loss(proj_point_cloud, proj_point_cloud_gt)
            # loss = torch.nn.functional.mse_loss(img, splat_image_gt) # this is not working!
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                print(
                    f"Step {step:05d}\tLoss {loss.item():.5f}" +
                    f"\tYaw {torch.rad2deg(yaw).item():.3f}" +
                    f"\tPitch {torch.rad2deg(pitch).item():.3f}" +
                    f"\tRoll {torch.rad2deg(roll).item():.3f}"
                )

            if splat_flag and plot_step_flag and show:
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.title("Groundtruth")
                plt.imshow(splat_image_gt.detach().numpy())
                plt.subplot(1, 2, 2)
                plt.imshow(img_pred.clip(0., 1.).detach().numpy())
                plt.title(f"Step {step}")
                plt.show()


def check_gradient_descent_on_point_features(
    show=False,
    device=DEVICE,
    num_samples: Optional[int] = None
):
    # NO BACKPROP ALONG SPLATTING INVOLVED !
    # Fit camera parameters from 2D projections and known 3D scene (fixed point cloud)
    point_cloud_3d, colors, wc_normals = generate_world(num_samples=num_samples, device=device)
    (proj_point_cloud_gt, cc_normals, yaw_gt, pitch_gt, roll_gt, cam_pos_gt,
     w, h, cam_int_gt, splat_image_gt) = generate_scene(
        point_cloud_3d, wc_normals, colors=colors, device=device)
    proj_point_cloud_gt = proj_point_cloud_gt.to(device)
    if splat_image_gt is not None:
        splat_image_gt = splat_image_gt.clip(0., 1.)
    with torch.autograd.set_detect_anomaly(True):
        color_pred = torch.rand_like(colors, requires_grad=True).to(device)
        optimizer = torch.optim.Adam([color_pred], lr=0.3)
        cam_int_gt.requires_grad = False  # freeze the camera intrinsics
        for step in range(50+1):
            plot_step_flag = step % 10 == 0
            optimizer.zero_grad()
            proj_point_cloud, img_pred, _ = forward_chain(
                point_cloud_3d, wc_normals,
                yaw_gt, pitch_gt, roll_gt,
                cam_pos_gt,
                cam_int_gt,
                w=w, h=h,
                colors=color_pred
            )
            # loss = torch.nn.functional.mse_loss(proj_point_cloud, proj_point_cloud_gt)
            loss = torch.nn.functional.mse_loss(img_pred, splat_image_gt)
            loss.backward()
            optimizer.step()
            if step % 1 == 0:
                print(
                    f"Step {step:05d}\tLoss {loss.item():.5f}"
                )

            if plot_step_flag and show:
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.title("Groundtruth")
                plt.imshow(splat_image_gt.detach().numpy())
                plt.subplot(1, 2, 2)
                plt.imshow(img_pred.clip(0., 1.).detach().numpy())
                plt.title(f"Step {step}")
                plt.show()


def forward_chain(
    point_cloud,
    wc_normals,
    yaw, pitch, roll, cam_pos, cam_int,
    colors=None, w=640, h=480
):
    wc_normals = wc_normals.to(point_cloud.device)
    cam_ext = get_camera_extrinsics(yaw, pitch, roll, cam_pos)
    proj_point_cloud, depth, cc_normals = project_3d_to_2d(point_cloud, cam_int, cam_ext, wc_normals, no_grad=False)
    if colors is not None:
        img = splat_points(
            proj_point_cloud, colors, depth,
            w, h,
            cam_int,
            cc_normals,
            no_grad=False,
            # debug=True
        )  # Optional here until this is differentiable
    else:
        img = None
    return proj_point_cloud, img, cc_normals


def main(visualize=False):
    check_gradient_descent_on_point_features(show=visualize, device=DEVICE, num_samples=1000)
    if visualize:
        check_gradient_descent_on_camera_coordinates(splat_flag=True, show=True, device=DEVICE, num_samples=1000)
    else:
        check_gradient_descent_on_camera_coordinates(device=DEVICE)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--visualize", action="store_true", default=False)
    args = parser.parse_args()
    main(visualize=args.visualize)