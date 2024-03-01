from pixr.camera.camera_geometry import get_camera_intrinsics, get_camera_extrinsics
from pixr.synthesis.forward_project import project_3d_to_2d
from interactive.interactive_projections import set_camera_parameters, generate_3d_scene, splat_points

import torch


def main():
    # Fit camera parameters from 2D projections and known 3D scene (fixed point cloud)
    with torch.no_grad():
        nice_config = {
            'yaw_deg': 8.861538461538487, 'pitch_deg': 10.523076923076928, 'roll_deg': -
            11.076923076923038, 'trans_x': -1.53846153846154, 'trans_y': 0.7999999999999972, 'trans_z': 0.0}
        yaw_gt, pitch_gt, roll_gt, cam_pos_gt = set_camera_parameters(**nice_config)
        proj_point_cloud_gt, splat_image_gt = forward_chain(yaw_gt, pitch_gt, roll_gt, cam_pos_gt)
        proj_point_cloud_gt.requires_grad = False
        splat_image_gt.requires_grad = False
    with torch.autograd.set_detect_anomaly(True):
        yaw = torch.tensor([0.], requires_grad=True)
        pitch = torch.tensor([0.], requires_grad=True)
        roll = torch.tensor([0.], requires_grad=True)
        cam_pos = torch.tensor([0., 0., 0.], requires_grad=True).unsqueeze(-1)
        yaw = torch.nn.Parameter(yaw)
        pitch = torch.nn.Parameter(pitch)
        roll = torch.nn.Parameter(roll)
        cam_pos = torch.nn.Parameter(cam_pos)
        optimizer = torch.optim.Adam([yaw, pitch, roll, cam_pos], lr=0.01)
        for step in range(4000):
            optimizer.zero_grad()
            proj_point_cloud, img = forward_chain(yaw, pitch, roll, cam_pos)
            loss = torch.nn.functional.mse_loss(proj_point_cloud, proj_point_cloud_gt)
            # loss = torch.nn.functional.mse_loss(img, splat_image_gt) # this is not working!
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                print(
                    f"Step {step}: Loss {loss.item()}, Yaw {torch.rad2deg(yaw).item()}, Pitch {torch.rad2deg(pitch).item()}, Roll {torch.rad2deg(roll).item()}")


def forward_chain(yaw, pitch, roll, cam_pos):
    cam_int, w, h = get_camera_intrinsics()
    point_cloud, colors = generate_3d_scene(delta_z=1.)
    point_cloud.requires_grad = False
    cam_ext = get_camera_extrinsics(yaw, pitch, roll, cam_pos)
    proj_point_cloud, _depth = project_3d_to_2d(point_cloud, cam_int, cam_ext)
    img = splat_points(proj_point_cloud, colors, w, h)
    return proj_point_cloud, img


if __name__ == "__main__":
    main()
    # w, h = 640, 480
    # camera_intrinsics, w, h = get_camera_intrinsics(w, h)
    # yaw, pitch, roll, cam_pos = set_camera_parameters(yaw_deg=0., pitch_deg
