from pixr.synthesis.world_simulation import generate_simulated_world, STAIRCASE
from pixr.synthesis.normals import extract_normals
from pixr.synthesis.extract_point_cloud import pick_point_cloud_from_triangles
from pixr.synthesis.forward_project import project_3d_to_2d
from interactive_pipe.data_objects.image import Image
from pixr.rendering.splatting import splat_points
from pixr.rasterizer.rasterizer import shade_screen_space
from pixr.multiview.scenes_utils import load_views
from interactive_pipe.data_objects.image import Image
from differentiate_points_projection import forward_chain
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import OUT_DIR
from pixr.properties import DEVICE
import argparse


def forward_chain_not_parametric(point_cloud, wc_normals, cam_ext, cam_int, colors, w, h):
    wc_normals = wc_normals.to(point_cloud.device)
    proj_point_cloud, depth, cc_normals = project_3d_to_2d(point_cloud, cam_int, cam_ext, wc_normals, no_grad=True)
    img = splat_points(
        proj_point_cloud,
        colors,
        depth,
        w, h,
        cam_int,
        cc_normals,
        no_grad=False,
    )
    return img


def prepare_data(views_path):
    views = load_views(views_path)
    rendered_images = []
    camera_intrinsics = []
    camera_extrinsics = []
    for idx, view_dict in enumerate(views):
        img = Image.load_image(view_dict["path_to_image"])
        img = torch.from_numpy(img).float()  # .permute(2, 0, 1)
        rendered_images.append(img)
        camera_intrinsics_single, w, h = view_dict["camera_intrinsics"], view_dict["w"], view_dict["h"]
        camera_extrinsics_single = view_dict["camera_extrinsics"]
        camera_intrinsics.append(camera_intrinsics_single)
        camera_extrinsics.append(camera_extrinsics_single)
    rendered_images = torch.stack(rendered_images)
    camera_intrinsics = torch.stack(camera_intrinsics)
    camera_extrinsics = torch.stack(camera_extrinsics)
    return rendered_images, camera_intrinsics, camera_extrinsics, w, h


def validation_step(
        wc_points, wc_normals, camera_extrinsics, camera_intrinsics, w, h, color_pred,
        target_img=None,
        save_path=None,
        suffix=""):
    with torch.no_grad():
        for batch_idx in range(camera_extrinsics.shape[0]):
            cam_ext, cam_int = camera_extrinsics[batch_idx, ...], camera_intrinsics[batch_idx, ...]
            img_pred = forward_chain_not_parametric(wc_points, wc_normals, cam_ext, cam_int, color_pred, w, h)
            img = Image(img_pred.cpu().numpy())
            img.save(save_path/f"{batch_idx:04d}_{suffix}.png")


def main(out_root=OUT_DIR, name=STAIRCASE, device=DEVICE, show=True, save=False):
    view_dir = out_root/f"{name}"
    views = sorted(list(view_dir.glob("view_*")))
    all_rendered_images, camera_intrinsics, camera_extrinsics, w, h = prepare_data(views)
    all_rendered_images.requires_grad = False
    # Limit the amount of views for now! -> split train and validation here!
    rendered_images = all_rendered_images[:2]
    print(rendered_images.shape, camera_intrinsics.shape, camera_extrinsics.shape, w, h)

    out_dir = out_root/f"{name}_splat_differentiate_points"
    out_dir.mkdir(exist_ok=True, parents=True)
    camera_extrinsics.requires_grad = False
    camera_intrinsics.requires_grad = False
    wc_triangles, colors = generate_simulated_world(scene_mode=name)
    wc_normals = extract_normals(wc_triangles)
    wc_points, points_colors, wc_normals = pick_point_cloud_from_triangles(
        wc_triangles, colors, wc_normals,
        # num_samples=20000
        num_samples=2000
    )

    color_pred = torch.randn(points_colors.shape, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([color_pred], lr=0.3)
    with torch.autograd.set_detect_anomaly(False):
        for step in range(200+1):
            optimizer.zero_grad()
            loss = 0.
            # Aggregate the loss over several views
            for batch_idx in range(rendered_images.shape[0]):
                cam_ext, cam_int = camera_extrinsics[batch_idx, ...], camera_intrinsics[batch_idx, ...]
                img_pred = forward_chain_not_parametric(wc_points, wc_normals, cam_ext, cam_int, color_pred, w, h)
                loss += torch.nn.functional.mse_loss(img_pred, rendered_images[batch_idx])
            loss.backward()
            optimizer.step()

            if step % 1 == 0:
                print(
                    f"Step {step:05d}\tLoss {loss.item():.5f}"
                )
            plot_step_flag = step % 10 == 0
            # This is the draft of a validation step
            if step % 100 == 0 and save:
                validation_step(
                    wc_points, wc_normals, camera_extrinsics, camera_intrinsics, w, h, color_pred,
                    target_img=all_rendered_images, save_path=out_dir, suffix=f"step_{step:05d}")
            if plot_step_flag and show:
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.title("Groundtruth")
                plt.imshow(rendered_images[batch_idx].detach().numpy())
                plt.subplot(1, 2, 2)
                print(img_pred.shape)
                plt.imshow(img_pred.clip(0., 1.).detach().numpy())
                plt.show()

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Groundtruth")
    plt.imshow(rendered_images[batch_idx].detach().numpy())
    plt.subplot(1, 2, 2)
    print(img_pred.shape)
    plt.imshow(img_pred.clip(0., 1.).detach().numpy())
    # plt.title(f"Step {step}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a scene using BlenderProc")
    parser.add_argument("-s", "--scene", type=str, help="Name of the scene to render", default=STAIRCASE)
    parser.add_argument("-e", "--export", action="store_true", help="Export validation image")
    args = parser.parse_args()
    main(name=args.scene, show=False, save=args.export)
