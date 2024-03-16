from pixr.synthesis.world_simulation import generate_simulated_world, STAIRCASE
from pixr.synthesis.normals import extract_normals
from pixr.synthesis.extract_point_cloud import pick_point_cloud_from_triangles
from pixr.synthesis.forward_project import project_3d_to_2d
from interactive_pipe.data_objects.image import Image
from pixr.rendering.splatting import splat_points
# from pixr.rendering.legacy_splatting import splat_points as splat_points  # Run the for loop
from pixr.multiview.scenes_utils import load_views
from interactive_pipe.data_objects.image import Image
import torch
import matplotlib.pyplot as plt
from config import OUT_DIR
from pixr.properties import DEVICE
import argparse


def forward_chain_not_parametric(point_cloud, wc_normals, cam_ext, cam_int, colors, w, h, scale=0, no_grad=False):
    # print(point_cloud.device, wc_normals.device, cam_ext.device, cam_int.device, colors.device)
    wc_normals = wc_normals.to(point_cloud.device)
    point_cloud = point_cloud
    proj_point_cloud, depth, cc_normals = project_3d_to_2d(point_cloud, cam_int, cam_ext, wc_normals, no_grad=True)
    img = splat_points(
        proj_point_cloud,
        colors,
        depth,
        w, h,
        cam_int,
        cc_normals,
        no_grad=no_grad,
        scale=scale,
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
        suffix="",
        scale=0):
    with torch.no_grad():
        for batch_idx in range(camera_extrinsics.shape[0]):
            cam_ext, cam_int = camera_extrinsics[batch_idx, ...], camera_intrinsics[batch_idx, ...]
            img_pred = forward_chain_not_parametric(
                wc_points, wc_normals, cam_ext, cam_int, color_pred, w, h, scale=scale)
            img = Image(img_pred.cpu().numpy())
            img.save(save_path/f"{batch_idx:04d}_{suffix}.png")


def main(out_root=OUT_DIR, name=STAIRCASE, device=DEVICE, show=True, save=False):
    view_dir = out_root/f"{name}"
    views = sorted(list(view_dir.glob("view_*")))
    all_rendered_images, camera_intrinsics, camera_extrinsics, w, h = prepare_data(views)
    all_rendered_images.requires_grad = False
    # Limit the amount of views for now! -> split train and validation here!
    # rendered_images = all_rendered_images[:5]
    rendered_images = all_rendered_images
    rendered_images = rendered_images.to(device)

    out_dir = out_root/f"{name}_splat_differentiate_points"
    out_dir.mkdir(exist_ok=True, parents=True)
    camera_extrinsics.requires_grad = False
    camera_intrinsics.requires_grad = False
    wc_triangles, colors = generate_simulated_world(scene_mode=name)
    wc_normals = extract_normals(wc_triangles)
    wc_points, points_colors, wc_normals = pick_point_cloud_from_triangles(
        wc_triangles, colors, wc_normals,
        # num_samples=20000
        num_samples=20000
    )
    # Move data to GPU
    wc_points = wc_points.to(device)
    camera_intrinsics = camera_intrinsics.to(device)
    camera_extrinsics = camera_extrinsics.to(device)
    color_pred = torch.randn(points_colors.shape, requires_grad=True, device=device)
    color_pred = color_pred.to(device)
    optimizer = torch.optim.Adam([color_pred], lr=0.3)

    n_steps = 200+1
    n_steps = 40+1
    scales = [0, 1, 2, 3]
    with torch.autograd.set_detect_anomaly(True):
        for step in range(n_steps):
            optimizer.zero_grad()
            loss = 0.
            # Aggregate the loss over several views
            for batch_idx in range(rendered_images.shape[0]):
                cam_ext, cam_int = camera_extrinsics[batch_idx, ...], camera_intrinsics[batch_idx, ...]
                for scale in scales:
                    img_pred = forward_chain_not_parametric(
                        wc_points, wc_normals, cam_ext, cam_int, color_pred, w, h, scale=scale)
                    if scale > 0:
                        # Downsample the target image using basic average pooling as an AA filter
                        img_target = torch.nn.functional.avg_pool2d(
                            rendered_images[batch_idx].permute(2, 0, 1), 2**scale).permute(1, 2, 0)
                    else:
                        img_target = rendered_images[batch_idx]
                    loss += torch.nn.functional.mse_loss(img_pred, img_target)
            loss.backward()
            optimizer.step()

            if step % 1 == 0:
                print(
                    f"Step {step:05d}\tLoss {loss.item():.5f}"
                )
            plot_step_flag = step % 10 == 0
            # This is the draft of a validation step
            # Plot convergence images
            if False and step % 100 == 0 and save:
                for scale in range(3):
                    validation_step(
                        wc_points, wc_normals, camera_extrinsics, camera_intrinsics, w, h, color_pred,
                        target_img=all_rendered_images, save_path=out_dir, suffix=f"step_{step:05d}_scale={scale:02d}",
                        scale=scale
                    )
            if plot_step_flag and show:
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.title("Groundtruth")
                plt.imshow(rendered_images[batch_idx].detach().cpu().numpy())
                plt.subplot(1, 2, 2)
                print(img_pred.shape)
                plt.imshow(img_pred.clip(0., 1.).detach().cpu().numpy())
                plt.show()
            torch.save({
                "point_cloud": wc_points,
                "normals": wc_normals,
                "colors": color_pred,
            },
                out_dir/f"checkpoint_{step:05d}.pt"
            )
    for scale in range(3):
        validation_step(
            wc_points, wc_normals, camera_extrinsics, camera_intrinsics, w, h, color_pred,
            target_img=all_rendered_images, save_path=out_dir, suffix=f"step_{step:05d}_scale={scale:02d}",
            scale=scale
        )
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Groundtruth")
    plt.imshow(rendered_images[batch_idx].detach().cpu().numpy())
    plt.subplot(1, 2, 2)
    print(img_pred.shape)
    plt.imshow(img_pred.clip(0., 1.).detach().cpu().numpy())
    # plt.title(f"Step {step}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a scene using BlenderProc")
    parser.add_argument("-s", "--scene", type=str, help="Name of the scene to render", default=STAIRCASE)
    parser.add_argument("-e", "--export", action="store_true", help="Export validation image")
    args = parser.parse_args()
    main(name=args.scene, show=False, save=args.export)
