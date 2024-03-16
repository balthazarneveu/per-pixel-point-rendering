from pixr.synthesis.world_simulation import STAIRCASE
from pixr.learning.experiments import get_training_content
import torch
from pixr.learning.utils import prepare_dataset
from config import OUT_DIR, TRAINING_DIR
from pixr.properties import DEVICE, TRAIN, VALIDATION, METRIC_PSNR, LOSS, LOSS_MSE, NB_EPOCHS, LR
from tqdm import tqdm
import argparse
from experiments_definition import get_experiment_from_id
from pathlib import Path
from pixr.learning.loss import compute_loss
from pixr.learning.metrics import compute_metrics
import wandb
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pixr.learning.dataloader import get_data_loader
from pixr.rendering.splatting import splat_points
from pixr.synthesis.forward_project import project_3d_to_2d


def infer_function(point_cloud, cam_int, cam_ext, wc_normals, colors, w, h, scale=0, no_grad=False):
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


def training_loop(
    model,
    optimizer,
    wc_points,
    wc_normals,
    color_pred,
    dl_dict: dict,
    config: dict,
    scheduler=None,
    device: str = DEVICE,
    wandb_flag: bool = False,
    output_dir: Path = None
):
    best_accuracy = 0.
    model.to(device)
    for n_epoch in tqdm(range(config[NB_EPOCHS])):
        current_metrics = {TRAIN: 0., VALIDATION: 0., LR: optimizer.param_groups[0]['lr'],
                           METRIC_PSNR: 0.
                           }
        for phase in [TRAIN, VALIDATION]:
            if phase == TRAIN:
                model.train()
            else:
                model.eval()
            for target_view, cam_int, cam_ext in tqdm(dl_dict[phase], desc=f"{phase} - Epoch {n_epoch}"):
                target_view = target_view.to(device)
                # x = torch.rand_like(target_view, device=device)
                h, w = target_view.shape[-2:]
                # print(cam_int.shape, cam_ext.shape, wc_points.shape, wc_normals.shape, color_pred.shape, w, h)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == TRAIN):
                    loss = 0
                    for scale in [3, 2, 1, 0]:
                        batch_splat = []
                        for img_index in range(cam_int.shape[0]):
                            img_splat = infer_function(
                                wc_points, cam_int[img_index], cam_ext[img_index], wc_normals, color_pred, w, h, scale=scale)
                            batch_splat.append(img_splat.permute(2, 0, 1))
                        batch_splat = torch.stack(batch_splat)
                        image_pred = model(batch_splat)
                        if scale > 0:
                            img_target = torch.nn.functional.avg_pool2d(target_view, 2**scale)
                        else:
                            img_target = target_view
                        loss += compute_loss(image_pred, img_target, mode=config.get(LOSS, LOSS_MSE))
                    if torch.isnan(loss):
                        print(f"Loss is NaN at epoch {n_epoch} and phase {phase}!")
                        continue
                    if phase == TRAIN:
                        loss.backward()
                        optimizer.step()
                current_metrics[phase] += loss.item()
                if phase == VALIDATION:
                    metrics_on_batch = compute_metrics(image_pred, target_view)
                    for k, v in metrics_on_batch.items():
                        current_metrics[k] += v

            current_metrics[phase] /= (len(dl_dict[phase]))
            if phase == VALIDATION:
                for k, v in metrics_on_batch.items():
                    current_metrics[k] /= (len(dl_dict[phase]))
                    try:
                        current_metrics[k] = current_metrics[k].item()
                    except AttributeError:
                        pass
        debug_print = f"{phase}: Epoch {n_epoch} - Loss: {current_metrics[phase]:.3e} "
        for k, v in current_metrics.items():
            if k not in [TRAIN, VALIDATION, LR]:
                debug_print += f"{k}: {v:.3} |"
        print(debug_print)
        if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(current_metrics[VALIDATION])
        if output_dir is not None:
            with open(output_dir/f"metrics_{n_epoch}.json", "w") as f:
                json.dump(current_metrics, f)
        if wandb_flag:
            wandb.log(current_metrics)
        if best_accuracy < current_metrics[METRIC_PSNR]:
            best_accuracy = current_metrics[METRIC_PSNR]
            if output_dir is not None:
                print("new best model saved!")
                torch.save(model.state_dict(), output_dir/"best_model.pt")
        torch.save({
            "point_cloud": wc_points,
            "normals": wc_normals,
            "colors": color_pred,
        },
            output_dir/f"point_cloud_checkpoint_{n_epoch:05d}.pt"
        )
    if output_dir is not None:
        torch.save(model.cpu().state_dict(), output_dir/"last_model.pt")
    return model


def main(out_root=OUT_DIR, name=STAIRCASE, device=DEVICE, exp: int = 1, num_samples=20000, pseudo_color_dim=3):
    config = get_experiment_from_id(exp)

    train_material, valid_material, (w, h), point_cloud_material = prepare_dataset(
        out_root, name, num_samples=num_samples)
    # Move training data to GPU
    wc_points, wc_normals = point_cloud_material
    wc_points = wc_points.to(device)
    wc_normals = wc_normals.to(device)
    dl_dict = get_data_loader(config, train_material, valid_material)
    # dl_dict[TRAIN]

    # for i, (batch_inp, batch_cam_int, batch_cam_ext) in enumerate(dl_dict[TRAIN]):
    #     print(batch_inp.shape, batch_cam_int.shape, batch_cam_ext.shape)  # Should print [batch_size, size[0], size[1], 3] for each batch
    #     if i == 0:  # Just to break the loop after two batches for demonstration
    #         import matplotlib.pyplot as plt
    #         plt.imshow(batch_inp[0].cpu().numpy())
    #         plt.show()
    #         break

    rendered_view_train, camera_intrinsics_train, camera_extrinsics_train = train_material
    rendered_view_train = rendered_view_train.to(device)
    camera_intrinsics_train = camera_intrinsics_train.to(device)
    camera_extrinsics_train = camera_extrinsics_train.to(device)
    # Validation images can remain on CPU
    rendered_view_valid, camera_intrinsics_valid, camera_extrinsics_valid = valid_material
    rendered_view_valid = rendered_view_valid.cpu()
    n_steps = 200
    color_pred = torch.nn.Parameter(torch.randn((num_samples, 1, pseudo_color_dim), requires_grad=True, device=device))
    model, optim = get_training_content(config, training_mode=True, extra_params=[color_pred])
    out_dir_train = TRAINING_DIR/f"__{exp:04d}"
    out_dir_train.mkdir(exist_ok=True, parents=True)
    training_loop(
        model,
        optim,
        wc_points,
        wc_normals,
        color_pred,
        dl_dict,
        config,
        device=device,
        output_dir=out_dir_train,
        wandb_flag=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a scene using BlenderProc")
    parser.add_argument("-s", "--scene", type=str, help="Name of the scene to render", default=STAIRCASE)
    parser.add_argument("-e", "--experiment", type=int, nargs="+", help="Training experiment", default=None)
    args = parser.parse_args()
    main(name=args.scene)
