from pixr.learning.experiments import get_training_content
import torch
from pixr.learning.utils import prepare_dataset
from config import OUT_DIR, TRAINING_DIR
from pixr.properties import (SCENE, DEVICE, TRAIN, VALIDATION, METRIC_PSNR, LOSS, LOSS_MSE, NB_EPOCHS, LR,
                             RATIO_TRAIN, PSEUDO_COLOR_DIMENSION, NB_POINTS, SCALE_LIST
                             )
from tqdm import tqdm
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
from shared_parser import get_shared_parser
from typing import List
from pixr.learning.utils import save_model


def infer_function(point_cloud, cam_int, cam_ext, wc_normals, colors, w, h, scale=0, no_grad=False):
    wc_normals = wc_normals.to(point_cloud.device)
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
        normal_culling_flag=True  # !WARNING ONLY DISABLE NORMAL CULLING TO MATCH BLENDER FOR OPEN OBJECTS!
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
    best_accuracy = 0
    scales_list = config[SCALE_LIST]
    model.to(device)
    for n_epoch in tqdm(range(config[NB_EPOCHS])):
        current_metrics = {
            TRAIN: 0.,
            VALIDATION: 0.,
            LR: optimizer.param_groups[0]['lr'],
            METRIC_PSNR: 0.,
        }
        for phase in [TRAIN, VALIDATION]:
            for scale in scales_list:
                current_metrics[f"{phase}_MSE_{scale}"] = 0.
        for phase in [TRAIN, VALIDATION]:
            if phase == TRAIN:
                model.train()
            else:
                model.eval()
            for target_view, cam_int, cam_ext in tqdm(dl_dict[phase], desc=f"{phase} - Epoch {n_epoch}"):
                target_view = target_view.to(device)
                h, w = target_view.shape[-2:]
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == TRAIN):
                    loss = 0
                    multiscale_batches = []
                    for scale in scales_list:
                        batch_splat = []
                        for img_index in range(cam_int.shape[0]):
                            img_splat = infer_function(
                                wc_points, cam_int[img_index], cam_ext[img_index], wc_normals, color_pred, w, h,
                                scale=scale
                            )
                            batch_splat.append(img_splat.permute(2, 0, 1))
                        batch_splat = torch.stack(batch_splat)
                        multiscale_batches.append(batch_splat)
                    image_pred = model(multiscale_batches)

                    img_target = [torch.nn.functional.avg_pool2d(
                        target_view, 2**sc) if sc > 0 else target_view for sc in scales_list]
                    from matplotlib import pyplot as plt
                    # plt.imshow(img_target[0][0].permute(1, 2, 0).cpu().numpy())
                    # plt.show()
                    # plt.imshow(img_target[1][0].permute(1, 2, 0).cpu().numpy())
                    # plt.show()

                    # >>> Multiscale supervision <<<
                    for scale_idx, scale in enumerate(scales_list):
                        per_scale_loss = compute_loss(image_pred[scale_idx], img_target[scale_idx],
                                                      mode=config.get(LOSS, LOSS_MSE))
                        # print(f"Loss at scale {scale} is {per_scale_loss}")
                        loss += per_scale_loss
                        current_metrics[f"{phase}_MSE_{scale}"] += per_scale_loss.item()
                        # if phase == VALIDATION and n_epoch % 10 == 0:
                        if n_epoch % 20 == 0 and phase == VALIDATION:
                            plt.figure(figsize=(10, 10))
                            n_img = img_target[scale_idx].shape[0]
                            for img_idx in range(n_img):
                                plt.subplot(n_img, 2, img_idx*2+1)
                                plt.imshow(img_target[scale_idx][img_idx].permute(1, 2, 0).clip(0, 1).cpu().numpy())
                                plt.subplot(n_img, 2, img_idx*2+2)
                                plt.imshow(image_pred[scale_idx][img_idx].permute(
                                    1, 2, 0).clip(0, 1).cpu().detach().numpy())
                            plt.savefig(output_dir/f"{phase}_{n_epoch:05d}_scale_{scale}.png")
                            plt.close()
                    if torch.isnan(loss):
                        print(f"Loss is NaN at epoch {n_epoch} and phase {phase}!")
                        continue
                    if phase == TRAIN:
                        loss.backward()
                        optimizer.step()
                current_metrics[phase] += loss.item()
                if phase == VALIDATION:
                    metrics_on_batch = compute_metrics(image_pred[0], img_target[0])
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
        for phase in [TRAIN, VALIDATION]:
            debug_print = f"{phase}: Epoch {n_epoch} - Loss: {current_metrics[phase]:.3e} "
            for k, v in current_metrics.items():
                if phase not in k and k not in [TRAIN, VALIDATION, LR]:
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
                # torch.save(model.state_dict(), output_dir/"best_model.pt")
                save_model(model, wc_points, wc_normals, color_pred, output_dir/"best_model.pt")

        # save_model(model, wc_points, wc_normals, color_pred, output_dir/f"point_cloud_checkpoint_{n_epoch:05d}.pt")
    if output_dir is not None:
        # torch.save(model.cpu().state_dict(), output_dir/"last_model.pt")
        save_model(model, wc_points, wc_normals, color_pred, output_dir/"last_model.pt")
    return model


def main(exp: int, out_root=OUT_DIR, device=DEVICE):
    config = get_experiment_from_id(exp)
    num_samples = config[NB_POINTS]
    pseudo_color_dim = config[PSEUDO_COLOR_DIMENSION]
    name = config[SCENE]
    train_material, valid_material, (w, h), point_cloud_material = prepare_dataset(
        out_root, name, num_samples=num_samples, ratio_train=config.get(RATIO_TRAIN, 0.8)
    )
    # Move training data to GPU
    wc_points, wc_normals = point_cloud_material
    wc_points = wc_points.to(device)
    wc_normals = wc_normals.to(device)
    dl_dict = get_data_loader(config, train_material, valid_material)

    rendered_view_train, camera_intrinsics_train, camera_extrinsics_train = train_material
    rendered_view_train = rendered_view_train.to(device)
    camera_intrinsics_train = camera_intrinsics_train.to(device)
    camera_extrinsics_train = camera_extrinsics_train.to(device)
    # Validation images can remain on CPU
    rendered_view_valid, camera_intrinsics_valid, camera_extrinsics_valid = valid_material
    rendered_view_valid = rendered_view_valid.cpu()
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


def loop_over_experiments(exp_list: List[int]):
    for exp in exp_list:
        main(exp=exp)


if __name__ == "__main__":
    parser = get_shared_parser(description="Neural point based rendering training")
    args = parser.parse_args()
    loop_over_experiments(args.experiment)
