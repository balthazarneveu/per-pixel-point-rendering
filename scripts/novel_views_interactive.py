from interactive_pipe import interactive
from pathlib import Path
from pixr.camera.camera_geometry import get_camera_extrinsics, get_camera_intrinsics, set_camera_parameters_orbit_mode
import torch
from pixr.rendering.splatting import splat_points
from pixr.interactive.interactive_plugins import define_default_sliders
# from novel_views import load_colored_point_cloud_from_files
from pixr.learning.utils import load_model
from pixr.rendering.forward_project import project_3d_to_2d
from interactive_pipe import interactive_pipeline
from pixr.interactive.utils import tensor_to_image, rescale_image
from pixr.properties import DEVICE, SCALE_LIST
from shared_parser import get_shared_parser
from experiments_definition import get_experiment_from_id
from pixr.learning.experiments import get_training_content
from typing import Optional
from pixr.rendering.splatting import ms_splatting


def infer_image(splatted_image, model, global_params={}) -> torch.Tensor:
    if model is not None:
        with torch.no_grad():
            ms_list = [spl.permute(2, 0, 1).unsqueeze(0) for spl in splatted_image]
            with torch.no_grad():
                inferred_image = model(ms_list)[global_params.get("scale", 0)]
            inferred_image = inferred_image.squeeze(0).permute(1, 2, 0)
            return inferred_image
    else:
        return splatted_image


def apply_pca_to_tensor(tensor, n_components=3):
    """
    Applies PCA on a tensor of shape (H, W, C) to reduce it to (H, W, n_components).

    Parameters:
    - tensor: Input tensor of shape (H, W, C)
    - n_components: Number of principal components to keep

    Returns:
    - pca_tensor: Tensor of shape (H, W, n_components) after PCA
    """

    # Validate inputs
    if tensor.dim() != 3 or tensor.size(2) < n_components:
        raise ValueError("Input tensor must be of shape (H, W, C) with C >= n_components.")

    H, W, C = tensor.shape

    # Flatten the (H, W) dimensions
    flat_tensor = tensor.reshape(-1, C)

    # Standardize the features
    mean = flat_tensor.mean(dim=0)
    std = flat_tensor.std(dim=0)
    standardized_tensor = (flat_tensor - mean) / std
    try:
        # Perform SVD, which is equivalent to PCA since the data is centered
        U, S, V = torch.svd(standardized_tensor)
    except Exception as e:
        print(e)
        return tensor[:, :n_components]
    # Keep the top n_components
    principal_components = V[:, :n_components]

    # Project the data onto the top n_components
    pca_result = torch.mm(standardized_tensor, principal_components)

    # Reshape back to (H, W, n_components)
    pca_tensor = pca_result.reshape(H, W, n_components)

    return pca_tensor


@interactive(pca_flag=(True,))
def debug_splat(img_ms, pca_flag=True, global_params={}):
    seleced_scale = global_params.get('scale', 0)
    if pca_flag:
        selected_full_res = apply_pca_to_tensor(img_ms[seleced_scale])
    else:
        selected_full_res = img_ms[seleced_scale][:, :, :3]

    return rescale_image(tensor_to_image(selected_full_res).clip(0, 1), global_params=global_params)


def splat_pipeline_novel_view(wc_points, wc_normals, points_colors, model, scales_list):
    # yaw, pitch, roll, cam_pos = set_camera_parameters()
    yaw, pitch, roll, cam_pos = set_camera_parameters_orbit_mode()
    camera_extrinsics = get_camera_extrinsics(yaw, pitch, roll, cam_pos)
    camera_intrinsics, w, h = get_camera_intrinsics()
    cc_points, points_depths, cc_normals = project_3d_to_2d(
        wc_points, camera_intrinsics, camera_extrinsics, wc_normals)

    # # Let's splat the triangle nodes
    # splatted_image = splat_points(cc_points, points_colors, points_depths, w, h, camera_intrinsics, cc_normals)
    splatted_image = ms_splatting(cc_points, points_colors, points_depths, w,
                                  h, camera_intrinsics, cc_normals, scales_list)
    inferred_image = infer_image(splatted_image, model)
    inferred_image = tensor_to_image(inferred_image)
    inferred_image = rescale_image(inferred_image)
    splatted_image_debug = debug_splat(splatted_image)
    return inferred_image, splatted_image_debug


def main_interactive_version(exp, training_dir):
    config = get_experiment_from_id(exp)
    model, optim = get_training_content(config, training_mode=False)
    # model_path = training_dir / f"__{exp:04d}" / "best_model.pt"
    model_path = training_dir / f"__{exp:04d}" / "last_model.pt"
    # wc_points, wc_normals, color_pred = load_colored_point_cloud_from_files(splat_scene_path)
    model_state_dict, wc_points, wc_normals, color_pred = load_model(model_path)
    wc_points = wc_points.detach().to(DEVICE)
    wc_normals = wc_normals.detach().to(DEVICE)
    color_pred = color_pred.detach().to(DEVICE)

    define_default_sliders(orbit_mode=True, multiscale=model.n_scales)
    if model_path is not None:
        model.load_state_dict(model_state_dict)
        model.to(DEVICE)
        model.eval()
    else:
        model = None
    interactive_pipeline(
        gui="qt",
        cache=True,
        safe_input_buffer_deepcopy=False,
        size=(20, 15)
    )(splat_pipeline_novel_view)(wc_points, wc_normals, color_pred, model, config[SCALE_LIST])


if __name__ == '__main__':
    parser = get_shared_parser()
    args = parser.parse_args()
    # splat_scene_path = Path(
    #     # "/Data/code/per-pixel-point-rendering/__output/staircase_splat_differentiate_points/checkpoint_00200.pt",
    #     "/Data/code/per-pixel-point-rendering/__training/__0001/point_cloud_checkpoint_00199.pt",
    # )
    # assert splat_scene_path.exists()
    # model_path = splat_scene_path.parent / "last_model.pt"
    # if not model_path.exists():
    # model_path = None
    exp = args.experiment[0]

    main_interactive_version(exp, training_dir=args.training_dir)
