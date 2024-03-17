from pathlib import Path
from pixr.camera.camera_geometry import get_camera_extrinsics, get_camera_intrinsics, set_camera_parameters_orbit_mode
import torch
from pixr.rendering.splatting import splat_points
from pixr.interactive.interactive_plugins import define_default_sliders
# from novel_views import load_colored_point_cloud_from_files
from pixr.learning.utils import load_model
from pixr.synthesis.forward_project import project_3d_to_2d
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
    return inferred_image


def main_interactive_version(exp, training_dir):
    config = get_experiment_from_id(exp)
    model, optim = get_training_content(config, training_mode=False)
    model_path = training_dir / f"__{exp:04d}" / "best_model.pt"
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
