from pathlib import Path
from pixr.camera.camera_geometry import get_camera_extrinsics, get_camera_intrinsics, set_camera_parameters_orbit_mode
import torch
from pixr.rendering.splatting import splat_points
from interactive_plugins import define_default_sliders
from novel_views import load_colored_point_cloud_from_files
from pixr.synthesis.forward_project import project_3d_to_2d
from interactive_pipe import interactive_pipeline
from pixr.interactive.utils import tensor_to_image, rescale_image
from pixr.properties import DEVICE
from shared_parser import get_shared_parser


def infer_image(splatted_image, model):
    if model is not None:
        splatted_image = splatted_image.permute(2, 0, 1)
        splatted_image = splatted_image.unsqueeze(0)
        with torch.no_grad():
            inferred_image = model(splatted_image)
        inferred_image = inferred_image.squeeze(0).permute(1, 2, 0)
        return inferred_image
    else:
        return splatted_image


def splat_pipeline_novel_view(wc_points, wc_normals, points_colors, model):
    # yaw, pitch, roll, cam_pos = set_camera_parameters()
    yaw, pitch, roll, cam_pos = set_camera_parameters_orbit_mode()
    camera_extrinsics = get_camera_extrinsics(yaw, pitch, roll, cam_pos)
    camera_intrinsics, w, h = get_camera_intrinsics()
    cc_points, points_depths, cc_normals = project_3d_to_2d(
        wc_points, camera_intrinsics, camera_extrinsics, wc_normals)

    # # Let's splat the triangle nodes
    splatted_image = splat_points(cc_points, points_colors, points_depths, w, h, camera_intrinsics, cc_normals)
    inferred_image = infer_image(splatted_image, model)
    splatted_image = tensor_to_image(splatted_image)
    splatted_image = rescale_image(splatted_image)
    inferred_image = tensor_to_image(inferred_image)
    inferred_image = rescale_image(inferred_image)
    return splatted_image, inferred_image


def main_interactive_version(splat_scene_path, model_path=None):
    wc_points, wc_normals, color_pred = load_colored_point_cloud_from_files(splat_scene_path)
    define_default_sliders(orbit_mode=True)
    if model_path is not None:
        from pixr.learning.unet import UNet
        model = UNet()
        model.load_state_dict(torch.load(model_path))
        model.to(DEVICE)
        model.eval()
    else:
        model = None
    interactive_pipeline(
        gui="qt",
        cache=True,
        safe_input_buffer_deepcopy=False,
        size=(20, 10)
    )(splat_pipeline_novel_view)(wc_points, wc_normals, color_pred, model)


if __name__ == '__main__':
    parser = get_shared_parser()
    args = parser.parse_args()
    splat_scene_path = Path(
        # "/Data/code/per-pixel-point-rendering/__output/staircase_splat_differentiate_points/checkpoint_00200.pt",
        "/Data/code/per-pixel-point-rendering/__training/__0001/point_cloud_checkpoint_00199.pt",
    )
    assert splat_scene_path.exists()
    model_path = splat_scene_path.parent / "last_model.pt"
    if not model_path.exists():
        model_path = None
    main_interactive_version(splat_scene_path, model_path=model_path)
