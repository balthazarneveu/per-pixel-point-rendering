from pixr.synthesis.world_simulation import generate_simulated_world, STAIRCASE
from pixr.synthesis.normals import extract_normals
from pixr.synthesis.extract_point_cloud import pick_point_cloud_from_triangles
from pixr.synthesis.forward_project import project_3d_to_2d
from interactive_pipe.data_objects.image import Image
from pixr.rendering.splatting import splat_points
from pixr.rasterizer.rasterizer import shade_screen_space
from pixr.multiview.scenes_utils import load_views
import torch
import numpy as np
from config import OUT_DIR
import argparse


def tensor_to_image(image: torch.Tensor) -> np.ndarray:
    image = image.cpu().numpy()
    return image


def main(out_root=OUT_DIR, name=STAIRCASE, splat_flag=True, raster_flag=True):
    # scene_path = scene_root/f"{name}.obj"
    # assert scene_path.exists(), f"Scene {scene_path} does not exist"
    view_dir = out_root/f"{name}"
    views = sorted(list(view_dir.glob("view_*")))
    out_dir = out_root/f"{name}_splat"
    out_dir.mkdir(exist_ok=True, parents=True)

    wc_triangles, colors = generate_simulated_world(scene_mode=name)
    wc_normals = extract_normals(wc_triangles)
    wc_points, points_colors, wc_normals = pick_point_cloud_from_triangles(
        wc_triangles, colors, wc_normals, num_samples=20000)
    # config_views_path = [current_view_path/"camera_params.json" for idx, current_view_path in enumerate(views)]
    views = load_views(views)
    for idx, view_dict in enumerate(views):
        camera_intrinsics, w, h = view_dict["camera_intrinsics"], view_dict["w"], view_dict["h"]
        camera_extrinsics = view_dict["camera_extrinsics"]
        if splat_flag:
            print(camera_extrinsics, camera_intrinsics)
            cc_points, points_depths, cc_normals = project_3d_to_2d(
                wc_points, camera_intrinsics, camera_extrinsics, wc_normals)
            splatted_image = splat_points(cc_points, points_colors, points_depths, w, h, camera_intrinsics, cc_normals)

            splatted_image = tensor_to_image(splatted_image)
            Image(splatted_image).save(out_dir/f"{idx:04}_splat.png")

        if raster_flag:
            cc_triangles, triangles_depths, _ = project_3d_to_2d(
                wc_triangles, camera_intrinsics, camera_extrinsics, None)
            rendered_image = shade_screen_space(cc_triangles, colors, triangles_depths, w, h)
            Image(tensor_to_image(rendered_image)).save(out_dir/f"{idx:04}_raster.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a scene using BlenderProc")
    parser.add_argument("-s", "--scene", type=str, help="Name of the scene to render", default=STAIRCASE)
    args = parser.parse_args()
    main(name=args.scene, splat_flag=True, raster_flag=True)
