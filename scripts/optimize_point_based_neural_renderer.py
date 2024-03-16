from pixr.synthesis.world_simulation import STAIRCASE
from pixr.learning.utils import prepare_dataset
import torch

from config import OUT_DIR
from pixr.properties import DEVICE
import argparse
# from experiments_definition import
DEFAULT_CONFIG = {}


def main(out_root=OUT_DIR, name=STAIRCASE, device=DEVICE, show=True, save=False, config: dict = DEFAULT_CONFIG):
    train_material, valid_material, (w, h), point_cloud_material = prepare_dataset(out_root, name)
    # Move training data to GPU
    wc_points, wc_normals = point_cloud_material
    wc_points = wc_points.to(device)
    wc_normals = wc_normals.to(device)
    rendered_view_train, camera_intrinsics_train, camera_extrinsics_train = train_material
    rendered_view_train = rendered_view_train.to(device)
    camera_intrinsics_train = camera_intrinsics_train.to(device)
    camera_extrinsics_train = camera_extrinsics_train.to(device)
    # Validation images can remain on CPU
    rendered_view_valid, camera_intrinsics_valid, camera_extrinsics_valid = valid_material
    rendered_view_valid = rendered_view_valid.cpu()
    n_steps = 200

    # # Intitialize trainable parameters
    # optimizer
    # optimizer = torch.optim.Adam([color_pred], lr=0.3)
    # for step in range(n_steps):
    #     optimizer.zero_grad()
    #     loss = 0.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a scene using BlenderProc")
    parser.add_argument("-s", "--scene", type=str, help="Name of the scene to render", default=STAIRCASE)
    parser.add_argument("-e", "--export", action="store_true", help="Export validation image")
    args = parser.parse_args()
    main(name=args.scene, show=False, save=args.export)
