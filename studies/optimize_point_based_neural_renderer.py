from pixr.synthesis.world_simulation import STAIRCASE
from pixr.learning.utils import prepare_dataset
import torch

from config import OUT_DIR
from pixr.properties import DEVICE
import argparse


def main(out_root=OUT_DIR, name=STAIRCASE, device=DEVICE, show=True, save=False):
    train_material, valid_material, (w, h), point_cloud_material = prepare_dataset(out_root, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a scene using BlenderProc")
    parser.add_argument("-s", "--scene", type=str, help="Name of the scene to render", default=STAIRCASE)
    parser.add_argument("-e", "--export", action="store_true", help="Export validation image")
    args = parser.parse_args()
    main(name=args.scene, show=False, save=args.export)
