import argparse
import numpy as np
import json
from config import SAMPLE_SCENES, OUT_DIR
import subprocess
import h5py
import shutil
import numpy as np
from pathlib import Path
from interactive_pipe.data_objects.image import Image
from itertools import product
NAME = "staircase"


def main(out_root=OUT_DIR, scene_root=SAMPLE_SCENES, name=NAME, w=640, h=480, f=1000, debug=False):
    scene_path = scene_root/f"{name}.obj"
    assert scene_path.exists(), f"Scene {scene_path} does not exist"
    out_dir = out_root/f"{name}"
    out_dir.mkdir(exist_ok=True, parents=True)
    full_camera_paths = []
    full_output_paths = []
    roll = 0
    pitch = 0
    yaw = 0
    view_counter = 0
    tx, ty, tz = 0, 0, 0
    # for yaw, pitch in product(range(-20, 20, 5), range(-15, 15, 5)):
    for tx, ty, tz in product(range(-5, 6, 3), range(-5, 6, 4), range(-5, 6, 4)):
        view_dir = out_dir/f"view_{view_counter:03d}"
        view_dir.mkdir(exist_ok=True, parents=True)
        full_output_paths.append(view_dir)
        pretty_name = f"{name}_{yaw}"
        position = [tx, -13.741+ty, tz]
        rotation_angles = [np.deg2rad(90.+pitch), 0+np.deg2rad(roll), np.deg2rad(yaw)]
        camera_dict = {
            "k_matrix": [
                [f, 0, w/2],
                [0, f, h/2],
                [0, 0, 1]
            ],
            "f": f,  # "focal length in pixels, should be the same as the first element of the k_matrix
            "w": w,
            "h": h,
            "position": position,
            "euler_rotation": rotation_angles,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
        }
        camera_path = view_dir/"camera_params.json"
        with open(str(camera_path), "w") as file_out:
            json.dump(camera_dict, file_out)
        full_camera_paths.append(str(camera_path))
        view_counter += 1
    # current_image = out_dir/"{yaw}.hdf5"
    # if not current_image.exists() or debug:

    subprocess.run([
        "blenderproc",
        "debug" if debug else "run",
        "studies/blender_proc_exports.py",
        "--scene", str(scene_path),
        "--output-dir", str(out_dir),
        "--camera",] + full_camera_paths,
    )

    for idx, pth in enumerate(full_output_paths):
        out_view = pth/"view.hdf5"
        shutil.move(out_dir/f"{idx}.hdf5", out_view)
        f = h5py.File(out_view, 'r')
        Image(np.array(f["colors"])/255.).save(out_view.with_suffix(".png"))
        Image(np.array(f["colors"])/255.).save(out_dir/f"{idx:04d}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a scene using BlenderProc")
    parser.add_argument("-s", "--scene", type=str, help="Name of the scene to render", default=NAME)
    parser.add_argument("-d", "--debug", action="store_true", help="Run BlenderProc in debug mode")
    args = parser.parse_args()
    main(name=args.scene, debug=args.debug)
