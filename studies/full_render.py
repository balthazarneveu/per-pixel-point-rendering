import argparse
import numpy as np
import json
from config import SAMPLE_SCENES, OUT_DIR
import subprocess
NAME = "staircase"


def main(out_root=OUT_DIR, scene_root=SAMPLE_SCENES, name=NAME, w=640, h=480, f=1000, debug=False):
    scene_path = scene_root/f"{name}.obj"
    assert scene_path.exists(), f"Scene {scene_path} does not exist"
    out_dir = out_root/f"{name}"
    out_dir.mkdir(exist_ok=True, parents=True)
    full_camera_paths = []
    roll = 0
    for yaw in range(-30, 30, 5):
        pretty_name = f"{name}_{yaw}"
        position = [0, -13.741, 0.]
        rotation_angles = [np.deg2rad(90.), 0+np.deg2rad(roll), np.deg2rad(yaw)]
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
            "euler_rotation": rotation_angles
        }
        camera_path = out_dir/f"camera_params_{pretty_name}.json"
        with open(str(camera_path), "w") as file_out:
            json.dump(camera_dict, file_out)
        full_camera_paths.append(str(camera_path))
    current_image = out_dir/"0.hdf5"
    if not current_image.exists() or debug:
        subprocess.run([
            "blenderproc",
            "debug" if debug else "run",
            "studies/blender_proc_exports.py",
            "--scene", str(scene_path),
            "--output-dir", str(out_dir),
            "--camera",] + full_camera_paths,
        )

    subprocess.run([
        "blenderproc", "vis", "hdf5", str(current_image)
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a scene using BlenderProc")
    parser.add_argument("-s", "--scene", type=str, help="Name of the scene to render", default=NAME)
    parser.add_argument("-d", "--debug", action="store_true", help="Run BlenderProc in debug mode")
    args = parser.parse_args()
    main(name=args.scene, debug=args.debug)
