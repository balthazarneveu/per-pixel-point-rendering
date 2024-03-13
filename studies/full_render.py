import argparse
import numpy as np
import json
from config import SAMPLE_SCENES, OUT_DIR
from pixr.properties import CAMERA_PARAMS_FILE, RGB_VIEW_FILE
import subprocess
import h5py
import shutil
from pathlib import Path
from interactive_pipe.data_objects.image import Image
from pixr.multiview.scene_definition import define_scene
from typing import List, Tuple
NAME = "staircase"


def prepare_multiviews_on_disk_for_blender_proc(
    out_dir,
    views_list=None,
    w=640, h=480, f=1000,
) -> Tuple[List[Path], List[Path]]:

    out_dir.mkdir(exist_ok=True, parents=True)
    full_output_paths = []
    full_camera_paths = []
    multiview_list = define_scene(views_list=views_list, camera=(w, h, f))
    for view_counter, camera_dict in enumerate(multiview_list):
        view_dir = out_dir/f"view_{view_counter:03d}"
        view_dir.mkdir(exist_ok=True, parents=True)
        full_output_paths.append(view_dir)

        camera_path = view_dir/CAMERA_PARAMS_FILE
        with open(str(camera_path), "w") as file_out:
            json.dump(camera_dict, file_out)
        full_camera_paths.append(str(camera_path))
    return full_camera_paths, full_output_paths


def main(out_root=OUT_DIR, scene_root=SAMPLE_SCENES, name=NAME, w=640, h=480, f=1000, debug=False):
    scene_path = scene_root/f"{name}.obj"
    assert scene_path.exists(), f"Scene {scene_path} does not exist"
    out_dir = out_root/(f"{name}" + ("debug" if debug else ""))
    full_camera_paths, full_output_paths = prepare_multiviews_on_disk_for_blender_proc(
        out_dir,
        views_list=None,
        w=w, h=h, f=f
    )
    for pth in full_output_paths:
        assert not (Path(pth)/"view.hdf5").exists(), f"View {pth} already exists - do not overwrite"
    subprocess.run([
        "blenderproc",
        "debug" if debug else "run",
        "studies/blender_proc_exports.py",
        "--scene", str(scene_path),
        "--output-dir", str(out_dir),
        "--camera",] + full_camera_paths
    )

    for idx, pth in enumerate(full_output_paths):
        out_view = pth/"view.hdf5"
        shutil.move(out_dir/f"{idx}.hdf5", out_view)
        f = h5py.File(out_view, 'r')
        Image(np.array(f["colors"])/255.).save(out_view.parent/RGB_VIEW_FILE)
        Image(np.array(f["colors"])/255.).save(out_dir/f"{idx:04d}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a scene using BlenderProc")
    parser.add_argument("-s", "--scene", type=str, help="Name of the scene to render", default=NAME)
    parser.add_argument("-d", "--debug", action="store_true", help="Run BlenderProc in debug mode")
    args = parser.parse_args()
    main(name=args.scene, debug=args.debug)
