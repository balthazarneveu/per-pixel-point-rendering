import random
import numpy as np
from typing import List


def define_random_positions(num_view: int, seed: int = 42) -> List[List[float]]:
    views = []
    random.seed(seed)
    for _ in range(num_view):
        yaw, pitch, roll = random.randint(-15, 16), random.randint(-8, 9), random.randint(-20, 20)
        tx, ty, tz = random.randint(-4, 5), random.randint(-7, 2), random.randint(-3, 4)
        ty += -13.741  # Take a step back away from the scene!
        views.append([yaw, pitch, roll, tx, ty, tz])
    return views


def define_default_camera(w: int = 640, h: int = 480, f: float = 1000) -> tuple:
    return w, h, f


def define_scene(views_list=None, camera=None):
    if views_list is None:
        views_list = define_random_positions(5)
    if camera is None:
        w, h, f = define_default_camera()
    multiview_def = [view + [w, h, f] for view in views_list]
    multiviews = [define_blender_proc_camera(*single_view) for single_view in multiview_def]
    return multiviews
    # view_dir = out_dir/f"view_{view_counter:03d}"
    # view_dir.mkdir(exist_ok=True, parents=True)
    # full_output_paths.append(view_dir)


def define_blender_proc_camera(yaw, pitch, roll, tx, ty, tz, w, h, fpix) -> dict:
    position = [tx, ty, tz]
    # That 90° is mandatory to bridge the conventions between blenderproc and pixr
    rotation_angles = [np.deg2rad(90.+pitch), np.deg2rad(roll), np.deg2rad(yaw)]
    camera_dict = {
        "k_matrix": [
            [fpix, 0, w/2],
            [0, fpix, h/2],
            [0, 0, 1]
        ],
        "f": fpix,  # "focal length in pixels, should be the same as the first element of the k_matrix
        "w": w,
        "h": h,
        "position": position,
        "euler_rotation": rotation_angles,
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
    }
    return camera_dict


if __name__ == "__main__":
    multiview_scene = define_scene()
    print("Done", multiview_scene)
