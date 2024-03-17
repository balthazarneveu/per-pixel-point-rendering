import random
import numpy as np
from typing import List
from pixr.camera.camera_geometry import set_camera_parameters_orbit_mode


def define_orbits(seed: int = 42, num_view=5, config={"distance": 20., "altitude": 0}) -> List[List[float]]:
    views = []
    random.seed(seed)
    distance = config["distance"]
    altitude = config["altitude"]
    for yaw_in in range(-180, 180, int(np.ceil(360/num_view))):
        pitch_in = (random.random() - 0.5) * 90.
        yaw, pitch, roll, cam_pos = set_camera_parameters_orbit_mode(
            yaw_deg=yaw_in, pitch_deg=pitch_in, roll=0., trans_z=distance, trans_y=-altitude)
        views.append([np.rad2deg(yaw.item()), np.rad2deg(pitch.item()), np.rad2deg(roll.item()),
                      -float(cam_pos[0]), -float(cam_pos[2]), -float(cam_pos[1])])
    return views


def define_random_positions(num_view: int = 5, seed: int = 42) -> List[List[float]]:
    views = []
    random.seed(seed)
    for _ in range(num_view):
        yaw, pitch, roll = random.randint(-15, 16), random.randint(-5, 1), random.randint(-5, 5)
        tx, ty, tz = random.randint(-4, 5), random.randint(-7, 2), random.randint(0, 2)
        ty += -13.741  # Take a step back away from the scene!
        tz += 2.0  # Take a step up!
        views.append([yaw, pitch, roll, tx, ty, tz])
    return views


def define_default_camera(w: int = 640, h: int = 480, f: float = 1000) -> tuple:
    return w, h, f


def define_scene(views_list=None, camera=None, mode="random", num_view=5, config=None):
    if views_list is None:
        if mode == "random":
            views_list = define_random_positions(num_view=num_view)
        elif mode == "orbit":
            views_list = define_orbits(num_view=num_view, config=config)
    if camera is None:
        w, h, f = define_default_camera()
    else:
        w, h, f = camera
    multiview_def = [view + [w, h, f] for view in views_list]
    multiviews = [define_blender_proc_camera(*single_view) for single_view in multiview_def]
    return multiviews


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
