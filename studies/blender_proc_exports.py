import blenderproc as bproc
import argparse
import json
import bpy


parser = argparse.ArgumentParser(description="Render a scene using BlenderProc use blenderproc run")
parser.add_argument('-c', '--camera', nargs="+",
                    help="Path to the camera file, should be examples/resources/camera_positions")
parser.add_argument('-s', '--scene', help="Path to the scene.obj file, should be examples/resources/scene.obj")
parser.add_argument(
    '-o', '--output-dir',  help="Path to where the final files, will be saved")
parser.add_argument('--backface-culling', help="Enable backface culling", action="store_true")
parser.add_argument('--background-map', help="Path to the background map")
args = parser.parse_args()

bproc.init()
if args.scene.endswith(".blend"):
    objs = bproc.loader.load_blend(args.scene)
else:
    objs = bproc.loader.load_obj(args.scene)
    if args.backface_culling:
        materials = bproc.material.collect_all()
        for idx, mat in enumerate(materials):
            bpy.context.object.active_material_index = idx
            bpy.context.object.active_material.use_backface_culling = True

if args.background_map is not None:
    print(f"Setting background map to {args.background_map}")
    bproc.world.set_world_background_hdr_img(args.background_map)

for idx, camera_file in enumerate(args.camera):
    with open(camera_file) as fi:
        data = json.load(fi)
        w = data["w"]
        h = data["h"]
        k_matrix = data["k_matrix"]
        position = data["position"]
        euler_rotation = data["euler_rotation"]
    bproc.camera.set_resolution(h, w)
    bproc.camera.set_intrinsics_from_K_matrix(k_matrix, w, h)
    matrix_world = bproc.math.build_transformation_mat(position, euler_rotation)
    bproc.camera.add_camera_pose(matrix_world)
# Enable transparency so the background becomes transparent
bproc.renderer.set_output_format(enable_transparency=True)
data = bproc.renderer.render()

bproc.writer.write_hdf5(args.output_dir, data)
