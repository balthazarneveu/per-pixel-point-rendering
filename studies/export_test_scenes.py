from pixr.synthesis.world_simulation import generate_simulated_world, ALL_SCENE_MODES, TEST_RECT, TEST_TRIANGLES, STAIRCASE
from config import SAMPLE_SCENES
from pathlib import Path


def export_obj_and_mtl_files(wc_triangles_original, colors_nodes, name: str, out_dir: Path = SAMPLE_SCENES):
    wc_triangles = wc_triangles_original.permute(0, 2, 1)
    obj_lines_with_mtl = [f"mtllib  {name}.mtl"]  # Reference to the .mtl file
    material_counter = 1  # To assign materials correctly
    vertex_count = 1  # OBJ files are 1-indexed
    for i, triangle in enumerate(wc_triangles):
        obj_lines_with_mtl.append(f"usemtl material{material_counter}")  # Assign material to the face
        for vertex in triangle[:3]:  # Ignore the w component
            obj_lines_with_mtl.append(f"v {' '.join(map(str, vertex[:3].numpy()))}")
        # Faces are defined by indices of previously defined vertices
        obj_lines_with_mtl.append(f"f {vertex_count} {vertex_count + 1} {vertex_count + 2}")
        vertex_count += 3  # Move to the next set of vertices
        material_counter += 1  # Move to the next material
    obj_content = "\n".join(obj_lines_with_mtl)

    # Calculate the average color for each triangle
    print(colors_nodes)
    average_colors = colors_nodes.mean(dim=1)
    print(average_colors)

    # Generate .mtl content
    mtl_lines = []
    for i, color in enumerate(average_colors):
        # Normalize the color values to [0, 1] range and create a material for each triangle
        normalized_color = color.numpy()
        mtl_lines.append(f"newmtl material{i+1}")
        mtl_lines.append("Ns 323.999994")
        mtl_lines.append("Ka 1.000000 1.000000 1.000000")
        mtl_lines.append(f"Kd {normalized_color[0]} {normalized_color[1]} {normalized_color[2]}")
        mtl_lines.append("Ke 0.0 0.0 0.0")
        mtl_lines.append("Ni 1.450000")
        mtl_lines.append("d 1.000000")

    # Combine lines into a single string for the .mtl file
    mtl_content = "\n".join(mtl_lines)
    mtl_content

    # Write the .obj and .mtl files
    obj_file_path = out_dir/f"{name}.obj"
    mtl_file_path = out_dir/f"{name}.mtl"

    with open(obj_file_path, "w") as obj_file:
        obj_file.write(obj_content)

    with open(mtl_file_path, "w") as mtl_file:
        mtl_file.write(mtl_content)


def export_scene(name: str = STAIRCASE):
    # Generate vertices and colors
    assert name in ALL_SCENE_MODES
    wc_triangles, colors_nodes = generate_simulated_world(z=0., delta_z=2., scene_mode=name)
    export_obj_and_mtl_files(wc_triangles, colors_nodes, name)
    print(f"Files {name}.obj and {name}.mtl have been saved to {SAMPLE_SCENES}.")


if __name__ == "__main__":
    for scene in ALL_SCENE_MODES:
        export_scene(name=scene)
