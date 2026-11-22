"""Import a DA3 reconstruction (folder written by scripts/infer.py) into Blender.

Usage, either:
  * Blender > Scripting > open this file > Run Script (it reads the folder it is saved in), or
  * blender --python blender_import.py -- --dir /path/to/output [--no-mesh] [--no-points] [--point-size 0.004]

Creates:
  * "DA3_points": the coloured point cloud (points.ply) shown as points via Geometry Nodes,
  * "DA3_mesh":   the GLB mesh (if present),
  * "DA3_cam_XXX": one camera per input view, with the input photo as camera background.

Coordinate conventions:
  exports are in the OpenCV frame of camera 0 (x right, y down, z forward). Blender is Z-up, and
  its cameras look along -Z with +Y up (OpenGL). We map
      world:  p_blender = R_w p_cv,      R_w = [[1,0,0],[0,0,1],[0,-1,0]]   (x, y, z) -> (x, z, -y)
      camera: M_cam = R_w * c2w_cv * diag(1, -1, -1)                       (OpenCV -> OpenGL axes)
  so camera 0 ends up at the origin looking along +Y with the image up = +Z. The GLB is written in
  glTF axes (x, -y, -z), which Blender's glTF importer converts with the same R_w.
"""

import json
import math
import os
import sys

import bpy
from mathutils import Matrix

R_W = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))
CV_TO_GL = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    opts = {"dir": None, "mesh": True, "points": True, "point_size": None}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--dir":
            opts["dir"], i = argv[i + 1], i + 1
        elif a == "--no-mesh":
            opts["mesh"] = False
        elif a == "--no-points":
            opts["points"] = False
        elif a == "--point-size":
            opts["point_size"], i = float(argv[i + 1]), i + 1
        i += 1
    if opts["dir"] is None:
        # Run from Blender's Text Editor, __file__ is only the text-block name ("\blender_import.py"),
        # so look at the on-disk path of the opened text blocks, then __file__, then the .blend folder.
        cands = [bpy.path.abspath(t.filepath) for t in bpy.data.texts
                 if t.filepath and os.path.basename(t.filepath) == "blender_import.py"]
        if "__file__" in globals():
            cands.append(os.path.abspath(__file__))
        cands = [os.path.dirname(c) for c in cands] + [bpy.path.abspath("//")]
        opts["dir"] = next((c for c in cands if os.path.exists(os.path.join(c, "transforms.json"))), None)
        if opts["dir"] is None:
            raise FileNotFoundError("transforms.json not found next to blender_import.py; open the script "
                                    "from the unzipped folder (Text > Open), or set opts['dir'] by hand")
    return opts


def import_ply(path):
    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "ply_import"):          # Blender >= 3.6
        bpy.ops.wm.ply_import(filepath=path)
    else:                                          # older Blender
        bpy.ops.import_mesh.ply(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    return new[0] if new else None


def point_material(color_attr):
    mat = bpy.data.materials.new("DA3_point_color")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = color_attr
    nt.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    if "Emission Color" in bsdf.inputs:            # Blender 4.x: make points readable regardless of lights
        nt.links.new(attr.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = 1.0
    return mat


def show_as_points(obj, radius):
    """Geometry Nodes: Mesh to Points -> Set Material (colour from the PLY vertex colours)."""
    color_attr = obj.data.color_attributes[0].name if len(obj.data.color_attributes) else "Col"
    mod = obj.modifiers.new("DA3_points", "NODES")
    tree = bpy.data.node_groups.new("DA3_points", "GeometryNodeTree")
    if hasattr(tree, "interface"):                 # Blender >= 4.0
        tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    else:
        tree.inputs.new("NodeSocketGeometry", "Geometry")
        tree.outputs.new("NodeSocketGeometry", "Geometry")
    n_in = tree.nodes.new("NodeGroupInput")
    n_out = tree.nodes.new("NodeGroupOutput")
    to_pts = tree.nodes.new("GeometryNodeMeshToPoints")
    to_pts.inputs["Radius"].default_value = radius
    set_mat = tree.nodes.new("GeometryNodeSetMaterial")
    set_mat.inputs["Material"].default_value = point_material(color_attr)
    tree.links.new(n_in.outputs[0], to_pts.inputs["Mesh"])
    tree.links.new(to_pts.outputs["Points"], set_mat.inputs["Geometry"])
    tree.links.new(set_mat.outputs["Geometry"], n_out.inputs[0])
    mod.node_group = tree


def add_camera(name, frame, W, H, image_path, scale):
    cam = bpy.data.cameras.new(name)
    fx, fy, cx, cy = frame["fl_x"], frame["fl_y"], frame["cx"], frame["cy"]
    cam.sensor_fit = "HORIZONTAL" if W >= H else "VERTICAL"
    cam.sensor_width = 36.0
    cam.sensor_height = 36.0 * H / W
    cam.lens = fx * cam.sensor_width / W if W >= H else fy * cam.sensor_height / H
    big = max(W, H)
    cam.shift_x = (W / 2.0 - cx) / big
    cam.shift_y = (cy - H / 2.0) / big
    cam.display_size = 0.3 * scale
    cam.clip_start, cam.clip_end = 0.001 * scale, 1000.0 * scale
    if image_path and os.path.exists(image_path):
        cam.show_background_images = True
        bg = cam.background_images.new()
        bg.image = bpy.data.images.load(image_path, check_existing=True)
        bg.alpha = 0.6
    obj = bpy.data.objects.new(name, cam)
    c2w_gl = Matrix(frame["transform_matrix"])     # already OpenGL axes (= c2w_cv * diag(1,-1,-1))
    obj.matrix_world = R_W @ c2w_gl
    bpy.context.scene.collection.objects.link(obj)
    return obj


def main():
    o = parse_args()
    d = o["dir"]
    meta = json.load(open(os.path.join(d, "transforms.json")))
    W, H = int(meta["w"]), int(meta["h"])
    centers = [Matrix(f["transform_matrix"]).translation for f in meta["frames"]]
    spread = max(max((c - centers[0]).length for c in centers), 1e-3)
    scale = spread if len(centers) > 1 else 1.0

    if o["points"] and os.path.exists(os.path.join(d, "points.ply")):
        pts = import_ply(os.path.join(d, "points.ply"))
        if pts is not None:
            pts.name = "DA3_points"
            pts.matrix_world = R_W
            show_as_points(pts, o["point_size"] or 0.004 * scale)
    if o["mesh"] and os.path.exists(os.path.join(d, "mesh.glb")):
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=os.path.join(d, "mesh.glb"))
        for ob in set(bpy.data.objects) - before:
            ob.name = "DA3_mesh"
            ob.hide_set(o["points"])                # hidden when points are shown; toggle in the outliner

    cams = []
    for i, f in enumerate(meta["frames"]):
        img = os.path.join(d, f["file_path"])
        cams.append(add_camera(f"DA3_cam_{i:03d}", f, W, H, img, scale))
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = W, H
    # A Blender camera has one focal length; fx != fy is modelled with the (per-scene) pixel aspect.
    # pixel_aspect_y = fx / fy reproduces the OpenCV projection exactly (checked headless, 1e-4 px).
    ratios = [f["fl_x"] / f["fl_y"] for f in meta["frames"]]
    scene.render.pixel_aspect_x, scene.render.pixel_aspect_y = 1.0, sum(ratios) / len(ratios)
    if cams:
        scene.camera = cams[0]
    print(f"[DA3] imported {len(cams)} cameras from {d}; fx/fy mean {sum(ratios) / len(ratios):.4f} "
          f"(range {min(ratios):.4f}-{max(ratios):.4f})")


main()
