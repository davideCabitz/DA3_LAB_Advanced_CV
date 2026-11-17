"""Headless check of scripts/blender_import.py with the `bpy` wheel (pip install bpy, Python 3.11).

Imports an infer.py output folder into an empty scene and verifies, for every camera, that points
projected with the exported OpenCV K / w2c land on the same pixels as with the Blender camera.

  python scripts/blender_smoke_test.py outputs/demo/<scene> [report.json]
"""

import json
import runpy
import sys

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def main(d):
    sys.argv = ["blender", "--", "--dir", d]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    runpy.run_path(f"{d}/blender_import.py", run_name="__main__")
    objs = {o.name: o for o in bpy.data.objects}
    pts = objs["DA3_points"]
    meta = json.load(open(f"{d}/transforms.json"))
    W, H = meta["w"], meta["h"]
    scene = bpy.context.scene
    idx = np.random.default_rng(0).choice(len(pts.data.vertices), 3000, replace=False)
    P = np.array([pts.data.vertices[int(i)].co[:] for i in idx])
    report = {"bpy": bpy.app.version_string, "points": len(pts.data.vertices),
              "mesh_vertices": sum(len(o.data.vertices) for o in objs.values() if o.name.startswith("DA3_mesh") and o.type == "MESH"),
              "cameras": {}}
    c0 = objs["DA3_cam_000"].matrix_world
    report["cam0_forward"] = [round(v, 3) for v in c0.to_3x3() @ Vector((0, 0, -1))]
    report["cam0_up"] = [round(v, 3) for v in c0.to_3x3() @ Vector((0, 1, 0))]
    for i, fr in enumerate(meta["frames"]):
        cam = objs[f"DA3_cam_{i:03d}"]
        K, w2c = np.array(fr["K"]), np.array(fr["w2c_opencv"])
        errs = []
        for p in P:
            xc = w2c[:3, :3] @ p + w2c[:3, 3]
            if xc[2] <= 0:
                continue
            u, v = (K @ (xc / xc[2]))[:2]
            if not (0 <= u < W and 0 <= v < H):
                continue
            ndc = world_to_camera_view(scene, cam, pts.matrix_world @ Vector(p))
            errs.append(np.hypot(ndc.x * W - u, (1 - ndc.y) * H - v))
        report["cameras"][cam.name] = {"n": len(errs), "median_px": float(np.median(errs)), "max_px": float(np.max(errs))}
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    rep = main(sys.argv[1])
    if len(sys.argv) > 2:  # optional output path for the JSON report (stdout also has Blender logs)
        with open(sys.argv[2], "w") as f:
            json.dump(rep, f, indent=2)
