"""Turntable / showcase video of a DA3 reconstruction in Blender (run AFTER blender_import.py).

Blender renders a short looping video (`turntable_<mode>.mp4`) of the reconstruction into its
folder, ready to drop into a slide. The camera motion is chosen from the scene itself:

  * "orbit":  the photos surround the subject (you walked around a statue / a building):
              a full 360° turn around it;
  * "arc":    the photos cover one side only (a façade, a palace front, a statue from 3-4 photos):
              a smooth back-and-forth swing over the photographed side, so the empty back is never
              shown;
  * "inside": the photos were taken from inside, looking outwards (a room, a pizzeria):
              the camera stands at the photographers' position and pans (360° if the photos cover
              all around, otherwise a swing over the covered directions), with a wide lens.

Usage, either:
  * Blender GUI: run blender_import.py first (Scripting > Open > Run Script), then open this file
    the same way and Run Script. Blender looks frozen while it renders (a few minutes); the video
    is written next to this script. Settings: edit CONFIG below (or keep "auto"). If the
    reconstruction is not imported yet, this script imports it first (blender_import.py must be in
    the same folder).
  * Terminal, no GUI (import + video in one command):
      blender -b --python blender_import.py --python blender_turntable.py -- --dir <folder> [options]
    options: --mode auto|orbit|arc|inside  --seconds 8  --fps 30  --resolution 1920x1080
             --background white|black|#RRGGBB  --transparent  --elevation 20  --zoom 1.2  --lens 35
             --point-scale 1.0  --crop 2.0  --with-mesh  --samples 16  --no-render

What it does to look good on any scene:
  * centre = median of the points, size = 90th percentile of their distances (far "sky" points do
    not shrink the subject); points farther than `crop` x that size are hidden (sky shells, stray
    background), without changing the files;
  * up = mean "up" direction of the photo cameras, so the scene stands upright even if the first
    photo was tilted; the motion starts on the photographed side;
  * point size from the measured point spacing (closes the gaps between neighbouring points) and at
    least ~1 pixel at the render resolution (no flicker), times `point_scale`;
  * points rendered with their true colours (emission, "Standard" view transform): the background
    colour does not tint them; H.264 MP4 (yuv420p) plays in PowerPoint, Google Slides and browsers.
"""

import glob
import math
import os
import runpy
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

CONFIG = {
    "mode": "auto",             # "auto" | "orbit" | "arc" | "inside"
    "seconds": 8.0,             # length of the loop
    "fps": 30,
    "resolution": (1920, 1080),
    "background": "white",      # "white" | "black" | "#RRGGBB"
    "transparent": False,       # True: PNG frames with alpha (an MP4 cannot store transparency)
    "elevation": None,          # camera height in degrees; None = per mode (orbit 20, arc 12, inside -5)
    "zoom": 1.2,                # orbit / arc: 1 = the subject (90% of the points) fits; larger = closer
    "lens": None,               # focal length in mm (36 mm sensor); None = per mode (35, or 18 inside)
    "point_scale": 1.0,         # multiplies the automatic point size
    "crop": 2.0,                # hide points farther than crop x the subject size (None = keep all)
    "with_mesh": False,         # also render the TSDF mesh (DA3_mesh)
    "samples": 16,              # EEVEE samples per frame (more = smoother, slower)
    "render": True,             # False: only create the camera (preview: View > Cameras > Active Camera, Space)
    "dir": None,                # output folder; None = the folder of this script / the reconstruction
}
RIG = ("DA3_turntable_base", "DA3_turntable_spin", "DA3_turntable_cam")
MODE_DEFAULTS = {"orbit": {"elevation": 20.0, "lens": 35.0},
                 "arc": {"elevation": 12.0, "lens": 35.0},
                 "inside": {"elevation": -5.0, "lens": 18.0}}


# ----------------------------------------------------------------------------- settings
def parse_args(cfg):
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    i = 0
    while i < len(argv):
        a, nxt = argv[i], (argv[i + 1] if i + 1 < len(argv) else None)
        if a in ("--dir", "--mode", "--background"):
            cfg[a[2:]], i = nxt, i + 1
        elif a in ("--seconds", "--elevation", "--zoom", "--point-scale", "--crop", "--lens"):
            cfg[a[2:].replace("-", "_")], i = float(nxt), i + 1
        elif a in ("--fps", "--samples"):
            cfg[a[2:]], i = int(nxt), i + 1
        elif a == "--resolution":
            w, h = nxt.lower().split("x")
            cfg["resolution"], i = (int(w), int(h)), i + 1
        elif a == "--transparent":
            cfg["transparent"] = True
        elif a == "--with-mesh":
            cfg["with_mesh"] = True
        elif a == "--no-render":
            cfg["render"] = False
        i += 1                      # unknown options (e.g. blender_import.py's) are skipped
    w, h = cfg["resolution"]
    cfg["resolution"] = (w - w % 2, h - h % 2)         # H.264 needs even sizes
    return cfg


def find_dir(cfg):
    """The reconstruction folder: --dir, else where this script / blender_import.py were opened from."""
    if cfg["dir"]:
        return os.path.abspath(cfg["dir"])
    cands = [os.path.dirname(bpy.path.abspath(t.filepath)) for t in bpy.data.texts
             if t.filepath and os.path.basename(t.filepath) in ("blender_turntable.py", "blender_import.py")]
    if "__file__" in globals():
        cands.append(os.path.dirname(os.path.abspath(__file__)))
    cands.append(bpy.path.abspath("//"))
    cands = [c for c in cands if c and os.path.isdir(c)]
    for c in cands:
        if os.path.exists(os.path.join(c, "transforms.json")):
            return c
    return cands[0] if cands else os.getcwd()


def color_rgb(name):
    named = {"white": (1.0, 1.0, 1.0), "black": (0.0, 0.0, 0.0), "grey": (0.5, 0.5, 0.5), "gray": (0.5, 0.5, 0.5)}
    if name.lower() in named:
        return named[name.lower()]
    h = name.lstrip("#")
    srgb = [int(h[k:k + 2], 16) / 255.0 for k in (0, 2, 4)]
    # world colours are linear: convert from the sRGB hex code
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb)


# ----------------------------------------------------------------------------- scene analysis
def ensure_imported(d):
    if "DA3_points" in bpy.data.objects:
        return
    script = os.path.join(d, "blender_import.py")
    if not os.path.exists(script):
        raise RuntimeError("DA3_points not found: run blender_import.py first (same folder as this script)")
    print("[turntable] reconstruction not imported yet: running", script)
    runpy.run_path(script, run_name="__main__")


def point_cloud_world(obj, max_points=200000):
    n = len(obj.data.vertices)
    co = np.empty(n * 3, np.float32)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3).astype(np.float64)
    if n > max_points:
        co = co[np.random.default_rng(0).choice(n, max_points, replace=False)]
    M = np.array(obj.matrix_world)
    return co @ M[:3, :3].T + M[:3, 3], n


def photo_cameras():
    cams = [o for o in bpy.data.objects if o.type == "CAMERA" and o.name.startswith("DA3_cam_")]
    pos = np.array([list(o.matrix_world.translation) for o in cams]) if cams else np.zeros((0, 3))
    fwd = np.array([list((o.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()) for o in cams]) \
        if cams else np.zeros((0, 3))
    up = np.array([list((o.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))).normalized()) for o in cams]) \
        if cams else np.zeros((0, 3))
    return pos, fwd, up


def angular_coverage(angles):
    """Angle (rad) covered by a set of directions = 2 pi - the largest gap between them."""
    if len(angles) < 2:
        return 0.0
    a = np.sort(np.mod(angles, 2 * np.pi))
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * np.pi]]))
    return float(2 * np.pi - gaps.max())


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def analyse(P, cam_pos, cam_fwd, cam_up):
    """Centre, size, up axis, scene type and the angles the photos cover."""
    center = np.median(P, axis=0)
    radius = max(float(np.percentile(np.linalg.norm(P - center, axis=1), 90)), 1e-6)
    up = unit(cam_up.mean(0)) if len(cam_up) and np.linalg.norm(cam_up.mean(0)) > 1e-6 else np.array([0.0, 0.0, 1.0])
    info = {"center": center, "radius": radius, "up": up, "n_cams": len(cam_pos)}
    if len(cam_pos) == 0:
        info.update(mode="orbit", why="no photo cameras found", start=unit(np.cross(up, [1.0, 0, 0]) + 1e-9))
        return info
    to_center = np.array([unit(center - p) for p in cam_pos])
    inward = float(np.mean(np.sum(cam_fwd * to_center, axis=1)))          # 1 = all look at the centre
    inside_ratio = float(np.median(np.linalg.norm(cam_pos - center, axis=1)) / radius)
    horiz = lambda v: v - np.outer(v @ up, up) if v.ndim == 2 else v - (v @ up) * up   # noqa: E731
    interior = inward < 0.3 or (inside_ratio < 0.5 and inward < 0.6)
    if interior:
        f = horiz(cam_fwd)
        start = unit(f.mean(0)) if np.linalg.norm(f.mean(0)) > 1e-6 else unit(horiz(np.array([0.0, 1.0, 0.0])))
        ex, ey = unit(np.cross(start, up)), start                     # camera looks along +ey at angle 0
        ang = np.arctan2(-(f @ ex), f @ ey)
        info.update(mode="inside", start=start, origin=cam_pos.mean(0), angles=ang, coverage=angular_coverage(ang))
        info["why"] = (f"photos taken from inside, looking outwards (inward {inward:.2f}, "
                       f"camera distance {inside_ratio:.2f} x size)")
    else:
        o = horiz(cam_pos - center)
        start = unit(o.mean(0)) if np.linalg.norm(o.mean(0)) > 1e-6 else unit(horiz(np.array([0.0, -1.0, 0.0])))
        ey = -start                                                    # camera sits at -ey at angle 0
        ex = unit(np.cross(ey, up))
        ang = np.arctan2(o @ ex, -(o @ ey))
        cov = angular_coverage(ang)
        info.update(start=start, angles=ang, coverage=cov)
        if cov >= math.radians(200):
            info.update(mode="orbit", why=f"photos all around the subject (cover {math.degrees(cov):.0f}°)")
        else:
            info.update(mode="arc", why=f"photos on one side (cover {math.degrees(cov):.0f}°): swing over it")
    info.update(inward=inward, inside_ratio=inside_ratio)
    return info


def swing_half_width(angles, lo_deg, hi_deg, margin_deg=20.0):
    a = np.angle(np.exp(1j * np.asarray(angles)))                      # wrap to (-pi, pi]
    half = (np.abs(a).max() if len(a) else 0.0) + math.radians(margin_deg)
    return float(np.clip(half, math.radians(lo_deg), math.radians(hi_deg)))


def median_spacing(P, n_total, probes=2000):
    """Typical distance between neighbouring points (KD-tree on a sample, corrected for the sampling:
    on a surface the spacing scales with 1 / sqrt(density))."""
    rng = np.random.default_rng(1)
    m = min(len(P), 100000)
    Q = P[rng.choice(len(P), m, replace=False)] if len(P) > m else P
    kd = KDTree(m)
    for i, p in enumerate(Q):
        kd.insert(p, i)
    kd.balance()
    probe = Q[rng.choice(m, min(m, probes), replace=False)]
    d = [kd.find_n(p, 2)[1][2] for p in probe] if m >= 2 else [0.0]
    return float(np.median(d)) * math.sqrt(m / max(n_total, 1))


# ----------------------------------------------------------------------------- rig
def remove_old_rig():
    for name in reversed(RIG):
        o = bpy.data.objects.get(name)
        if o is not None:
            data = o.data
            bpy.data.objects.remove(o, do_unlink=True)
            if data is not None and data.users == 0:
                bpy.data.cameras.remove(data)


def build_rig(scene, info, cfg):
    """base (fixed, local Z = up) -> spin (turns about Z, driven by the frame) -> camera.
    Returns (camera, camera start position, horizontal FOV, number of frames, motion description)."""
    remove_old_rig()
    mode = info["mode"]
    W, H = cfg["resolution"]
    lens = cfg["lens"] or MODE_DEFAULTS[mode]["lens"]
    elev = math.radians(cfg["elevation"] if cfg["elevation"] is not None else MODE_DEFAULTS[mode]["elevation"])
    cam_data = bpy.data.cameras.new(RIG[2])
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width, cam_data.lens = 36.0, lens
    hfov = 2 * math.atan(18.0 / lens)
    vfov = 2 * math.atan((36.0 * H / W) / 2 / lens)
    up = Vector(info["up"])
    start = Vector(info["start"])
    ez = up
    if mode == "inside":
        ey = start                                # look direction at angle 0
        origin = Vector(info["origin"])
        dist = 0.0
        pos = Vector((0.0, 0.0, 0.0))
        look = Vector((0.0, math.cos(elev), math.sin(elev)))
    else:
        ey = -start                               # the camera starts at -Y of the base = the photos' side
        origin = Vector(info["center"])
        dist = info["radius"] / math.sin(min(vfov, hfov) / 2) / max(cfg["zoom"], 1e-3)
        pos = Vector((0.0, -dist * math.cos(elev), dist * math.sin(elev)))
        look = -pos
    ex = ey.cross(ez).normalized()
    ey = ez.cross(ex).normalized()
    near = info["radius"] * 1e-3
    cam_data.clip_start, cam_data.clip_end = near, max(dist, info["radius"]) * 1e3

    base = bpy.data.objects.new(RIG[0], None)
    base.matrix_world = Matrix.Translation(origin) @ Matrix((ex, ey, ez)).transposed().to_4x4()
    spin = bpy.data.objects.new(RIG[1], None)
    spin.parent = base
    cam = bpy.data.objects.new(RIG[2], cam_data)
    cam.parent = spin
    cam.location = pos
    cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
    for o in (base, spin, cam):
        scene.collection.objects.link(o)
    base.empty_display_size = spin.empty_display_size = info["radius"] * 0.1

    frames = max(2, int(round(cfg["seconds"] * cfg["fps"])))
    full = mode == "orbit" or (mode == "inside" and info.get("coverage", 0.0) >= math.radians(250))
    fc = spin.driver_add("rotation_euler", 2)
    fc.driver.type = "SCRIPTED"
    if full:      # constant-speed full turn; frame `frames + 1` equals frame 1, so it loops seamlessly
        fc.driver.expression = f"(frame - 1) * {2 * math.pi / frames:.12f}"
        desc = "full 360° turn"
    else:         # smooth back-and-forth swing (starts in the middle, loops seamlessly)
        half = swing_half_width(info.get("angles", []), *((25, 75) if mode == "arc" else (30, 90)))
        fc.driver.expression = f"{half:.6f} * sin((frame - 1) * {2 * math.pi / frames:.12f})"
        desc = f"swing ±{math.degrees(half):.0f}°"
    scene.frame_start, scene.frame_end = 1, frames
    scene.frame_set(1)
    cam_start = Vector(info["origin"]) if mode == "inside" else (
        origin + start * dist * math.cos(elev) + up * dist * math.sin(elev))
    return cam, np.array(cam_start), hfov, frames, desc


# ----------------------------------------------------------------------------- look of the points
def points_tree(obj):
    for mod in obj.modifiers:
        if mod.type == "NODES" and mod.node_group is not None and mod.node_group.name.startswith("DA3_points"):
            return mod.node_group
    return None


def style_points(obj, radius, center_world, crop_radius):
    """True colours (emission), automatic point radius, and far points hidden (Geometry Nodes, the
    files on disk are not changed). Re-running the script updates the same nodes."""
    for mat in bpy.data.materials:
        if not (mat.name.startswith("DA3_point_color") and mat.use_nodes):
            continue
        nt = mat.node_tree
        attr = next((n for n in nt.nodes if n.type == "ATTRIBUTE"), None)
        out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
        if attr is None or out is None:
            continue
        em = nt.nodes.get("DA3_emission") or nt.nodes.new("ShaderNodeEmission")
        em.name = "DA3_emission"
        em.inputs["Strength"].default_value = 1.0
        nt.links.new(attr.outputs["Color"], em.inputs["Color"])
        nt.links.new(em.outputs["Emission"], out.inputs["Surface"])

    tree = points_tree(obj)
    if tree is None:
        print("[turntable] point-cloud node tree not found: point size and crop unchanged")
        return
    to_pts = next((n for n in tree.nodes if n.type == "MESH_TO_POINTS"), None)
    set_mat = next((n for n in tree.nodes if n.type == "SET_MATERIAL"), None)
    if to_pts is None or set_mat is None:
        return
    to_pts.inputs["Radius"].default_value = radius
    delete = tree.nodes.get("DA3_crop_delete")
    if crop_radius is None:
        if delete is not None:
            delete.mute = True
        return
    if delete is None:
        pos = tree.nodes.new("GeometryNodeInputPosition")
        pos.name = "DA3_crop_position"
        dist = tree.nodes.new("ShaderNodeVectorMath")
        dist.name, dist.operation = "DA3_crop_distance", "DISTANCE"
        cmp_ = tree.nodes.new("FunctionNodeCompare")
        cmp_.name = "DA3_crop_compare"
        cmp_.data_type, cmp_.operation = "FLOAT", "GREATER_THAN"
        delete = tree.nodes.new("GeometryNodeDeleteGeometry")
        delete.name, delete.domain = "DA3_crop_delete", "POINT"
        tree.links.new(pos.outputs["Position"], dist.inputs[0])
        tree.links.new(dist.outputs["Value"], cmp_.inputs[0])
        tree.links.new(cmp_.outputs["Result"], delete.inputs["Selection"])
        tree.links.new(to_pts.outputs["Points"], delete.inputs["Geometry"])
        tree.links.new(delete.outputs["Geometry"], set_mat.inputs["Geometry"])
    delete.mute = False
    center_local = obj.matrix_world.inverted() @ Vector(center_world)     # Geometry Nodes work in object space
    tree.nodes["DA3_crop_distance"].inputs[1].default_value = center_local
    tree.nodes["DA3_crop_compare"].inputs[1].default_value = crop_radius   # rotation only: same units


# ----------------------------------------------------------------------------- render settings
def setup_render(scene, cam, cfg, out_dir):
    scene.camera = cam
    r = scene.render
    r.resolution_x, r.resolution_y = cfg["resolution"]
    r.resolution_percentage = 100
    r.pixel_aspect_x = r.pixel_aspect_y = 1.0          # blender_import.py may set fx/fy here
    r.fps = cfg["fps"]
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            r.engine = engine
            break
        except TypeError:
            continue
    try:
        scene.eevee.taa_render_samples = cfg["samples"]
    except AttributeError:
        pass
    try:
        scene.view_settings.view_transform = "Standard"   # colours as in the photos (no Filmic/AgX tone curve)
        scene.view_settings.look = "None"
        scene.view_settings.exposure, scene.view_settings.gamma = 0.0, 1.0
    except (TypeError, AttributeError):
        pass

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = next((n for n in nt.nodes if n.type == "BACKGROUND"), None) or nt.nodes.new("ShaderNodeBackground")
    out = next((n for n in nt.nodes if n.type == "OUTPUT_WORLD"), None) or nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bg.inputs["Color"].default_value = (*color_rgb(cfg["background"]), 1.0)
    bg.inputs["Strength"].default_value = 1.0

    keep = ("DA3_points",) + (("DA3_mesh",) if cfg["with_mesh"] else ())
    for o in scene.objects:
        if o.type in ("MESH", "CURVE", "CURVES", "POINTCLOUD", "VOLUME", "SURFACE", "META", "FONT"):
            o.hide_render = not o.name.startswith(keep)

    s = r.image_settings
    video = not cfg["transparent"]
    if video:
        try:
            if hasattr(s, "media_type"):                    # newer Blender: image vs video output
                s.media_type = "VIDEO"
            s.file_format = "FFMPEG"
            r.ffmpeg.format = "MPEG4"
            r.ffmpeg.codec = "H264"                          # yuv420p: plays in PowerPoint / Slides / browsers
            r.ffmpeg.constant_rate_factor = "HIGH"
            r.ffmpeg.ffmpeg_preset = "GOOD"
            r.ffmpeg.gopsize = cfg["fps"]
            try:
                r.ffmpeg.audio_codec = "NONE"
            except (TypeError, AttributeError):
                pass
            r.film_transparent = False
            r.filepath = os.path.join(out_dir, "turntable")      # Blender adds the frame range + .mp4
        except (TypeError, AttributeError) as e:
            print(f"[turntable] video output not available here ({e!r}): writing PNG frames instead")
            video = False
    if not video:
        try:
            if hasattr(s, "media_type"):
                s.media_type = "IMAGE"
        except TypeError:
            pass
        s.file_format = "PNG"
        s.color_mode = "RGBA" if cfg["transparent"] else "RGB"
        r.film_transparent = bool(cfg["transparent"])
        r.filepath = os.path.join(out_dir, "turntable_frames", "frame_####")
    return video


# ----------------------------------------------------------------------------- main
def main():
    cfg = parse_args(dict(CONFIG))
    d = find_dir(cfg)
    ensure_imported(d)
    scene = bpy.context.scene
    pts = bpy.data.objects["DA3_points"]
    P, n_total = point_cloud_world(pts)
    info = analyse(P, *photo_cameras())
    if cfg["mode"] in MODE_DEFAULTS and cfg["mode"] != info["mode"]:
        if cfg["mode"] == "inside" and "origin" not in info:
            info["origin"] = info["center"]
        info["why"] = f"forced by CONFIG (auto would be {info['mode']}: {info['why']})"
        info["mode"] = cfg["mode"]
        if info["mode"] == "inside" and "coverage" not in info:
            info["coverage"] = 2 * math.pi
    cam, cam_start, hfov, frames, motion = build_rig(scene, info, cfg)

    crop = cfg["crop"] * info["radius"] if cfg["crop"] else None
    keep = P if crop is None else P[np.linalg.norm(P - info["center"], axis=1) <= crop]
    spacing = median_spacing(keep, n_total * len(keep) / max(len(P), 1))
    pixel = 2 * float(np.median(np.linalg.norm(keep - cam_start, axis=1))) * math.tan(hfov / 2) / cfg["resolution"][0]
    radius = max(0.75 * spacing, 0.6 * pixel) * cfg["point_scale"]
    style_points(pts, radius, info["center"], crop)
    video = setup_render(scene, cam, cfg, d)

    print(f"[turntable] mode: {info['mode']} ({info['why']}); motion: {motion}")
    print(f"[turntable] subject size {info['radius']:.3f}, point radius {radius:.5f} "
          f"(spacing {spacing:.5f}, 1 px = {pixel:.5f}), crop {'off' if crop is None else f'{crop:.3f}'}; "
          f"{frames} frames at {cfg['fps']} fps, {cfg['resolution'][0]}x{cfg['resolution'][1]}")
    if not cfg["render"]:
        print("[turntable] camera ready (render = False): View > Cameras > Active Camera, then Space to play")
        return
    print("[turntable] rendering ... (Blender looks frozen until it is done)")
    bpy.ops.render.render(animation=True)
    if video:
        found = sorted(glob.glob(os.path.join(d, "turntable[0-9]*.mp4")), key=os.path.getmtime)
        if found:
            final = os.path.join(d, f"turntable_{info['mode']}.mp4")
            os.replace(found[-1], final)
            print("[turntable] video:", final)
        else:
            print("[turntable] video written to", d)
    else:
        print("[turntable] frames:", os.path.join(d, "turntable_frames"))


main()
