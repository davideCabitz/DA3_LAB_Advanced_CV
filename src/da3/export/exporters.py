"""Exports for viewing reconstructions in Blender / other tools.

World frame = OpenCV frame of the reference (first) camera: x right, y down, z forward.
* PLY: binary little-endian, float xyz + uchar rgb.
* transforms.json: NeRF/instant-ngp style (camera-to-world in OpenGL camera axes, i.e. x right,
  y up, z backward), plus per-frame OpenCV matrices (`w2c_opencv`, `K`) so no information is lost.
* cameras.txt / images.txt: COLMAP text model (PINHOLE, world-to-camera quaternion + translation).
* GLB: triangle mesh with vertex colours (TSDF fusion with Open3D when available, otherwise a
  per-view depth-map triangulation).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def _to44(T):
    T = np.asarray(T, np.float64)
    if T.shape[-2] == 4:
        return T
    out = np.tile(np.eye(4), T.shape[:-2] + (1, 1))
    out[..., :3, :4] = T
    return out


def write_ply(path: str | Path, xyz: np.ndarray, rgb: np.ndarray | None = None):
    xyz = np.asarray(xyz, np.float32)
    n = len(xyz)
    if rgb is None:
        rgb = np.full((n, 3), 200, np.uint8)
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * (255 if rgb.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr = np.empty(n, dtype=dt)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


def depth_edge_mask(depth, rtol: float = 0.05) -> np.ndarray:
    """True at depth discontinuities ("flying pixels"): the depth range of the 3x3 neighbourhood is
    larger than `rtol` x the pixel's depth. At an object / background boundary the network outputs
    in-between depths, which back-project to points strung along the camera rays. depth: (N, H, W)
    or (H, W); invalid pixels (<= 0 or not finite) never count as neighbours and are not flagged."""
    from scipy.ndimage import maximum_filter, minimum_filter

    d = np.asarray(depth, np.float64)
    valid = np.isfinite(d) & (d > 0)
    size = (1, 3, 3) if d.ndim == 3 else (3, 3)
    hi = maximum_filter(np.where(valid, d, 0.0), size=size)
    lo = minimum_filter(np.where(valid, d, np.inf), size=size)
    return valid & ((hi - lo) > rtol * np.where(valid, d, 1.0))


def fuse_point_cloud(points, colors, conf=None, conf_percentile: float = 40.0, max_points: int = 2_000_000, seed=0,
                     mask=None):
    """points (N,H,W,3), colors (N,H,W,3) uint8, conf (N,H,W). Keeps pixels inside `mask` (e.g. not on
    a depth edge) whose confidence is above the given percentile (per scene), then randomly
    subsamples to max_points."""
    P = points.reshape(-1, 3)
    C = colors.reshape(-1, 3)
    keep = np.isfinite(P).all(1)
    if mask is not None:
        keep &= np.asarray(mask).reshape(-1)
    if conf is not None and conf_percentile > 0:
        c = conf.reshape(-1)
        keep &= c >= np.percentile(c[keep], conf_percentile)
    P, C = P[keep], C[keep]
    if len(P) > max_points:
        idx = np.random.default_rng(seed).choice(len(P), max_points, replace=False)
        P, C = P[idx], C[idx]
    return P, C


def _per_view_hw(hw, n: int) -> list:
    """hw = one (H, W) for all views, or a list of n per-view (H, W)."""
    if len(hw) == 2 and np.ndim(hw[0]) == 0:
        return [(int(hw[0]), int(hw[1]))] * n
    return [(int(h), int(w)) for h, w in hw]


def write_transforms_json(path, w2c, K, hw, image_names, extra: dict | None = None):
    """NeRF-style transforms.json with per-frame intrinsics (and per-frame image sizes when `hw` is a
    list of per-view (H, W); the top-level w / h are then those of the first view)."""
    hws = _per_view_hw(hw, len(image_names))
    H, W = hws[0]
    w2c = _to44(w2c)
    frames = []
    for i, name in enumerate(image_names):
        c2w_cv = np.linalg.inv(w2c[i])
        k = np.asarray(K[i], np.float64)
        frames.append({
            "file_path": name,
            "transform_matrix": (c2w_cv @ CV_TO_GL).tolist(),  # OpenGL / NeRF convention
            "fl_x": k[0, 0], "fl_y": k[1, 1], "cx": k[0, 2], "cy": k[1, 2], "w": hws[i][1], "h": hws[i][0],
            "w2c_opencv": w2c[i].tolist(),
            "K": k.tolist(),
        })
    out = {"camera_model": "OPENCV", "w": W, "h": H, "world_frame": "OpenCV frame of camera 0 (x right, y down, z forward)",
           "frames": frames, **(extra or {})}
    Path(path).write_text(json.dumps(out, indent=2))


def write_colmap_text(out_dir, w2c, K, hw, image_names):
    """COLMAP text model (cameras.txt, images.txt, empty points3D.txt); `hw` as in write_transforms_json."""
    from scipy.spatial.transform import Rotation

    hws = _per_view_hw(hw, len(image_names))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w2c = _to44(w2c)
    cams, imgs = ["# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]"], ["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME", "#"]
    for i, name in enumerate(image_names):
        k = K[i]
        H, W = hws[i]
        cams.append(f"{i + 1} PINHOLE {W} {H} {k[0, 0]:.6f} {k[1, 1]:.6f} {k[0, 2]:.6f} {k[1, 2]:.6f}")
        qx, qy, qz, qw = Rotation.from_matrix(w2c[i][:3, :3]).as_quat()
        t = w2c[i][:3, 3]
        imgs.append(f"{i + 1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {i + 1} {name}")
        imgs.append("")
    (out / "cameras.txt").write_text("\n".join(cams) + "\n")
    (out / "images.txt").write_text("\n".join(imgs) + "\n")
    (out / "points3D.txt").write_text("# empty\n")


def _valid_values(maps) -> np.ndarray:
    """Finite positive values of a stacked (N, H, W) array or of a list of maps of any sizes."""
    out = []
    for d in maps:
        d = np.asarray(d)
        out.append(d[np.isfinite(d) & (d > 0)].ravel())
    return np.concatenate(out)


def tsdf_mesh(depth, rgb, K, w2c, voxel: float | None = None, depth_trunc: float | None = None):
    """Open3D TSDF fusion of per-view depth maps -> (vertices, faces, colors uint8). depth / rgb: stacked
    arrays or lists of per-view maps (views may have different sizes; K in each view's pixels)."""
    import open3d as o3d

    d_valid = _valid_values(depth)
    med = float(np.median(d_valid))
    voxel = voxel or med / 256.0
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=4 * voxel, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    trunc = depth_trunc or float(np.percentile(d_valid, 98)) * 1.05
    w2c = _to44(w2c)
    for i in range(len(depth)):
        h, w = depth[i].shape
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb[i]).astype(np.uint8)),
            o3d.geometry.Image(np.nan_to_num(depth[i]).astype(np.float32)),
            depth_scale=1.0, depth_trunc=trunc, convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, K[i][0, 0], K[i][1, 1], K[i][0, 2], K[i][1, 2])
        vol.integrate(rgbd, intr, w2c[i])
    mesh = vol.extract_triangle_mesh()
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if len(v) == 0 or len(f) == 0:
        raise RuntimeError(f"Open3D {o3d.__version__} TSDF fusion returned an empty mesh "
                           "(see scripts/check_open3d.py)")
    c = (np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8)
    return v, f, c


def depth_mesh(points, rgb, conf=None, conf_percentile=40.0, step: int = 2, max_edge_ratio: float = 0.05,
               mask=None):
    """Fallback mesher: triangulate each view's pixel grid, dropping triangles across depth edges
    (and pixels outside `mask`). Stacked arrays or lists of per-view maps (any sizes)."""
    verts, faces, cols, off = [], [], [], 0
    thr = None
    if conf is not None:
        sel = [np.asarray(c).ravel() if mask is None else np.asarray(c)[np.asarray(m)].ravel()
               for c, m in zip(conf, mask if mask is not None else [None] * len(conf))]
        thr = np.percentile(np.concatenate(sel), conf_percentile)
    for i in range(len(points)):
        P = points[i][::step, ::step]
        C = rgb[i][::step, ::step]
        h, w = P.shape[:2]
        ok = np.isfinite(P).all(-1)
        if thr is not None:
            ok &= conf[i][::step, ::step] >= thr
        if mask is not None:
            ok &= np.asarray(mask[i])[::step, ::step]
        idx = np.arange(h * w).reshape(h, w)
        a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
        flat = P.reshape(-1, 3)
        dist = np.linalg.norm(flat, axis=1)
        for tri in (np.stack([a, c, b], -1), np.stack([b, c, d], -1)):
            tri = tri.reshape(-1, 3)
            good = ok.reshape(-1)[tri].all(1)
            e = np.max([np.linalg.norm(flat[tri[:, j]] - flat[tri[:, (j + 1) % 3]], axis=1) for j in range(3)], 0)
            good &= e < max_edge_ratio * dist[tri].mean(1) * step
            faces.append(tri[good] + off)
        verts.append(flat)
        cols.append(C.reshape(-1, 3))
        off += h * w
    return np.concatenate(verts), np.concatenate(faces), np.concatenate(cols).astype(np.uint8)


def write_glb(path, vertices, faces, colors):
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)
    mesh.remove_unreferenced_vertices()
    mesh.export(str(path), file_type="glb")
    return len(mesh.vertices), len(mesh.faces)
