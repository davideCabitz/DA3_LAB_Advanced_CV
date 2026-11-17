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


def fuse_point_cloud(points, colors, conf=None, conf_percentile: float = 40.0, max_points: int = 2_000_000, seed=0):
    """points (N,H,W,3), colors (N,H,W,3) uint8, conf (N,H,W). Keeps pixels whose confidence is above
    the given percentile (per scene), then randomly subsamples to max_points."""
    P = points.reshape(-1, 3)
    C = colors.reshape(-1, 3)
    keep = np.isfinite(P).all(1)
    if conf is not None and conf_percentile > 0:
        c = conf.reshape(-1)
        keep &= c >= np.percentile(c[keep], conf_percentile)
    P, C = P[keep], C[keep]
    if len(P) > max_points:
        idx = np.random.default_rng(seed).choice(len(P), max_points, replace=False)
        P, C = P[idx], C[idx]
    return P, C


def write_transforms_json(path, w2c, K, hw, image_names, extra: dict | None = None):
    """NeRF-style transforms.json with per-frame intrinsics."""
    H, W = hw
    w2c = _to44(w2c)
    frames = []
    for i, name in enumerate(image_names):
        c2w_cv = np.linalg.inv(w2c[i])
        k = np.asarray(K[i], np.float64)
        frames.append({
            "file_path": name,
            "transform_matrix": (c2w_cv @ CV_TO_GL).tolist(),  # OpenGL / NeRF convention
            "fl_x": k[0, 0], "fl_y": k[1, 1], "cx": k[0, 2], "cy": k[1, 2], "w": W, "h": H,
            "w2c_opencv": w2c[i].tolist(),
            "K": k.tolist(),
        })
    out = {"camera_model": "OPENCV", "w": W, "h": H, "world_frame": "OpenCV frame of camera 0 (x right, y down, z forward)",
           "frames": frames, **(extra or {})}
    Path(path).write_text(json.dumps(out, indent=2))


def write_colmap_text(out_dir, w2c, K, hw, image_names):
    """COLMAP text model (cameras.txt, images.txt, empty points3D.txt)."""
    from scipy.spatial.transform import Rotation

    H, W = hw
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w2c = _to44(w2c)
    cams, imgs = ["# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]"], ["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME", "#"]
    for i, name in enumerate(image_names):
        k = K[i]
        cams.append(f"{i + 1} PINHOLE {W} {H} {k[0, 0]:.6f} {k[1, 1]:.6f} {k[0, 2]:.6f} {k[1, 2]:.6f}")
        qx, qy, qz, qw = Rotation.from_matrix(w2c[i][:3, :3]).as_quat()
        t = w2c[i][:3, 3]
        imgs.append(f"{i + 1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {i + 1} {name}")
        imgs.append("")
    (out / "cameras.txt").write_text("\n".join(cams) + "\n")
    (out / "images.txt").write_text("\n".join(imgs) + "\n")
    (out / "points3D.txt").write_text("# empty\n")


def tsdf_mesh(depth, rgb, K, w2c, voxel: float | None = None, depth_trunc: float | None = None):
    """Open3D TSDF fusion of per-view depth maps -> (vertices, faces, colors uint8)."""
    import open3d as o3d

    d_valid = depth[np.isfinite(depth) & (depth > 0)]
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
    c = (np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8)
    return v, f, c


def depth_mesh(points, rgb, conf=None, conf_percentile=40.0, step: int = 2, max_edge_ratio: float = 0.05):
    """Fallback mesher: triangulate each view's pixel grid, dropping triangles across depth edges."""
    verts, faces, cols, off = [], [], [], 0
    thr = None if conf is None else np.percentile(conf, conf_percentile)
    for i in range(len(points)):
        P = points[i][::step, ::step]
        C = rgb[i][::step, ::step]
        h, w = P.shape[:2]
        ok = np.isfinite(P).all(-1)
        if thr is not None:
            ok &= conf[i][::step, ::step] >= thr
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
