"""Phase 5: feed-forward reconstruction from a folder of images, with Blender-ready exports.

  python scripts/infer.py --images path/to/folder --out outputs/demo/scene --weights official:da3-small
  python scripts/infer.py --images ... --poses poses.json     # optional known cameras (E_c conditioning)
  python scripts/infer.py --images ... --weights runs/<exp>/ckpt/last.pt  # our reduced-scale model
  python scripts/infer.py --images ... --per-image   # each image alone (N_v = 1): depth + ray map per image

poses.json: {"frames": [{"file": "img.jpg", "K": 3x3 (pixels of the original image),
             "c2w": 4x4 OpenCV camera-to-world}, ...]} (or "w2c" instead of "c2w").

Outputs in --out:
  depth/<name>.npy, depth/<name>.png (colour-mapped), conf/<name>.npy,
  ray/<name>.npy (h, w, 6) [origin, direction], ray/<name>.png (direction as RGB, like paper Fig. 2)
  cameras.json            K, w2c, c2w per view from the camera head D_C and from the ray map (Eq. 1-2)
  points.ply              fused coloured point cloud (confidence-filtered)
  transforms.json         NeRF-style cameras (+ OpenCV matrices); colmap/ text model
  mesh.glb                TSDF (Open3D) or depth-grid mesh, glTF axes (y up)
  preview.png             RGB / depth / confidence strip
  blender_import.py + <out>.zip   everything packaged for Blender
  blender_turntable.py    optional: renders a looping showcase video (turntable_<mode>.mp4) after the import;
                          orbit / arc / inside motion chosen from the photo cameras
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from da3.camera import affine_inverse, to_homogeneous
from da3.export import (
    depth_edge_mask,
    depth_mesh,
    fuse_point_cloud,
    tsdf_mesh,
    write_colmap_text,
    write_glb,
    write_ply,
    write_transforms_json,
)
from da3.geometry import points_from_depth_ray, unproject_depth
from da3.io_utils import list_images, load_views, scale_intrinsics, to_model_input
from da3.model import DA3

REPO = Path(__file__).resolve().parents[1]


def load_model(weights: str, device):
    from da3.checkpoint_io import official_model

    if weights.startswith("official:"):
        m, _ = official_model(weights.split(":", 1)[1])
        return m.to(device), "official"
    state = torch.load(weights, map_location="cpu", weights_only=False)
    m = DA3(state["config"]["model"])
    m.load_state_dict(state["model"])
    return m.eval().to(device), "reduced-scale"


def colorize(d: np.ndarray, high_is_red: bool = False) -> np.ndarray:
    """TURBO colour map over the 2-98% range. Depth uses high_is_red=False (near = red, far = blue);
    confidence uses high_is_red=True (confident = red, uncertain = blue)."""
    v = d[np.isfinite(d)]
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    x = np.clip((d - lo) / max(hi - lo, 1e-8), 0, 1)
    x = x if high_is_red else 1 - x
    return cv2.cvtColor(cv2.applyColorMap((255 * x).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def ray_to_rgb(ray: np.ndarray, hw) -> np.ndarray:
    """Ray-map visualization (cf. paper Fig. 2): unit ray direction mapped to RGB, (d/|d| + 1) / 2.
    ray (h, w, 6) = [origin, direction]; returns uint8 (H, W, 3). The origin channels are constant
    per view (the camera centre), so only the direction is shown."""
    d = ray[..., 3:6]
    d = d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-8)
    img = ((d + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="official:da3-small")
    ap.add_argument("--poses", default=None)
    ap.add_argument("--adapter", default=None, help="LoRA adapter (scripts/train_lora.py) on the released --weights")
    ap.add_argument("--per-image", action="store_true",
                    help="run every image on its own (N_v = 1): depth + ray map per image, one sub-folder each")
    ap.add_argument("--max-views", type=int, default=0)
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--cameras-from", choices=["head", "ray"], default="head",
                    help="which predicted cameras drive the point cloud / exports (both are saved)")
    ap.add_argument("--conf-percentile", type=float, default=40.0)
    ap.add_argument("--edge-rtol", type=float, default=0.05,
                    help="drop 'flying pixels' at depth edges (3x3 depth range > rtol x depth) from the "
                         "point cloud and the mesh; 0 = keep them")
    ap.add_argument("--mesh", choices=["tsdf", "depth", "none"], default="tsdf")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    paths = list_images(args.images)
    if not paths:
        raise SystemExit(f"no images found in {args.images}")
    if args.max_views:
        paths = paths[:: max(1, len(paths) // args.max_views)][: args.max_views]
    model, setting = load_model(args.weights, args.device)
    if args.adapter:
        from da3.lora import attach_adapter

        info = attach_adapter(model, args.weights, args.adapter)
        setting = "official+LoRA"
        print("adapter:", {k: info[k] for k in ("path", "arm", "seed", "step")})
    out = Path(args.out)
    if args.per_image:
        if args.poses:
            raise SystemExit("--poses is not supported with --per-image")
        for p in paths:
            reconstruct([p], out / p.stem, args, model, setting, package=False)
        shutil.copy(REPO / "scripts" / "blender_import.py", out / "blender_import.py")
        print("package:", shutil.make_archive(str(out), "zip", root_dir=out))
    else:
        reconstruct(paths, out, args, model, setting)


def reconstruct(paths, out: Path, args, model, setting, package: bool = True):
    for sub in ("depth", "conf", "ray"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    names = [p.name for p in paths]
    released = model.cfg.ray_head == "released"

    rgb, orig_sizes = load_views(paths, args.process_res)
    N, H, W = rgb.shape[:3]
    x = to_model_input(rgb).to(args.device)
    w2c_in = K_in = None
    if args.poses:
        meta = {f["file"]: f for f in json.loads(Path(args.poses).read_text())["frames"]}
        K_in = np.stack([scale_intrinsics(meta[n]["K"], sz, (H, W)) for n, sz in zip(names, orig_sizes)])
        w2c_in = np.stack([np.asarray(meta[n]["w2c"]) if "w2c" in meta[n] else np.linalg.inv(np.asarray(meta[n]["c2w"]))
                           for n in names])

    t0 = time.time()
    with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.float16, enabled=x.device.type == "cuda"):
        pred = model(x,
                     w2c=None if w2c_in is None else torch.from_numpy(w2c_in).float()[None].to(x.device),
                     K=None if K_in is None else torch.from_numpy(K_in).float()[None].to(x.device),
                     ref_strategy="saddle_balanced" if released else "first")
    dt = time.time() - t0
    w2c_head, K_head = DA3.cameras_from_head(pred["pose_enc"], (H, W))
    w2c_ray, K_ray = DA3.cameras_from_rays(pred, (H, W))
    depth = pred["depth"][0].float().cpu()
    conf = pred["depth_conf"][0].float().cpu().numpy()
    ray = pred["ray"][0].float().cpu()

    if w2c_in is not None:  # known poses: use them, with depth expressed in their metric scale
        from da3.eval.metrics import align_sim3_ransac

        _, _, s, _ = align_sim3_ransac(w2c_head[0].cpu().numpy(), to_homogeneous(torch.from_numpy(w2c_in)).numpy(),
                                       ransac=N >= 10)
        depth = depth / s
        w2c_use, K_use = torch.from_numpy(w2c_in[:, :3]).float(), torch.from_numpy(K_in).float()
        source = "input poses"
    elif args.cameras_from == "head":
        w2c_use, K_use, source = w2c_head[0].cpu(), K_head[0].cpu(), "camera head D_C"
    else:
        w2c_use, K_use, source = w2c_ray[0].cpu(), K_ray[0].cpu(), "ray map (Eq. 1-2)"
    c2w_use = affine_inverse(to_homogeneous(w2c_use))
    points = unproject_depth(depth, K_use, c2w_use).numpy()
    points_ray = points_from_depth_ray(depth, ray).numpy()  # P = t + D * d, for comparison

    # ---- per-view maps
    for i, n in enumerate(names):
        stem = Path(n).stem
        np.save(out / "depth" / f"{stem}.npy", depth[i].numpy())
        cv2.imwrite(str(out / "depth" / f"{stem}.png"), cv2.cvtColor(colorize(depth[i].numpy()), cv2.COLOR_RGB2BGR))
        np.save(out / "conf" / f"{stem}.npy", conf[i])
        np.save(out / "ray" / f"{stem}.npy", ray[i].numpy())
        cv2.imwrite(str(out / "ray" / f"{stem}.png"), cv2.cvtColor(ray_to_rgb(ray[i].numpy(), (H, W)), cv2.COLOR_RGB2BGR))
    cams = {
        "image_size_hw": [H, W], "setting": setting, "weights": args.weights, "cameras_used": source,
        "frames": [{"file": n,
                    "head": {"K": K_head[0, i].tolist(), "w2c": w2c_head[0, i].tolist()},
                    "ray": {"K": K_ray[0, i].tolist(), "w2c": w2c_ray[0, i].tolist()},
                    "used": {"K": K_use[i].tolist(), "w2c": w2c_use[i].tolist(), "c2w": c2w_use[i].tolist()}}
                   for i, n in enumerate(names)],
    }
    (out / "cameras.json").write_text(json.dumps(cams, indent=2))

    # ---- point cloud, cameras, mesh
    # pixels kept for the 3D outputs: not on a depth edge (flying pixels) and confident enough
    edge = depth_edge_mask(depth.numpy(), args.edge_rtol) if args.edge_rtol > 0 else np.zeros(conf.shape, bool)
    not_edge = ~edge
    P, C = fuse_point_cloud(points, rgb, conf, args.conf_percentile, mask=not_edge)
    write_ply(out / "points.ply", P, C)
    write_transforms_json(out / "transforms.json", w2c_use.numpy(), K_use.numpy(), (H, W),
                          [f"images/{n}" for n in names], {"cameras_used": source})
    write_colmap_text(out / "colmap", w2c_use.numpy(), K_use.numpy(), (H, W), names)
    (out / "images").mkdir(exist_ok=True)
    for n, im in zip(names, rgb):
        cv2.imwrite(str(out / "images" / n), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    mesh_info, mesher = None, "none"
    if args.mesh != "none":
        v = None
        if args.mesh == "tsdf":
            try:
                # same pixels as the point cloud: depth 0 = not integrated (edges, low confidence)
                good = not_edge & (conf >= np.percentile(conf[not_edge], args.conf_percentile))
                v, f, c = tsdf_mesh(np.where(good, depth.numpy(), 0.0), rgb, K_use.numpy(), w2c_use.numpy())
                mesher = "tsdf"
            except (ImportError, RuntimeError) as e:  # no Open3D, or a broken TSDF in this Open3D build
                print(f"[infer] TSDF meshing unavailable ({e}); using depth-grid meshing instead")
        if v is None:
            v, f, c = depth_mesh(points, rgb, conf, args.conf_percentile, mask=not_edge)
            mesher = "depth-grid"
        v = v * np.array([1.0, -1.0, -1.0])  # OpenCV world -> glTF (y up, z towards the viewer)
        mesh_info = write_glb(out / "mesh.glb", v, f, c)

    # ---- preview strip: RGB | depth (red = near) | ray map (direction) | depth confidence (red = high)
    k = min(N, 6)
    rows = [np.concatenate([rgb[i], colorize(depth[i].numpy()), ray_to_rgb(ray[i].numpy(), (H, W)),
                            colorize(np.log(conf[i]), high_is_red=True)], 1) for i in range(k)]
    cv2.imwrite(str(out / "preview.png"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))

    # ---- consistency diagnostics (depth + ray vs depth + camera)
    rel = np.linalg.norm(points - points_ray, axis=-1) / (np.linalg.norm(points, axis=-1) + 1e-8)
    rot = [float(np.degrees(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1))))
           for a, b in zip(w2c_head[0].cpu().numpy(), w2c_ray[0].cpu().numpy())]
    summary = {"views": N, "size_hw": [H, W], "forward_s": dt, "device": args.device, "setting": setting,
               "weights": args.weights, "points": int(len(P)), "mesh_vertices_faces": mesh_info, "mesher": mesher,
               "edge_rtol": args.edge_rtol, "edge_pixels_removed_pct": float(100 * edge.mean()),
               "cameras_used": source, "median_rel_diff_points_head_vs_ray": float(np.median(rel)),
               "max_rot_diff_head_vs_ray_deg": max(rot)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    shutil.copy(REPO / "scripts" / "blender_import.py", out / "blender_import.py")
    shutil.copy(REPO / "scripts" / "blender_turntable.py", out / "blender_turntable.py")
    if package:
        print("package:", shutil.make_archive(str(out), "zip", root_dir=out))


if __name__ == "__main__":
    main()
