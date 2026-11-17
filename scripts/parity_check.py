"""Phase 2: numerical parity between our DA3 modules and the official implementation.

Both networks load the same released safetensors and receive the same preprocessed tensors (fp32).
Compared outputs: depth, depth confidence, ray map, ray confidence, camera-head pose
(extrinsics / intrinsics) in three modes (monocular, multi-view unposed, multi-view posed), plus
ray-map -> pose recovery (official confidence-weighted RANSAC + QL routine vs. our plain
Eq. 1 + Eq. 2 DLT + RQ). Ray maps are compared in the paper's [origin, direction] order.

The official package is only imported here, from --official-src (a clone of
github.com/ByteDance-Seed/Depth-Anything-3); src/da3 never imports it.

Example:
  python scripts/parity_check.py --preset da3-small --official-src /path/Depth-Anything-3/src \
      --images /path/Depth-Anything-3/assets/examples/SOH --out experiments/parity/da3-small.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from da3.camera import affine_inverse, to_homogeneous
from da3.checkpoint_io import HF_REPOS, download_official, load_safetensors, official_model
from da3.geometry import pose_from_ray_map
from da3.io_utils import list_images, load_views, to_model_input
from da3.model import DA3


def build_official(official_src: str, preset: str, weights: Path):
    sys.path.insert(0, official_src)
    from depth_anything_3.cfg import create_object, load_config  # noqa: E402

    cfg_path = Path(official_src) / "depth_anything_3" / "configs" / f"{preset}.yaml"
    net = create_object(load_config(str(cfg_path)))
    sd = {k.removeprefix("model."): v for k, v in load_safetensors(weights).items()}
    res = net.load_state_dict(sd, strict=False)  # the released loader is non-strict too
    return net.eval(), {"missing": res.missing_keys, "unexpected": res.unexpected_keys}


def official_raw(net, x, w2c=None, K=None, ref="saddle_balanced"):
    """Official forward, keeping the ray map (the public forward deletes it)."""
    H, W = x.shape[-2:]
    cam_token = net.cam_enc(w2c, K, (H, W)) if w2c is not None else None
    feats, _ = net.backbone(x, cam_token=cam_token, export_feat_layers=[], ref_view_strategy=ref)
    out = dict(net.head(feats, H, W, patch_start_idx=0))
    out["ray_raw"] = out["ray"]  # released order [direction, origin]
    out["ray"] = torch.cat([out["ray"][..., 3:6], out["ray"][..., 0:3]], dim=-1)  # -> [origin, direction]
    full = net(x, extrinsics=w2c, intrinsics=K, ref_view_strategy=ref)
    out["w2c"], out["K"] = full["extrinsics"], full["intrinsics"]
    return out


def official_normalize_extrinsics(ex):
    """Copy of DepthAnything3._normalize_extrinsics (official API)."""
    transform = affine_inverse(ex[:, :1])
    ex_n = ex @ transform
    dists = affine_inverse(ex_n)[..., :3, 3].norm(dim=-1)
    ex_n[..., :3, 3] = ex_n[..., :3, 3] / torch.clamp(torch.median(dists), min=1e-1)
    return ex_n


def maxdiff(a, b):
    return float((a.float() - b.float()).abs().max())


def reldiff(a, b):
    return float(((a.float() - b.float()).abs() / b.float().abs().clamp_min(1e-6)).max())


def compare(ours: dict, off: dict) -> dict:
    hw = ours["depth"].shape[-2:]
    w2c, K = DA3.cameras_from_head(ours["pose_enc"], hw)
    return {
        "depth_max_abs": maxdiff(ours["depth"], off["depth"]),
        "depth_max_rel": reldiff(ours["depth"], off["depth"]),
        "depth_conf_max_abs": maxdiff(ours["depth_conf"], off["depth_conf"]),
        "ray_max_abs": maxdiff(ours["ray"], off["ray"]),
        "ray_conf_max_abs": maxdiff(ours["ray_conf"], off["ray_conf"]),
        "pose_w2c_max_abs": maxdiff(w2c, off["w2c"][..., :3, :]),
        "pose_K_max_abs": maxdiff(K, off["K"]),
        "depth_range": [float(off["depth"].min()), float(off["depth"].max())],
    }


def rot_deg(Ra, Rb):
    c = ((Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.rad2deg(torch.arccos(c.clamp(-1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="da3-small", choices=list(HF_REPOS))
    ap.add_argument("--official-src", required=True)
    ap.add_argument("--images", required=True, help="folder with >= 2 images")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    weights = Path(args.weights) if args.weights else download_official(args.preset)
    ours, load_info = official_model(args.preset, weights)
    off, off_info = build_official(args.official_src, args.preset, weights)
    ours, off = ours.to(args.device), off.to(args.device)

    paths = list_images(args.images)
    rgb, _ = load_views(paths, args.process_res)
    if len(rgb) < args.views:  # pad with horizontally flipped copies (parity only needs inputs)
        rgb = np.concatenate([rgb, rgb[:, :, ::-1]], 0)
    rgb = np.ascontiguousarray(rgb[: args.views])
    x = to_model_input(rgb).to(args.device)
    report = {
        "preset": args.preset,
        "setting": "official weights, ours vs official code, fp32",
        "device": args.device,
        "torch": torch.__version__,
        "input_shape": list(x.shape),
        "our_load": {k: (v if k != "dropped_unused" else len(v)) for k, v in load_info.items()},
        "official_missing_keys": off_info["missing"],
        "official_unexpected_keys": off_info["unexpected"],
        "modes": {},
    }

    with torch.no_grad():
        # 1) monocular, 2) multi-view unposed (reference-view selection active)
        for name, xi in [("mono", x[:, :1]), ("multiview_unposed", x)]:
            t0 = time.time()
            a = ours(xi, ref_strategy="saddle_balanced")
            b = official_raw(off, xi)
            report["modes"][name] = compare(a, b) | {"ref_index_ours": a["ref_index"].tolist() if a["ref_index"] is not None else None}
            print(name, f"{time.time() - t0:.1f}s", json.dumps(report["modes"][name]))

        # 3) multi-view posed: use the predicted cameras of the unposed run as "known" poses
        w2c, K = DA3.cameras_from_head(ours(x, ref_strategy="saddle_balanced")["pose_enc"], x.shape[-2:])
        w2c = to_homogeneous(w2c)
        a = ours(x, w2c=w2c, K=K)
        b = official_raw(off, x, w2c=official_normalize_extrinsics(w2c.clone()), K=K)
        report["modes"]["multiview_posed"] = compare(a, b)
        print("multiview_posed", json.dumps(report["modes"]["multiview_posed"]))

        # 4) ray map -> pose: official routine vs ours, on the same (official) ray output
        sys.path.insert(0, args.official_src)
        from depth_anything_3.utils.ray_utils import get_extrinsic_from_camray

        b = official_raw(off, x)
        H, W = x.shape[-2:]
        torch.manual_seed(0)
        ext, focal, pp = get_extrinsic_from_camray(b["ray_raw"], b["ray_conf"], b["ray"].shape[-3], b["ray"].shape[-2])
        off_w2c = affine_inverse(ext)  # official returns c2w-like [R|T]; the model inverts it
        off_fx = focal[..., 0] / 2 * W
        ours_w2c, ours_K = pose_from_ray_map(b["ray"], (H, W))  # plain Eq. 1 + Eq. 2, no weights
        head_w2c = b["w2c"][..., :3, :]
        report["ray_pose"] = {
            "rot_deg_ours_vs_official_ray": rot_deg(ours_w2c[..., :3, :3], off_w2c[..., :3, :3]).tolist(),
            "center_diff_ours_vs_official_ray": (ours_w2c[..., :3, 3] - off_w2c[..., :3, 3]).norm(dim=-1).tolist(),
            "fx_ours": ours_K[..., 0, 0].tolist(),
            "fx_official_ray": off_fx.tolist(),
            "fx_official_cam_head": b["K"][..., 0, 0].tolist(),
            "rot_deg_ours_ray_vs_cam_head": rot_deg(ours_w2c[..., :3, :3], head_w2c[..., :3, :3]).tolist(),
        }
        print("ray_pose", json.dumps(report["ray_pose"]))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
