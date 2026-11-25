"""KITTI clip evaluation of DA3 arms: the E3 go/no-go now, the held-out evaluation (H1, S1-S3) later.

Arms (all on identical clips, crops, resolution and reference view "first"):
  released         released DA3, unposed (the reference arm)
  posed            released DA3 conditioned on the OXTS poses through E_c (E3's teacher)
  lora             released DA3 + an adapter (--adapter), unposed

For every clip of `views` frames (every `stride`-th frame, `clips_per_drive` evenly spaced,
non-overlapping windows), images are centre-cropped to 1.8:1 and resized to 504x280; the predicted
depth is resized back to the crop and compared with the LiDAR depth there (src/da3/eval/clip_metrics.py).
Cameras come from D_C (the API default) for every arm. In the posed arm the D_C output is compared
with the input poses as well (does E_c copy its input?).

  python scripts/kitti_eval.py --root /tmp/canonical/kitti --drives val --arms released posed \
      --out /kaggle/working/kitti/gonogo_da3-small.json

`--drives val|test|train|all|<drive ids>` reads split.json under --root. The go/no-go uses val
drives (never the test drives). Gate (docs/da3_enhancement_report.md §4.4): GO if the posed arm's
clip-scale AbsRel is >= 5% lower than the released arm's (ratio of drive means <= 0.95).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from da3.camera import affine_inverse
from da3.data.canonical import CanonicalScene
from da3.data.kitti import center_crop_box, crop_intrinsics
from da3.eval.clip_metrics import clip_depth_metrics, trajectory_metrics
from da3.io_utils import resize_rgb, to_model_input
from da3.model import DA3

REPO = Path(__file__).resolve().parents[1]
GATE_RATIO = 0.95


def select_drives(root: Path, spec: list[str]) -> list[str]:
    split = json.loads((root / "split.json").read_text()) if (root / "split.json").exists() else {}
    if spec == ["all"]:
        return sorted(p.name for p in root.iterdir() if (p / "cameras.json").exists())
    if len(spec) == 1 and spec[0] in ("train", "val", "test"):
        if not split.get(spec[0]):
            raise SystemExit(f"split.json has no {spec[0]} drives")
        return split[spec[0]]
    return spec


def clip_starts(n_frames: int, views: int, stride: int, n_clips: int) -> list[int]:
    span = (views - 1) * stride + 1
    if n_frames < span:
        return []
    k = min(n_clips, n_frames // span)            # non-overlapping windows
    return [int(round(x)) for x in np.linspace(0, n_frames - span, k)]


def load_clip(scene: CanonicalScene, idx: list[int], hw=(280, 504), aspect=1.8):
    rgbs, gts, Ks, w2cs = [], [], [], []
    for i in idx:
        rgb, depth = scene.load(i)
        y0, x0, ch, cw = box = center_crop_box(*rgb.shape[:2], aspect)
        rgbs.append(resize_rgb(rgb[y0:y0 + ch, x0:x0 + cw], hw))
        gts.append(depth[y0:y0 + ch, x0:x0 + cw])
        Ks.append(crop_intrinsics(scene.K[i], box, hw))
        w2cs.append(np.linalg.inv(scene.c2w[i]))
    return np.stack(rgbs), np.stack(gts), np.stack(Ks), np.stack(w2cs)


@torch.no_grad()
def run_arm(model, x, w2c=None, K=None, amp=True):
    dev = next(model.parameters()).device
    kw = {}
    if w2c is not None:
        kw = dict(w2c=torch.from_numpy(w2c).float()[None].to(dev), K=torch.from_numpy(K).float()[None].to(dev))
    with torch.autocast(dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda"):
        out = model(x.to(dev), ref_strategy="first", **kw)
    pw2c, _ = DA3.cameras_from_head(out["pose_enc"], x.shape[-2:])
    return out["depth"][0].float().cpu().numpy(), pw2c[0].float().cpu().numpy()


def evaluate(args) -> dict:
    from da3.checkpoint_io import download_official, official_model
    from da3.lora import file_sha256, load_adapter, set_lora_enabled

    device = torch.device(args.device)
    model, _ = official_model(args.preset)
    model = model.to(device).eval()
    base_sha = file_sha256(download_official(args.preset))
    adapter_info = None
    if args.adapter:
        adapter_info = load_adapter(model, args.adapter, base_sha256=base_sha, base_preset=args.preset)
        set_lora_enabled(model, False)
    root = Path(args.root)
    drives = select_drives(root, args.drives)
    rows, t0 = [], time.time()
    for drive in drives:
        scene = CanonicalScene(root / drive)
        for c, start in enumerate(clip_starts(len(scene), args.views, args.stride, args.clips_per_drive)):
            idx = list(range(start, start + args.views * args.stride, args.stride))
            rgb, gt, K, w2c_gt = load_clip(scene, idx)
            x = to_model_input(rgb)
            # poses relative to view 0 (the gauge of every arm; E_c normalises the scale itself)
            w2c_rel = (w2c_gt @ np.linalg.inv(w2c_gt[0])).astype(np.float32)
            for arm in args.arms:
                if arm == "lora":
                    set_lora_enabled(model, True)
                depth, pw2c = run_arm(model, x, w2c_rel if arm == "posed" else None,
                                      K.astype(np.float32) if arm == "posed" else None, amp=not args.fp32)
                if arm == "lora":
                    set_lora_enabled(model, False)
                up = np.stack([cv2.resize(d, (gt.shape[2], gt.shape[1]), interpolation=cv2.INTER_LINEAR)
                               for d in depth])
                row = {"drive": drive, "clip": c, "start": start, "arm": arm,
                       **clip_depth_metrics(up, gt, max_depth=args.max_depth),
                       **trajectory_metrics(pw2c, w2c_rel)}
                rows.append(row)
            print(f"{drive} clip {c}: " + " | ".join(
                f"{r['arm']} AbsRel {r['absrel_clip']:.4f} ATE {r['ate']:.3f}" for r in rows[-len(args.arms):]),
                flush=True)
    return {"rows": rows, "drives": drives, "time_s": time.time() - t0, "adapter": adapter_info,
            "base_sha256": base_sha}


def summarise(rows: list[dict], arms: list[str]) -> dict:
    """Drive means (the unit), then the ratio of drive means vs the released arm, with a drive bootstrap
    CI when there are >= 3 drives (descriptive: the go/no-go is not a confirmatory test)."""
    from da3.enhancements.foveated.stats import ratio_of_means

    metrics = ["absrel_clip", "delta1_clip", "absrel_image", "delta1_image", "ate", "rpe_trans", "rpe_rot",
               "auc03", "auc30"]
    drives = sorted({r["drive"] for r in rows})
    per_drive = {a: {d: {m: float(np.nanmean([r[m] for r in rows if r["arm"] == a and r["drive"] == d]))
                         for m in metrics} for d in drives} for a in arms}
    out = {"per_drive": per_drive, "means": {a: {m: float(np.mean([per_drive[a][d][m] for d in drives]))
                                                 for m in metrics} for a in arms}, "vs_released": {}}
    if "released" in arms:
        for a in arms:
            if a == "released":
                continue
            out["vs_released"][a] = {}
            for m in metrics:
                test = [per_drive[a][d][m] for d in drives]
                ref = [per_drive["released"][d][m] for d in drives]
                est, lo, hi = (ratio_of_means(test, ref, n=10000, seed=0) if len(drives) >= 3
                               else (float(np.mean(test) / np.mean(ref)), float("nan"), float("nan")))
                out["vs_released"][a][m] = {"ratio": est, "ci_lo": lo, "ci_hi": hi}
    if "posed" in out["vs_released"]:
        r = out["vs_released"]["posed"]["absrel_clip"]["ratio"]
        out["gate"] = {"rule": f"posed / released clip-scale AbsRel <= {GATE_RATIO}", "ratio": r,
                       "decision": "GO" if r <= GATE_RATIO else "NO-GO",
                       "pose_headroom_note": "if only ATE/RPE improve with poses, E3 reduces to pose-supervised "
                                             "fine-tuning (report §4.4)"}
    return out


def markdown(summary: dict, arms: list[str], args) -> str:
    md = [f"# KITTI clip evaluation ({args.preset}, drives: {' '.join(args.drives)})\n",
          f"{args.views} views, stride {args.stride}, up to {args.clips_per_drive} clips per drive, 504x280 "
          f"(1.8:1 centre crop), cameras from D_C, LiDAR <= {args.max_depth} m. Unit = drive.\n",
          "| Arm | AbsRel (clip) | δ1 (clip) | AbsRel (image) | ATE (m) | RPE-t (m) | RPE-r (°) | AUC@3 | AUC@30 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        m = summary["means"][a]
        md.append(f"| {a} | {m['absrel_clip']:.4f} | {m['delta1_clip']:.3f} | {m['absrel_image']:.4f} | {m['ate']:.3f} "
                  f"| {m['rpe_trans']:.3f} | {m['rpe_rot']:.3f} | {100 * m['auc03']:.1f} | {100 * m['auc30']:.1f} |")
    for a, d in summary["vs_released"].items():
        md.append(f"\n**{a} / released** (ratio of drive means, 95% drive-bootstrap CI if ≥ 3 drives): " + ", ".join(
            f"{m} {v['ratio']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]" for m, v in d.items()))
    if "gate" in summary:
        g = summary["gate"]
        md.append(f"\n**Go/no-go:** {g['decision']} ({g['rule']}; observed {g['ratio']:.3f}). {g['pose_headroom_note']}.")
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="canonical KITTI root (scripts/prepare_kitti.py)")
    ap.add_argument("--drives", nargs="+", default=["val"])
    ap.add_argument("--arms", nargs="+", default=["released", "posed"], choices=["released", "posed", "lora"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--preset", default="da3-small", choices=["da3-small", "da3-base"])
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--clips-per-drive", type=int, default=10)
    ap.add_argument("--max-depth", type=float, default=80.0)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if "lora" in args.arms and not args.adapter:
        ap.error("--arms lora needs --adapter")

    from da3.train_utils import environment_info, git_commit

    res = evaluate(args)
    summary = summarise(res["rows"], args.arms)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"args": vars(args), "commit": git_commit(REPO), "env": environment_info(), **res, "summary": summary}
    out.write_text(json.dumps(payload, indent=2, default=float))
    md = markdown(summary, args.arms, args)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print("wrote", out, "and", out.with_suffix(".md"))


if __name__ == "__main__":
    main()
