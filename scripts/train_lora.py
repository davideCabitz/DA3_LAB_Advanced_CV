"""Train one LoRA arm on KITTI (enhancement #3; recipe frozen in configs/lora_analysis.yaml, docs/LORA.md §3.4).

  python scripts/train_lora.py --root /tmp/canonical/kitti --arm e3 --seed 0 \
      --cache /tmp/lora_cache/da3-small --out /kaggle/working/lora/runs

1. Cache (once per model, shared by every arm and seed; resumable, one file per clip): for each
   training clip, the released DA3 conditioned on the known poses (E3's teacher) and the released DA3
   unposed (anchor / own-output targets). Training drives only (split.json "train"); validation and
   test drives are never read here.
2. Train: LoRA on blocks 4-11 (everything else frozen), unposed student with reference view 0, arm
   targets + anchor (src/da3/lora_train.py), AdamW on A/B only, warm-up + cosine, fp16 autocast +
   GradScaler, backbone gradient checkpointing. Clip order is a function of (seed, step), so --resume
   continues the exact stream.
3. Outputs in <out>/<preset>_<arm>_s<seed>/: adapter.pt (adapter only, tied to the base weights'
   SHA-256), snapshots/step_XXXXX.pt every `--snapshot-every` steps (step 0 = B = 0, i.e. the released
   model), ckpt/last.pt (resume), log.jsonl, train_summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from da3.data.canonical import CanonicalScene
from da3.data.kitti import CLIP_HW, clip_starts, load_clip
from da3.io_utils import to_model_input
from da3.lora import (
    LoRAConfig,
    file_sha256,
    inject_lora,
    lora_parameters,
    lora_state_dict,
    save_adapter,
)
from da3.lora_train import ARMS, align_targets_scale, anchor_loss, arm_loss_config, clip_targets, to_device
from da3.losses import DA3Loss
from da3.model import DA3
from da3.train_utils import environment_info, git_commit, lr_lambda, rng_state, save_checkpoint, set_rng_state

REPO = Path(__file__).resolve().parents[1]
CACHE_VERSION = 1


# ----------------------------------------------------------------------------- cache
def training_clips(root: Path, views: int, stride: int, per_drive: int, max_clips: int) -> list[tuple[str, int]]:
    split = json.loads((root / "split.json").read_text())
    if not split.get("train"):
        raise SystemExit("split.json has no training drives: add more (non-Eigen-test) KITTI drives")
    clips = []
    for d in split["train"]:
        n = len(json.loads((root / d / "cameras.json").read_text())["frames"])
        clips += [(d, s) for s in clip_starts(n, views, stride, per_drive)]
    if len(clips) > max_clips:   # deterministic, evenly spread subset
        clips = [clips[int(i)] for i in np.linspace(0, len(clips) - 1, max_clips)]
    return clips


@torch.no_grad()
def build_cache(preset: str, root: Path, clips, cache_dir: Path, views: int, stride: int, device) -> None:
    from da3.checkpoint_io import official_model

    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [(i, c) for i, c in enumerate(clips) if not (cache_dir / f"clip_{i:05d}.npz").exists()]
    if not todo:
        print(f"cache complete: {len(clips)} clips in {cache_dir}", flush=True)
        return
    model, _ = official_model(preset)
    model = model.to(device).eval()
    amp = device.type == "cuda"
    scenes, t0 = {}, time.time()
    for k, (i, (drive, start)) in enumerate(todo):
        scene = scenes.setdefault(drive, CanonicalScene(root / drive))
        idx = list(range(start, start + views * stride, stride))
        rgb, _, K, w2c = load_clip(scene, idx)
        x = to_model_input(rgb).to(device)
        H, W = rgb.shape[1:3]
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            t = model(x, w2c=torch.from_numpy(w2c).float()[None].to(device),
                      K=torch.from_numpy(K).float()[None].to(device), ref_strategy="first")
            f = model(x, ref_strategy="first")
        tw2c, _ = DA3.cameras_from_head(t["pose_enc"], (H, W))
        fw2c, fK = DA3.cameras_from_head(f["pose_enc"], (H, W))
        np_ = lambda v: v[0].float().cpu().numpy()
        rec = {"rgb": rgb, "K": K.astype(np.float32), "w2c": w2c.astype(np.float32),
               "teacher_depth": np_(t["depth"]).astype(np.float16), "teacher_conf": np_(t["depth_conf"]).astype(np.float16),
               "teacher_head_w2c": np_(tw2c), "frozen_depth": np_(f["depth"]), "frozen_conf": np_(f["depth_conf"]).astype(np.float16),
               "frozen_ray": np_(f["ray"]).astype(np.float16), "frozen_pose_enc": np_(f["pose_enc"]),
               "frozen_head_w2c": np_(fw2c), "frozen_head_K": np_(fK),
               "meta": np.array(json.dumps({"drive": drive, "start": start, "views": views, "stride": stride,
                                            "preset": preset, "version": CACHE_VERSION}))}
        tmp = cache_dir / f"clip_{i:05d}.tmp.npz"
        np.savez(tmp, **rec)
        tmp.replace(cache_dir / f"clip_{i:05d}.npz")
        if (k + 1) % 25 == 0 or k + 1 == len(todo):
            print(f"cache {k + 1}/{len(todo)} clips ({time.time() - t0:.0f} s)", flush=True)
    del model
    torch.cuda.empty_cache()


def load_cached(cache_dir: Path, i: int) -> dict:
    with np.load(cache_dir / f"clip_{i:05d}.npz") as z:
        return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="canonical KITTI root with split.json (scripts/prepare_kitti.py)")
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--preset", default="da3-small", choices=["da3-small", "da3-base"])
    ap.add_argument("--plan", default=str(REPO / "configs" / "lora_analysis.yaml"))
    ap.add_argument("--cache", required=True, help="teacher / frozen-output cache folder (shared by arms and seeds)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips-per-drive", type=int, default=40)
    ap.add_argument("--max-clips", type=int, default=600)
    ap.add_argument("--snapshot-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--steps", type=int, default=None, help="override the plan's steps (smoke tests only; logged)")
    ap.add_argument("--cache-only", action="store_true", help="build the cache and stop")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--time-budget-h", type=float, default=0.0, help="stop cleanly after this many hours (resume later)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    t_start = time.time()

    plan_text = Path(args.plan).read_text(encoding="utf-8")
    plan = yaml.safe_load(plan_text)
    tc, lcfg = plan["train"], LoRAConfig.from_dict(plan["lora"])
    steps = int(args.steps or tc["steps"])
    views, stride = int(tc["views"]), int(plan["kitti"]["clip_stride_frames"])
    if list(tc["resolution"]) != list(CLIP_HW):
        raise SystemExit(f"plan resolution {tc['resolution']} != clip resolution {CLIP_HW}")
    device = torch.device(args.device)
    root, cache_dir = Path(args.root), Path(args.cache)

    clips = training_clips(root, views, stride, args.clips_per_drive, args.max_clips)
    print(f"{len(clips)} training clips from {len({d for d, _ in clips})} drives", flush=True)
    build_cache(args.preset, root, clips, cache_dir, views, stride, device)
    if args.cache_only:
        return

    from da3.checkpoint_io import download_official, official_model

    run = Path(args.out) / f"{args.preset}_{args.arm}_s{args.seed}"
    if (run / "train_summary.json").exists() and (run / "adapter.pt").exists():
        print(f"{run.name}: already trained ({run / 'adapter.pt'}), skipped", flush=True)
        return
    (run / "snapshots").mkdir(parents=True, exist_ok=True)
    base_sha = file_sha256(download_official(args.preset))
    model, _ = official_model(args.preset, grad_checkpointing=True)
    model = model.to(device)
    inject_lora(model, lcfg, seed=args.seed)
    model.train()   # only switches gradient checkpointing on (no dropout / batch norm in DA3)
    params = lora_parameters(model)
    optim = torch.optim.AdamW(params, lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]), betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda(int(tc["warmup"]), steps, float(tc["min_lr_ratio"])))
    use_amp = device.type == "cuda" and tc.get("amp", "fp16") == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = DA3Loss(arm_loss_config(args.arm))
    lam_a = float(tc["anchor_weight"])
    meta = {"arm": args.arm, "seed": args.seed, "preset": args.preset, "base_sha256": base_sha,
            "plan_version": plan["version"], "plan_sha256": __import__("hashlib").sha256(plan_text.encode()).hexdigest(),
            "steps": steps, "steps_overridden": args.steps is not None, "clips": len(clips),
            "train_drives": sorted({d for d, _ in clips}), "commit": git_commit(REPO), "env": environment_info()}

    step = 0
    ck = run / "ckpt" / "last.pt"
    if args.resume and ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        own = dict(model.named_parameters())
        with torch.no_grad():
            for k, v in st["lora"].items():
                own[k].copy_(v.to(device))
        optim.load_state_dict(st["optim"]); sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        set_rng_state(st["rng"]); step = st["step"]
        print(f"resumed {run.name} at step {step}", flush=True)
    else:
        save_adapter(run / "snapshots" / "step_00000.pt", model, base_preset=args.preset, base_sha256=base_sha,
                     seed=args.seed, meta={**meta, "step": 0})

    def checkpoint():
        save_checkpoint(ck, lora=lora_state_dict(model), optim=optim.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), rng=rng_state(), step=step, meta=meta)

    log = open(run / "log.jsonl", "a", encoding="utf-8")
    hist, t0, last_print = [], time.time(), step
    while step < steps:
        i = int(np.random.default_rng([args.seed, step]).integers(len(clips)))   # (seed, step) -> clip
        c = load_cached(cache_dir, i)
        x = to_model_input(c["rgb"]).to(device)
        H, W = c["rgb"].shape[1:3]
        gt = to_device(clip_targets(args.arm, c, (H, W), model.ray_hw(H, W), float(tc["teacher_conf_percentile"]),
                                    float(tc["max_depth_m"])), device)
        frozen = {"depth": torch.from_numpy(c["frozen_depth"])[None].to(device),
                  "ray": torch.from_numpy(c["frozen_ray"].astype(np.float32))[None].to(device),
                  "pose_enc": torch.from_numpy(c["frozen_pose_enc"])[None].to(device)}
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            pred = model(x, ref_strategy="first")
        s_star = align_targets_scale(pred["depth"], gt)
        loss, terms = loss_fn(pred, gt)
        a = anchor_loss(pred, frozen)
        total = loss + lam_a * a
        if not torch.isfinite(total):
            raise SystemExit(f"non-finite loss at step {step}: {terms}")
        scaler.scale(total).backward()
        scaler.unscale_(optim)
        gn = torch.nn.utils.clip_grad_norm_(params, float(tc["grad_clip"]))
        scaler.step(optim); scaler.update(); sched.step()
        step += 1
        rec = {"step": step, "total": float(total.detach()), "anchor": float(a.detach()), "s_star": s_star, "grad_norm": float(gn),
               "lr": sched.get_last_lr()[0], **{k: float(v) for k, v in terms.items()}}
        hist.append(rec["total"])
        log.write(json.dumps(rec) + "\n")
        if step % 20 == 0 or step == 1:
            mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
            print(f"{run.name} step {step}/{steps} | total {rec['total']:.4f} depth {rec.get('depth', 0):.4f} "
                  f"ray {rec['ray']:.4f} cam {rec.get('cam', 0):.4f} anchor {rec['anchor']:.4f} | s* {s_star:.3f} "
                  f"| gn {rec['grad_norm']:.2f} | {(time.time() - t0) / max(1, step - last_print):.2f} s/step | mem {mem:.1f}G",
                  flush=True)
            log.flush(); t0, last_print = time.time(), step
        if step % args.snapshot_every == 0:
            save_adapter(run / "snapshots" / f"step_{step:05d}.pt", model, base_preset=args.preset, base_sha256=base_sha,
                         seed=args.seed, meta={**meta, "step": step})
        if step % args.ckpt_every == 0:
            checkpoint()
        if args.time_budget_h and time.time() - t_start > args.time_budget_h * 3600:
            checkpoint()
            print(f"time budget reached at step {step}: checkpoint saved, continue with --resume", flush=True)
            return
    checkpoint()
    save_adapter(run / "adapter.pt", model, base_preset=args.preset, base_sha256=base_sha, seed=args.seed,
                 meta={**meta, "step": step})
    summary = {**meta, "final_step": step, "last50_total": float(np.mean(hist[-50:])) if hist else None,
               "time_s": time.time() - t_start}
    (run / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print("done:", run / "adapter.pt", flush=True)


if __name__ == "__main__":
    main()
