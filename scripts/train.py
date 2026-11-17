"""Reduced-scale DA3 training (Phase 3). Every result produced by this script is [reduced-scale].

  python scripts/train.py --config configs/train/reduced_vits_tartanair_vkitti.yaml
  python scripts/train.py --config ... --resume            # continue from <out_dir>/ckpt/last.pt
  python scripts/train.py --config ... --overfit-one-batch  # sanity check: one fixed batch

Single GPU (or CPU). fp16 autocast + GradScaler on CUDA, SDPA attention, optional gradient
checkpointing and accumulation. Checkpoints hold model, optimizer, scheduler, scaler, step and RNG
states; data order is a function of (seed, step), so resuming continues the exact batch stream.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from da3 import DA3
from da3.checkpoint_io import download_official, load_dinov2_into_backbone, load_official_into
from da3.config import load_model_config, load_yaml
from da3.data import DataConfig, MultiViewDataset, TokenBudgetSampler, collate
from da3.losses import DA3Loss, LossConfig, normalize_gt
from da3.train_utils import (
    append_registry,
    environment_info,
    git_commit,
    latest_checkpoint,
    lr_lambda,
    rng_state,
    save_checkpoint,
    set_rng_state,
)

DINOV2_URLS = {
    "vits": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    "vitb": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    "vitl": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
}
REPO = Path(__file__).resolve().parents[1]


def init_weights(model: DA3, init: dict) -> dict:
    kind = init.get("type", "dinov2")
    if kind == "dinov2":
        src = init.get("path") or DINOV2_URLS[model.cfg.name]
        sd = (torch.hub.load_state_dict_from_url(src, map_location="cpu") if src.startswith("http")
              else torch.load(src, map_location="cpu"))
        return {"init": "dinov2", "src": src, **load_dinov2_into_backbone(model.backbone, sd)}
    if kind == "official":
        path = init.get("path") or download_official(init["preset"])
        return {"init": "official", "src": str(path), **load_official_into(model, path)}
    return {"init": "random"}


def move(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overfit-one-batch", action="store_true")
    ap.add_argument("--steps", type=int, default=None, help="override train.steps")
    ap.add_argument("--out-dir", default=None, help="override out_dir")
    ap.add_argument("--data-root", default=None, help="prefix for relative data source roots")
    ap.add_argument("--time-budget-h", type=float, default=0.0,
                    help="stop cleanly (checkpoint + exit 0) after this many hours; 0 = no limit. "
                         "Use it below the Kaggle session limit so outputs are always saved.")
    args = ap.parse_args()
    t_start = time.time()

    cfg = load_yaml(args.config)
    tcfg = cfg["train"]
    if args.steps:
        tcfg["steps"] = args.steps
    run_name = cfg["experiment"] + ("_overfit" if args.overfit_one_batch else "")
    out_dir = Path(args.out_dir or cfg.get("out_dir", "runs")) / run_name
    ckpt_dir = out_dir / "ckpt"
    seed = int(cfg.get("seed", 0))
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and tcfg.get("amp", "fp16") == "fp16"

    # ---------------- data
    dcfg_d = dict(cfg["data"])
    if args.data_root:
        for s in dcfg_d["sources"]:
            if not Path(s["root"]).is_absolute():
                s["root"] = str(Path(args.data_root) / s["root"])
    dcfg = DataConfig(**{k: (tuple(v) if k == "views" else v) for k, v in dcfg_d.items()})
    dataset = MultiViewDataset(dcfg)
    print("data:", json.dumps(dataset.summary()))

    # ---------------- model / optim
    model = DA3(load_model_config(cfg["model"])).to(device)
    init_info = init_weights(model, cfg.get("init", {}))
    print("init:", {k: (v if not isinstance(v, list) else len(v)) for k, v in init_info.items()})
    loss_fn = DA3Loss(LossConfig(**cfg.get("loss", {})))
    # Paper: one peak LR (2e-4) with warm-up; the optimizer is not stated -> AdamW, no weight decay
    # on norms / biases / tokens.
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "token" in n or "pos_embed" in n else decay).append(p)
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": tcfg.get("weight_decay", 0.05)}, {"params": no_decay, "weight_decay": 0.0}],
        lr=tcfg["lr"], betas=(0.9, 0.999),
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda(tcfg.get("warmup", 200), tcfg["steps"], tcfg.get("min_lr_ratio", 0.01))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = 0
    if args.resume and (ck := latest_checkpoint(ckpt_dir)):
        state = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        set_rng_state(state["rng"])
        step = state["step"]
        print(f"resumed from {ck} at step {step}")

    accum = int(tcfg.get("accum", 1))
    sampler = TokenBudgetSampler(dataset, dcfg, seed, start_step=step * accum,
                                 fixed_batch=args.overfit_one_batch)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate,
                        num_workers=dcfg.num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=dcfg.num_workers > 0)
    it = iter(loader)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(out_dir / "tb"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    meta = {"run": run_name, "setting": "reduced-scale", "seed": seed, "commit": git_commit(REPO),
            "config_file": args.config, "env": environment_info(), "init": init_info.get("init")}
    print("meta:", json.dumps(meta))

    pose_p = float(tcfg.get("pose_cond_prob", 0.2))
    ref_strategy = tcfg.get("ref_strategy", "first")
    log_every, ckpt_every = tcfg.get("log_every", 10), tcfg.get("ckpt_every", 250)
    clip = tcfg.get("grad_clip", 1.0)
    model.train()
    t0, tok_count, hist = time.time(), 0, []

    def checkpoint():
        save_checkpoint(ckpt_dir / "last.pt", model=model.state_dict(), optim=optim.state_dict(),
                        sched=sched.state_dict(), scaler=scaler.state_dict(), step=step,
                        rng=rng_state(), config=cfg, meta=meta)

    while step < tcfg["steps"]:
        optim.zero_grad(set_to_none=True)
        logs = {}
        for _ in range(accum):
            batch = move(next(it), device)
            B, S, _, H, W = batch["images"].shape
            gt = normalize_gt(batch["depth"], batch["valid"], batch["K"], batch["w2c"], model.ray_hw(H, W))
            cond = model.cam_enc is not None and random.random() < pose_p
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(batch["images"], w2c=gt["w2c"] if cond else None,
                             K=gt["K"] if cond else None, ref_strategy=ref_strategy)
            loss, terms = loss_fn(pred, gt)
            scaler.scale(loss / accum).backward()
            for k, v in terms.items():
                logs[k] = logs.get(k, 0.0) + v.item() / accum
            tok_count += B * S * (H // 14) * (W // 14)
        scaler.unscale_(optim)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optim)
        scaler.update()
        sched.step()
        step += 1
        hist.append(logs["total"])

        if step % log_every == 0 or step == 1:
            dt = time.time() - t0
            for k, v in logs.items():
                writer.add_scalar(f"loss/{k}", v, step)
            writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
            writer.add_scalar("train/grad_norm", float(gnorm), step)
            writer.add_scalar("train/tokens_per_s", tok_count / dt, step)
            mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
            print(f"step {step} | " + " ".join(f"{k} {v:.4f}" for k, v in logs.items())
                  + f" | B{B} S{S} {H}x{W} | lr {sched.get_last_lr()[0]:.2e} | gn {float(gnorm):.2f}"
                  + f" | {tok_count / dt:.0f} tok/s | mem {mem:.1f}G", flush=True)
            t0, tok_count = time.time(), 0
        if step % ckpt_every == 0:
            checkpoint()
        if args.time_budget_h and time.time() - t_start > args.time_budget_h * 3600:
            print(f"time budget of {args.time_budget_h} h reached at step {step}: checkpointing and stopping "
                  f"(continue with --resume)", flush=True)
            break

    checkpoint()
    final = {"final_loss_avg_last50": float(np.mean(hist[-50:])) if hist else None,
             "first_loss": hist[0] if hist else None, "steps": step, "finished": step >= tcfg["steps"]}
    (out_dir / "train_summary.json").write_text(json.dumps({**meta, **final}, indent=2))
    append_registry(REPO / "experiments" / "registry.jsonl", {**meta, "kind": "train", **final,
                                                              "out_dir": str(out_dir)})
    print("done:", json.dumps(final))


if __name__ == "__main__":
    main()
