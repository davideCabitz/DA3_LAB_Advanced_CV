"""Training utilities: LR schedule, resumable checkpoints, experiment registry."""

from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


def lr_lambda(warmup: int, total: int, min_ratio: float):
    """Linear warm-up to 1, then cosine decay to min_ratio at `total`."""
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        p = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return f


def rng_state() -> dict:
    s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s: dict):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def save_checkpoint(path: Path, **state):
    """Atomic save (write to tmp, then rename) so a killed session never leaves a broken file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    p = ckpt_dir / "last.pt"
    return p if p.exists() else None


def git_commit(repo: Path | None = None) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout
        return out.stdout.strip() + ("-dirty" if dirty.strip() else "")
    except Exception:
        return "unknown"


def environment_info() -> dict:
    info = {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform()}
    if torch.cuda.is_available():
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["cuda"] = torch.version.cuda
    return info


def append_registry(registry: Path, record: dict):
    registry.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **record}
    with open(registry, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
