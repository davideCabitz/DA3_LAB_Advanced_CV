"""Multi-view training dataset + dynamic token-budget batching (Sec. 3.4, reduced scale).

A batch is described by (resolution, n_views, batch_size) with
    batch_size * n_views * (H/14) * (W/14) <= token_budget,
so the number of patch tokens per step stays roughly constant. The sampler is a pure function of
(seed, step), which makes resuming exact: restarting at step k yields the same batches.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from da3.model import IMAGENET_MEAN, IMAGENET_STD
from .canonical import CanonicalScene, list_scenes


@dataclass
class SourceSpec:
    name: str
    root: str
    weight: float = 1.0
    max_frame_gap: int = 8        # views come from a window of n_views * max_frame_gap frames
    max_depth: float | None = None  # depths beyond this are invalid (e.g. sky in TartanAir)


@dataclass
class DataConfig:
    sources: list = field(default_factory=list)  # list[SourceSpec | dict]
    resolutions: list = field(default_factory=lambda: [[252, 252], [252, 336], [336, 252], [252, 308]])
    views: tuple = (2, 8)
    token_budget: int = 6000
    max_batch: int = 8
    num_workers: int = 2


class MultiViewDataset(Dataset):
    """Index = (source_id, scene_id, n_views, H, W, seed) produced by TokenBudgetSampler."""

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.sources = [s if isinstance(s, SourceSpec) else SourceSpec(**s) for s in cfg.sources]
        self.scenes = []
        for s in self.sources:
            paths = list_scenes(s.root)
            if not paths:
                raise FileNotFoundError(f"no scenes with cameras.json under {s.root}")
            self.scenes.append([CanonicalScene(p) for p in paths])

    def __len__(self):
        return sum(len(s) for s in self.scenes)

    def summary(self) -> dict:
        return {s.name: {"scenes": len(sc), "frames": sum(len(x) for x in sc)}
                for s, sc in zip(self.sources, self.scenes)}

    def _pick_frames(self, scene: CanonicalScene, n: int, gap: int, rng: random.Random) -> list[int]:
        N = len(scene)
        n = min(n, N)
        span = min(N, max(n, n * gap))
        start = rng.randint(0, N - span)
        idx = rng.sample(range(start, start + span), n)
        rng.shuffle(idx)
        return idx

    def _crop(self, rgb, depth, K, H, W, rng):
        """Resize so the image covers H x W (aspect preserved), then crop to the training
        resolution. No other augmentation (none is described in the paper)."""
        h, w = rgb.shape[:2]
        s = max(H / h, W / w)
        nh, nw = max(H, math.ceil(h * s)), max(W, math.ceil(w * s))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST)
        K = K.copy()
        K[0] *= nw / w
        K[1] *= nh / h
        y0, x0 = rng.randint(0, nh - H), rng.randint(0, nw - W)
        K[0, 2] -= x0
        K[1, 2] -= y0
        return rgb[y0 : y0 + H, x0 : x0 + W], depth[y0 : y0 + H, x0 : x0 + W], K

    def __getitem__(self, key):
        src_id, scene_id, n_views, H, W, seed = key
        rng = random.Random(seed)
        spec, scene = self.sources[src_id], self.scenes[src_id][scene_id]
        idx = self._pick_frames(scene, n_views, spec.max_frame_gap, rng)
        imgs, depths, Ks, c2ws = [], [], [], []
        for i in idx:
            rgb, d = scene.load(i)
            rgb, d, K = self._crop(rgb, d, scene.K[i], H, W, rng)
            imgs.append(rgb)
            depths.append(d)
            Ks.append(K)
            c2ws.append(scene.c2w[i])
        img = torch.from_numpy(np.stack(imgs)).float().div(255).permute(0, 3, 1, 2)
        img = (img - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(IMAGENET_STD)[:, None, None]
        depth = torch.from_numpy(np.stack(depths)).float()
        valid = depth > 0
        if spec.max_depth:
            valid &= depth < spec.max_depth
        c2w = torch.from_numpy(np.stack(c2ws)).float()
        return {
            "images": img,
            "depth": depth,
            "valid": valid,
            "K": torch.from_numpy(np.stack(Ks)).float(),
            "w2c": torch.linalg.inv(c2w),
            "source": src_id,
        }


class TokenBudgetSampler(Sampler):
    """Infinite batch sampler; batch k is a deterministic function of (seed, k)."""

    def __init__(self, dataset: MultiViewDataset, cfg: DataConfig, seed: int, start_step: int = 0,
                 patch: int = 14, fixed_batch: bool = False):
        self.ds, self.cfg, self.seed, self.step, self.patch = dataset, cfg, seed, start_step, patch
        self.fixed_batch = fixed_batch
        w = np.array([s.weight for s in dataset.sources], np.float64)
        self.src_p = w / w.sum()

    def batch_spec(self, step: int):
        rng = random.Random(self.seed * 1_000_003 + (0 if self.fixed_batch else step))
        H, W = rng.choice(self.cfg.resolutions)
        S = rng.randint(*self.cfg.views)
        per_view = (H // self.patch) * (W // self.patch)
        B = max(1, min(self.cfg.max_batch, self.cfg.token_budget // (S * per_view)))
        items = []
        for _ in range(B):
            src = int(np.searchsorted(np.cumsum(self.src_p), rng.random(), side="right"))
            src = min(src, len(self.src_p) - 1)
            scene = rng.randrange(len(self.ds.scenes[src]))
            items.append((src, scene, S, H, W, rng.getrandbits(32)))
        return items

    def __iter__(self):
        while True:
            yield self.batch_spec(self.step)
            self.step += 1

    def __len__(self):  # infinite; DataLoader only needs it for progress bars
        return 2**31


def collate(batch: list[dict]) -> dict:
    out = {k: torch.stack([b[k] for b in batch]) for k in ("images", "depth", "valid", "K", "w2c")}
    out["source"] = torch.tensor([b["source"] for b in batch])
    return out
