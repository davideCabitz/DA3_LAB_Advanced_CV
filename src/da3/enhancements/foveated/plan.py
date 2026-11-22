"""Focus-view selection, per-view resolution plans and token / FLOP budgets (enhancement #2).

A *foveated* configuration gives a fraction of the views (the focus views) the full processing
resolution and the others (context views) a lower one, all in one forward pass
(docs/da3_enhancement_report.md §4.3). Resolutions follow the released API: `res` is the longest
side, both sides are then rounded to the nearest multiple of the patch size (io_utils.target_size).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from da3.io_utils import target_size

PATCH = 14
STRATEGIES = ("stride", "random", "first")


@dataclass(frozen=True)
class FoveationSpec:
    """One foveated configuration.

    fraction / focus_res / context_res: share of focus views and the two resolutions (longest side).
    strategy: which views are focus views: "stride" (evenly spaced, default), "random" (seeded) or
      "first" (the first k views). View 0 is always a focus view: it is the reference view of the
      paper protocol (ref_view_strategy "first"), and keeping the reference at full resolution is the
      report's mitigation of a resolution-biased reference choice.
    ref: candidates when DA3 picks the reference view itself (released "saddle_balanced"): "focus"
      (default) or "all".
    balance: attention-mass balancing in the cross-view blocks (routing.py), training-free.
    """

    fraction: float = 0.25
    focus_res: int = 504
    context_res: int = 252
    strategy: str = "stride"
    seed: int = 0
    balance: bool = False
    ref: str = "focus"

    def __post_init__(self):
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction}")
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {self.strategy!r}")
        if self.ref not in ("focus", "all"):
            raise ValueError(f"ref must be 'focus' or 'all', got {self.ref!r}")

    @property
    def name(self) -> str:
        s = f"fov{int(round(100 * self.fraction))}_{self.focus_res}-{self.context_res}"
        if self.strategy != "stride":
            s += f"_{self.strategy}{self.seed if self.strategy == 'random' else ''}"
        if self.ref != "focus":
            s += "_refall"
        if self.balance:
            s += "_bal"
        return s

    @classmethod
    def parse(cls, text: str) -> "FoveationSpec":
        """'fraction=0.25,focus_res=504,context_res=252,balance=1' (any subset of the fields)."""
        types = {f.name: f.type for f in fields(cls)}
        kw = {}
        for item in filter(None, (t.strip() for t in text.split(","))):
            key, sep, val = item.partition("=")
            key = key.strip()
            if not sep or key not in types:
                raise ValueError(f"bad foveation spec item {item!r} in {text!r}; fields: {sorted(types)}")
            val = val.strip()
            if key in ("fraction",):
                kw[key] = float(val)
            elif key in ("focus_res", "context_res", "seed"):
                kw[key] = int(val)
            elif key == "balance":
                kw[key] = val.lower() in ("1", "true", "yes", "on")
            else:
                kw[key] = val
        return cls(**kw)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ----------------------------------------------------------------------------- focus views
def n_focus(n: int, fraction: float | None = None, count: int | None = None) -> int:
    k = count if count is not None else int(round(float(fraction) * n))
    return int(min(n, max(1, k)))


def select_focus(n: int, fraction: float | None = None, count: int | None = None, strategy: str = "stride",
                 seed: int = 0) -> np.ndarray:
    """Boolean mask of the focus views among n views; view 0 is always included."""
    if n <= 0:
        return np.zeros(0, bool)
    k = n_focus(n, fraction, count)
    if strategy == "stride":
        idx = np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))
    elif strategy == "first":
        idx = np.arange(k)
    elif strategy == "random":
        others = np.random.default_rng(seed).permutation(np.arange(1, n))[: k - 1]
        idx = np.concatenate([[0], others]).astype(int)
    else:
        raise ValueError(strategy)
    mask = np.zeros(n, bool)
    mask[idx] = True
    return mask


def view_resolutions(focus: np.ndarray, focus_res: int, context_res: int) -> list[int]:
    return [int(focus_res) if f else int(context_res) for f in np.asarray(focus, bool)]


def subset_with_focus(n: int, focus: np.ndarray, k: int) -> np.ndarray:
    """k views (k >= #focus): all focus views plus evenly spaced others (equal-budget 'fewer views')."""
    focus = np.asarray(focus, bool)
    k = int(min(n, max(int(focus.sum()), k)))
    mask = focus.copy()
    others = np.flatnonzero(~focus)
    extra = k - int(focus.sum())
    if extra > 0 and len(others):
        pick = np.unique(np.round(np.linspace(0, len(others) - 1, extra)).astype(int))
        mask[others[pick]] = True
    return mask


# ----------------------------------------------------------------------------- budgets
def processed_hw(orig_hw, res: int, patch: int = PATCH) -> tuple[int, int]:
    """Size of an (h, w) image after the released preprocessing at longest side `res`."""
    return target_size(int(orig_hw[0]), int(orig_hw[1]), int(res), patch)


def patch_tokens(hw, patch: int = PATCH) -> int:
    return (int(hw[0]) // patch) * (int(hw[1]) // patch)


def backbone_cost(hws, embed_dim: int = 384, depth: int = 12, cross_blocks: int = 4, n_special: int = 1,
                  patch: int = PATCH) -> dict:
    """Backbone cost of one pass over views of sizes `hws`.

    tokens: patch tokens of all views; gflops: 2 x multiply-adds of the linear layers (24 D^2 per
    token and block) plus attention (4 N^2 D per sequence of N tokens: QK^T and AV), within-view
    blocks over each view, cross-view blocks over the concatenation of all views."""
    n = np.array([patch_tokens(hw, patch) + n_special for hw in hws], np.float64)
    T = float(n.sum())
    within = depth - cross_blocks
    lin = 24.0 * embed_dim ** 2 * T * depth
    attn_within = 4.0 * embed_dim * within * float((n ** 2).sum())
    attn_cross = 4.0 * embed_dim * cross_blocks * T ** 2
    return {"tokens": int(T - n_special * len(n)), "gflops": (lin + attn_within + attn_cross) / 1e9,
            "gflops_cross_attn": attn_cross / 1e9}


def model_dims(model) -> dict:
    bb = model.backbone
    return {"embed_dim": bb.embed_dim, "depth": bb.depth,
            "cross_blocks": sum(bb.is_cross_view(i) for i in range(bb.depth)), "n_special": bb.n_special,
            "patch": bb.patch_size}


def equal_budget_res(orig_hw, n_views: int, target_tokens: float, candidates=None, patch: int = PATCH) -> int:
    """Uniform resolution whose total patch-token count over n_views is closest to target_tokens."""
    candidates = range(2 * patch, 1009, patch) if candidates is None else candidates
    return int(min(candidates, key=lambda r: (abs(n_views * patch_tokens(processed_hw(orig_hw, r, patch), patch)
                                                  - target_tokens), -r)))
