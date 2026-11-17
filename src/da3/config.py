"""Model configuration. YAML files in configs/model/ map 1:1 onto ModelConfig.

Two families of presets:

* `paper-*`  : the paper as written (Sec. 3.1-3.2). This is OUR BASELINE and the only family used
               for training. Where the paper is silent on a necessary detail (tapped layers, head
               widths, output activations) the value is taken from the authors' release.
* `da3-*`    : the released architecture, which differs from the paper text in several places
               (docs/PLAN.md). Used only to load the released checkpoints (parity, Phase 2, and the
               "[official]" evaluation rows in Phase 4).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    name: str = "vits"                    # vits | vitb | vitl (DINOv2)
    patch_size: int = 14
    out_layers: tuple = (5, 7, 9, 11)     # blocks whose outputs feed the 4 DPT levels
    # --- backbone layout: blocks [0, alt_start) within-view (L_s); from alt_start on (L_g) blocks
    # alternate cross-view / within-view, starting with cross-view if cross_first.
    alt_start: int = 8
    cross_first: bool = True
    # --- camera token
    cam_token_mode: str = "prepend"       # "prepend": extra token per view from the input on
    # Learnable tokens when poses are unknown. The paper describes one shared token, but then the
    # network cannot tell which view is view 0 (whose frame the pose targets use): camera loss 0.97
    # vs 0.065 in our overfit test. Decision (project owner, 2026-11-17): 2 = [reference view, shared
    # token for all other views], as in the release. The only deliberate deviation from the paper text.
    num_cam_tokens: int = 2
    cam_enc_depth: int = 0                # E_c = MLP (0) [+ transformer trunk blocks, released]
    use_cam_enc: bool = True
    use_cam_head: bool = True
    # --- head (Dual-DPT)
    cat_token: bool = False               # released: concat within-view + current features
    head_features: int = 64
    head_out_channels: tuple = (48, 96, 192, 384)
    ray_head: str = "paper"               # "paper": 6-ch ray at H/2 x W/2 | "released"
    # --- released-only features (off = paper)
    qknorm_start: int = -1
    rope_start: int = -1
    rope_base: float = 100.0
    ref_view_selection: bool = False
    head_token_norm: bool = False
    head_pos_embed: bool = False
    # --- engineering (no effect on the math)
    head_chunk_size: int = 8
    grad_checkpointing: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        d = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}
        return cls(**d)

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


_PAPER_S = ModelConfig()
_RELEASED = dict(
    alt_start=4, cross_first=False, cam_token_mode="replace_cls", num_cam_tokens=2, cam_enc_depth=4,
    cat_token=True, ray_head="released", qknorm_start=4, rope_start=4, ref_view_selection=True,
    head_token_norm=True, head_pos_embed=True,
)
_BASE_HEAD = dict(name="vitb", head_features=128, head_out_channels=(96, 192, 384, 768))
_LARGE_HEAD = dict(name="vitl", out_layers=(11, 15, 19, 23), head_features=256,
                   head_out_channels=(256, 512, 1024, 1024))

PRESETS = {
    # paper (our baseline): L = 12 -> L_s = 8, L_g = 4; L = 24 -> L_s = 16, L_g = 8
    "paper-small": _PAPER_S,
    "paper-base": replace(_PAPER_S, **_BASE_HEAD),
    "paper-large": replace(_PAPER_S, alt_start=16, **_LARGE_HEAD),
    # released checkpoints
    "da3-small": replace(_PAPER_S, **_RELEASED),
    "da3-base": replace(_PAPER_S, **_RELEASED, **_BASE_HEAD),
    "da3-large": replace(_PAPER_S, **{**_RELEASED, "alt_start": 8, "qknorm_start": 8, "rope_start": 8}, **_LARGE_HEAD),
}


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_model_config(spec) -> ModelConfig:
    """Preset name, YAML path, or dict (a dict may set `preset:` plus overrides)."""
    if isinstance(spec, ModelConfig):
        return spec
    if isinstance(spec, str) and spec in PRESETS:
        return PRESETS[spec]
    d = dict(spec) if isinstance(spec, dict) else load_yaml(spec)
    d = dict(d.get("model", d))
    base = PRESETS[d.pop("preset")].to_dict() if "preset" in d else {}
    base.update(d)
    return ModelConfig.from_dict(base)
