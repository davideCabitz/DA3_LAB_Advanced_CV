"""Loading external weights into our modules.

* official DA3 checkpoints (HF `depth-anything/DA3-*`, safetensors) -> our key names
* DINOv2 pretrained ViT weights (training init) -> backbone
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from da3.config import PRESETS, ModelConfig

HF_REPOS = {"da3-small": "depth-anything/DA3-SMALL", "da3-base": "depth-anything/DA3-BASE",
            "da3-large": "depth-anything/DA3-LARGE-1.1"}

# (official regex, our replacement). Applied in order on keys with the "model." prefix removed.
_RULES = [
    (r"^backbone\.pretrained\.", "backbone."),
    (r"^cam_dec\.", "cam_head."),
    # Dual-DPT shared part
    (r"^head\.scratch\.layer(\d)_rn\.", lambda m: f"head.adapters.{int(m[1]) - 1}."),
    # fusion stacks: refinenet4 (coarsest) is our blocks.0, refinenet1 is blocks.3
    (r"^head\.scratch\.refinenet(\d)_aux\.", lambda m: f"head.ray_fusion.blocks.{4 - int(m[1])}."),
    (r"^head\.scratch\.refinenet(\d)\.", lambda m: f"head.depth_fusion.blocks.{4 - int(m[1])}."),
    (r"^head\.scratch\.output_conv1\.", "head.depth_neck."),
    (r"^head\.scratch\.output_conv2\.", "head.depth_out."),
    (r"^head\.scratch\.output_conv1_aux\.3\.", "head.ray_neck."),
    (r"^head\.scratch\.output_conv2_aux\.3\.0\.", "head.ray_out.0."),
    # The released DualDPT builds this LayerNorm once and reuses the same module in all four ray
    # output levels, so the checkpoint stores it only under level 0. Level 3 uses those weights.
    (r"^head\.scratch\.output_conv2_aux\.0\.2\.", "head.ray_out.1."),
    (r"^head\.scratch\.output_conv2_aux\.3\.5\.", "head.ray_out.3."),
    (r"\.resConfUnit1\.", ".res1."),
    (r"\.resConfUnit2\.", ".res2."),
]
# Released but never used in the forward pass: ray-branch output layers for pyramid levels 1-3.
_UNUSED = re.compile(r"^head\.scratch\.(output_conv1_aux\.[012]\.|output_conv2_aux\.[012]\.[05]\.)")


def map_official_key(k: str) -> str | None:
    k = k.removeprefix("model.")
    if _UNUSED.match(k):
        return None
    for pat, rep in _RULES:
        k = re.sub(pat, rep, k)
    return k


def convert_official_state_dict(sd: dict) -> tuple[dict, list[str]]:
    out, dropped = {}, []
    for k, v in sd.items():
        nk = map_official_key(k)
        if nk is None:
            dropped.append(k)
        else:
            out[nk] = v
    return out, dropped


def load_safetensors(path: str | Path) -> dict:
    from safetensors.torch import load_file

    return load_file(str(path))


def download_official(preset: str, local_dir: str | Path = "weights") -> Path:
    from huggingface_hub import hf_hub_download

    repo = HF_REPOS[preset]
    return Path(hf_hub_download(repo, "model.safetensors", local_dir=str(Path(local_dir) / repo.split("/")[1])))


def load_official_into(model: torch.nn.Module, path: str | Path, strict: bool = True) -> dict:
    sd, dropped = convert_official_state_dict(load_safetensors(path))
    res = model.load_state_dict(sd, strict=strict)
    return {"dropped_unused": dropped, "missing": list(res.missing_keys), "unexpected": list(res.unexpected_keys)}


def official_model(preset: str = "da3-small", path: str | Path | None = None, **overrides):
    """Build our DA3 with a preset config and load the released weights."""
    from da3.model import DA3

    cfg = ModelConfig.from_dict({**PRESETS[preset].to_dict(), **overrides})
    model = DA3(cfg)
    path = path or download_official(preset)
    info = load_official_into(model, path)
    return model.eval(), info


def load_dinov2_into_backbone(backbone: torch.nn.Module, sd: dict) -> dict:
    """DINOv2 hub checkpoint (keys: cls_token, pos_embed, patch_embed.*, blocks.*, norm.*)."""
    sd = {k: v for k, v in sd.items() if k != "mask_token"}
    res = backbone.load_state_dict(sd, strict=False)
    # camera_token and the QK-norm parameters are new (randomly / identity initialized)
    allowed_missing = re.compile(r"^(camera_token|blocks\.\d+\.attn\.[qk]_norm\.(weight|bias))$")
    bad = [k for k in res.missing_keys if not allowed_missing.match(k)]
    if bad or res.unexpected_keys:
        raise RuntimeError(f"DINOv2 load mismatch: missing={bad} unexpected={res.unexpected_keys}")
    return {"missing_new_params": list(res.missing_keys)}
