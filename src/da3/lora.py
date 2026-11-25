"""Switchable LoRA adapters for the released DA3 (platform of enhancement #3, docs/LORA.md Phase 3).

For a frozen linear layer y = W x + b the adapter adds a trainable low-rank path

    y = W x + b + (alpha / r) * B (A dropout(x)),   A in R^{r x in}, B in R^{out x r}

with A random (Kaiming-uniform, as in PEFT) and B = 0, so at step 0 the model is exactly the
released DA3. Design rules:

* Adapters are inserted **by module name** (`backbone.blocks.{i}.{target}`), never by class: the
  camera encoder E_c reuses the backbone's Attention / Mlp classes and must stay untouched (it is the
  frozen teacher's path).
* `LoRALinear` subclasses nn.Linear and keeps the base `weight` / `bias` Parameters under their
  original names, so the model's state-dict keys for the base weights do not change; only
  `<layer>.lora_A` / `<layer>.lora_B` are added.
* Off = bit-identical: a disabled layer computes exactly nn.Linear's F.linear(x, W, b); the low-rank
  path is skipped, not multiplied by zero.
* Merge writes W + (alpha/r) B A into the weight for zero overhead; unmerge restores the saved
  original W bit-exactly (not W - delta, which would accumulate rounding).
* Adapter-only checkpoints store the LoRA config, A/B tensors and the SHA-256 of the base weights
  file; loading onto a different base is refused.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ADAPTER_FORMAT = "da3-lora/1"


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    blocks: tuple = (4, 5, 6, 7, 8, 9, 10, 11)                       # released L_g blocks (S / B)
    targets: tuple = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    @classmethod
    def from_dict(cls, d: dict) -> "LoRAConfig":
        d = {k: v for k, v in dict(d).items() if k != "frozen"}  # "frozen" in YAML = documentation only
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown LoRA config keys: {sorted(unknown)}")
        d = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}
        return cls(**d)

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    def module_names(self) -> list[str]:
        return [f"backbone.blocks.{i}.{t}" for i in self.blocks for t in self.targets]


class LoRALinear(nn.Linear):
    """nn.Linear with a switchable low-rank residual. Shares the base layer's Parameters."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0,
                 generator: torch.Generator | None = None):
        nn.Module.__init__(self)
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight, self.bias = base.weight, base.bias          # same Parameter objects, same key names
        self.rank, self.alpha, self.scale = rank, float(alpha), float(alpha) / rank
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=dev, dtype=dt))
        self.reset_lora(generator)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_enabled = True
        self.merged = False
        self._weight_backup: torch.Tensor | None = None

    @torch.no_grad()
    def reset_lora(self, generator: torch.Generator | None = None):
        """A ~ kaiming_uniform_(a=sqrt(5)) as in nn.Linear / PEFT (bound 1/sqrt(fan_in)), B = 0.
        Drawn on the CPU (optionally with a seeded generator) so the init does not depend on the device."""
        bound = 1.0 / math.sqrt(self.in_features)
        a = torch.empty(self.lora_A.shape).uniform_(-bound, bound, generator=generator)
        self.lora_A.copy_(a)
        self.lora_B.zero_()

    def delta_weight(self) -> torch.Tensor:
        return self.scale * (self.lora_B.float() @ self.lora_A.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if not self.lora_enabled or self.merged:
            return y
        return y + self.scale * F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)

    @torch.no_grad()
    def merge(self):
        if self.merged:
            return
        if not self.lora_enabled:
            raise RuntimeError("cannot merge a disabled LoRA layer")
        self._weight_backup = self.weight.detach().clone()
        self.weight.copy_((self.weight.float() + self.delta_weight()).to(self.weight.dtype))
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if not self.merged:
            return
        self.weight.copy_(self._weight_backup)
        self._weight_backup = None
        self.merged = False

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, rank={self.rank}, alpha={self.alpha}, enabled={self.lora_enabled}"


# ----------------------------------------------------------------------------- model-level helpers
def lora_layers(model: nn.Module) -> dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def has_lora(model: nn.Module) -> bool:
    return any(isinstance(m, LoRALinear) for m in model.modules())


def inject_lora(model: nn.Module, cfg: LoRAConfig | dict, seed: int = 0, freeze_base: bool = True) -> list[str]:
    """Replace the configured nn.Linear modules by LoRALinear (in place). Returns their names.

    Inject AFTER loading the base weights. A's init is a function of (seed, layer index) only, so the
    same seed gives the same adapters on any device. With freeze_base, every non-LoRA parameter gets
    requires_grad = False.
    """
    cfg = cfg if isinstance(cfg, LoRAConfig) else LoRAConfig.from_dict(cfg)
    if has_lora(model):
        raise RuntimeError("model already has LoRA layers")
    names = cfg.module_names()
    for k, name in enumerate(names):
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        base = getattr(parent, child)
        if type(base) is not nn.Linear:
            raise TypeError(f"{name} is {type(base).__name__}, expected nn.Linear")
        g = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + k)
        setattr(parent, child, LoRALinear(base, cfg.rank, cfg.alpha, cfg.dropout, generator=g))
    if freeze_base:
        for n, p in model.named_parameters():
            p.requires_grad_(".lora_" in n)
    model.lora_config = cfg
    return names


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if ".lora_" in n]


def count_lora_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in lora_parameters(model))


def set_lora_enabled(model: nn.Module, enabled: bool):
    """Switch every adapter on / off. Turning a merged adapter off unmerges it first (bit-exact)."""
    for m in lora_layers(model).values():
        if not enabled and m.merged:
            m.unmerge()
        m.lora_enabled = bool(enabled)


@contextlib.contextmanager
def lora_disabled(model: nn.Module):
    """Run the released model inside the block (e.g. frozen anchor targets), then restore."""
    layers = lora_layers(model).values()
    state = [(m, m.lora_enabled, m.merged) for m in layers]
    set_lora_enabled(model, False)
    try:
        yield model
    finally:
        for m, enabled, merged in state:
            m.lora_enabled = enabled
            if merged:
                m.merge()


def merge_lora(model: nn.Module):
    for m in lora_layers(model).values():
        m.merge()


def unmerge_lora(model: nn.Module):
    for m in lora_layers(model).values():
        m.unmerge()


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if ".lora_" in n}


# ----------------------------------------------------------------------------- adapter files
def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def save_adapter(path: str | Path, model: nn.Module, *, base_preset: str, base_sha256: str | None,
                 seed: int, meta: dict | None = None) -> Path:
    """Adapter-only checkpoint (a few MB). Merged layers are saved from their A / B, not the weight."""
    if not has_lora(model):
        raise RuntimeError("model has no LoRA layers")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"format": ADAPTER_FORMAT, "lora_config": model.lora_config.to_dict(), "base_preset": base_preset,
             "base_sha256": base_sha256, "seed": int(seed), "tensors": lora_state_dict(model), "meta": meta or {}}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    return path


def load_adapter(model: nn.Module, path: str | Path, base_sha256: str | None = None,
                 base_preset: str | None = None) -> dict:
    """Inject (if needed) and load an adapter file. Refuses a different base (SHA-256 / preset)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != ADAPTER_FORMAT:
        raise ValueError(f"{path}: not a {ADAPTER_FORMAT} adapter file")
    if base_sha256 and state.get("base_sha256") and base_sha256 != state["base_sha256"]:
        raise ValueError(f"{path}: adapter was trained on base weights {state['base_sha256'][:12]}…, "
                         f"not {base_sha256[:12]}…")
    if base_preset and state.get("base_preset") and base_preset != state["base_preset"]:
        raise ValueError(f"{path}: adapter is for {state['base_preset']}, not {base_preset}")
    cfg = LoRAConfig.from_dict(state["lora_config"])
    if not has_lora(model):
        inject_lora(model, cfg, seed=state.get("seed", 0))
    elif model.lora_config != cfg:
        raise ValueError(f"{path}: LoRA config {cfg} differs from the model's {model.lora_config}")
    if any(m.merged for m in lora_layers(model).values()):
        raise RuntimeError("unmerge the model before loading an adapter")
    own = dict(model.named_parameters())
    missing = [k for k in own if ".lora_" in k and k not in state["tensors"]]
    unexpected = [k for k in state["tensors"] if k not in own]
    if missing or unexpected:
        raise ValueError(f"{path}: adapter tensors do not match (missing {missing[:3]}, unexpected {unexpected[:3]})")
    with torch.no_grad():
        for k, v in state["tensors"].items():
            own[k].copy_(v.to(own[k].device, own[k].dtype))
    return {k: state[k] for k in ("lora_config", "base_preset", "base_sha256", "seed", "meta")}


def adapter_info(path: str | Path) -> dict:
    """What a result record stores about an adapter (no model needed)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != ADAPTER_FORMAT:
        raise ValueError(f"{path}: not a {ADAPTER_FORMAT} adapter file")
    meta = state.get("meta", {})
    return {"path": str(path), "sha256": file_sha256(path), "lora_config": state["lora_config"],
            "base_preset": state.get("base_preset"), "base_sha256": state.get("base_sha256"), "seed": state.get("seed"),
            "arm": meta.get("arm"), "step": meta.get("step"), "plan_version": meta.get("plan_version"),
            "plan_sha256": meta.get("plan_sha256"), "train_drives": meta.get("train_drives")}


def attach_adapter(model: nn.Module, weights: str, adapter: str | Path) -> dict:
    """Load an adapter onto a model built from released weights ("official:<preset>"); the adapter must
    have been trained on exactly those weights (SHA-256 check). The LoRA stays unmerged (exact path)."""
    if not weights.startswith("official:"):
        raise ValueError("adapters are defined on the released weights: use --weights official:<preset>")
    preset = weights.split(":", 1)[1]
    from da3.checkpoint_io import download_official

    load_adapter(model, adapter, base_sha256=file_sha256(download_official(preset)), base_preset=preset)
    return adapter_info(adapter)
