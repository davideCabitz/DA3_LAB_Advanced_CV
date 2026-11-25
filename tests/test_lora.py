"""LoRA platform (enhancement #3, docs/LORA.md Phase 3): off = bit-identical, B = 0 at init, only A/B
train, merge = unmerged, unmerge exact, E_c untouched, adapter round trip."""

import copy

import pytest
import torch

from da3.lora import (
    LoRAConfig,
    LoRALinear,
    count_lora_parameters,
    inject_lora,
    load_adapter,
    lora_disabled,
    lora_layers,
    lora_parameters,
    merge_lora,
    save_adapter,
    set_lora_enabled,
    unmerge_lora,
)
from helpers import tiny_model

H, W = 70, 98  # 5 x 7 patches
KEYS = ("depth", "depth_conf", "ray", "ray_conf", "pose_enc")


def _inputs(S=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, S, 3, H, W, generator=g)


def _run(m, x, **kw):
    with torch.no_grad():
        return m(x, **kw)


def _assert_equal(a, b):
    for k in KEYS:
        if k in a:
            assert torch.equal(a[k], b[k]), k


def _randomize_B(m, seed=3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in lora_layers(m).values():
            layer.lora_B.copy_(torch.randn(layer.lora_B.shape, generator=g) * 0.05)


def test_injection_targets_only_blocks_4_to_11_and_freezes_the_base():
    m = tiny_model("da3-small")
    names = inject_lora(m, LoRAConfig())
    assert len(names) == 8 * 4
    assert set(lora_layers(m)) == set(names)
    for i in range(4):  # L_s blocks untouched
        assert type(m.backbone.blocks[i].attn.qkv) is torch.nn.Linear
    # E_c reuses the Attention / Mlp classes but must not be adapted (teacher path)
    assert not any(isinstance(mod, LoRALinear) for mod in m.cam_enc.modules())
    assert not any(isinstance(mod, LoRALinear) for mod in m.head.modules())
    assert not any(isinstance(mod, LoRALinear) for mod in m.cam_head.modules())
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all(".lora_" in n for n in trainable)
    assert count_lora_parameters(m) == 786_432  # ViT-S, r = 16, qkv + proj + fc1 + fc2, 8 blocks (docs/LORA.md)


def test_base_state_dict_keys_are_unchanged():
    m = tiny_model("da3-small")
    ref_keys = set(m.state_dict())
    inject_lora(m, LoRAConfig(rank=4))
    keys = set(m.state_dict())
    assert ref_keys <= keys and all(".lora_" in k for k in keys - ref_keys)
    res = m.load_state_dict(tiny_model("da3-small").state_dict(), strict=False)  # released ckpt still loads
    assert not res.unexpected_keys and all(".lora_" in k for k in res.missing_keys)


@pytest.mark.parametrize("posed", [False, True])
def test_B_zero_at_init_gives_the_released_outputs(posed):
    from da3.camera import affine_inverse
    from helpers import random_camera

    ref = tiny_model("da3-small")
    m = copy.deepcopy(ref)
    inject_lora(m, LoRAConfig(), seed=0)
    assert all(torch.count_nonzero(l.lora_B) == 0 for l in lora_layers(m).values())
    assert all(torch.count_nonzero(l.lora_A) > 0 for l in lora_layers(m).values())
    x, kw = _inputs(), {}
    if posed:
        K, c2w = random_camera(B=3, H=H, W=W, dtype=torch.float32)
        kw = dict(w2c=affine_inverse(c2w)[None], K=K[None])
    _assert_equal(_run(ref, x, **kw), _run(m, x, **kw))


@pytest.mark.parametrize("S", [1, 4])
def test_lora_off_is_bit_identical_and_on_changes_outputs(S):
    ref = tiny_model("da3-small")
    m = copy.deepcopy(ref)
    inject_lora(m, LoRAConfig(rank=8))
    _randomize_B(m)
    x = _inputs(S)
    on = _run(m, x)
    assert not torch.allclose(on["depth"], _run(ref, x)["depth"])
    set_lora_enabled(m, False)
    _assert_equal(_run(ref, x), _run(m, x))
    set_lora_enabled(m, True)
    _assert_equal(on, _run(m, x))
    with lora_disabled(m):                      # context manager: released model inside, restored after
        _assert_equal(_run(ref, x), _run(m, x))
    _assert_equal(on, _run(m, x))


def test_only_A_and_B_receive_gradients_and_change():
    m = tiny_model("da3-small", grad_checkpointing=True)
    inject_lora(m, LoRAConfig(rank=4))
    base = {n: p.detach().clone() for n, p in m.named_parameters() if ".lora_" not in n}
    opt = torch.optim.AdamW(lora_parameters(m), lr=1e-3)
    m.train()
    out = m(_inputs())
    (out["depth"].mean() + out["ray"].abs().mean() + out["pose_enc"].abs().mean()).backward()
    for n, p in m.named_parameters():
        if ".lora_" in n:
            assert p.grad is not None, n
        else:
            assert p.grad is None, n
    layers = list(lora_layers(m).values())
    # with B = 0, dL/dA = 0 and dL/dB != 0 (the intended LoRA asymmetry)
    assert all(torch.count_nonzero(l.lora_A.grad) == 0 for l in layers)
    assert any(torch.count_nonzero(l.lora_B.grad) > 0 for l in layers)
    opt.step()
    for n, p in m.named_parameters():
        if ".lora_" not in n:
            assert torch.equal(p, base[n]), n
    assert any(torch.count_nonzero(l.lora_B) > 0 for l in layers)


def test_merge_matches_unmerged_and_unmerge_is_exact():
    m = tiny_model("da3-small")
    inject_lora(m, LoRAConfig(rank=8))
    _randomize_B(m)
    w0 = {n: l.weight.detach().clone() for n, l in lora_layers(m).items()}
    x = _inputs()
    unmerged = _run(m, x)
    merge_lora(m)
    assert all(l.merged for l in lora_layers(m).values())
    merged = _run(m, x)
    for k in KEYS:
        if k in merged:
            assert torch.allclose(merged[k], unmerged[k], atol=1e-4, rtol=1e-4), k
    unmerge_lora(m)
    for n, l in lora_layers(m).items():
        assert torch.equal(l.weight, w0[n]), n
    _assert_equal(unmerged, _run(m, x))
    # switching a merged model off restores the released weights exactly
    ref_out = _run(tiny_model("da3-small"), x)
    merge_lora(m)
    set_lora_enabled(m, False)
    _assert_equal(ref_out, _run(m, x))


def test_seed_determines_init():
    a, b, c = (tiny_model("da3-small") for _ in range(3))
    inject_lora(a, LoRAConfig(rank=4), seed=0)
    inject_lora(b, LoRAConfig(rank=4), seed=0)
    inject_lora(c, LoRAConfig(rank=4), seed=1)
    la, lb, lc = (list(lora_layers(m).values()) for m in (a, b, c))
    assert all(torch.equal(x.lora_A, y.lora_A) for x, y in zip(la, lb))
    assert not torch.equal(la[0].lora_A, lc[0].lora_A)


def test_adapter_save_load_round_trip(tmp_path):
    m = tiny_model("da3-small")
    inject_lora(m, LoRAConfig(rank=4), seed=2)
    _randomize_B(m)
    x = _inputs()
    out = _run(m, x)
    p = save_adapter(tmp_path / "adapter.pt", m, base_preset="da3-small", base_sha256="ab" * 32, seed=2,
                     meta={"note": "test"})
    assert p.stat().st_size < 4 * 2**20
    m2 = tiny_model("da3-small")
    info = load_adapter(m2, p, base_sha256="ab" * 32, base_preset="da3-small")
    assert info["seed"] == 2 and info["meta"]["note"] == "test"
    _assert_equal(out, _run(m2, x))
    with pytest.raises(ValueError):
        load_adapter(tiny_model("da3-small"), p, base_sha256="cd" * 32)
    with pytest.raises(ValueError):
        load_adapter(tiny_model("da3-small"), p, base_preset="da3-base")


def test_config_round_trip_and_analysis_plan_constants():
    from pathlib import Path

    import yaml

    cfg = LoRAConfig.from_dict({"rank": 8, "blocks": [4, 5], "frozen": ["doc only"]})
    assert cfg.blocks == (4, 5) and LoRAConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError):
        LoRAConfig.from_dict({"rnak": 8})
    plan = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "lora_analysis.yaml").read_text())
    assert LoRAConfig.from_dict(plan["lora"]) == LoRAConfig()  # pre-registered recipe = the defaults
