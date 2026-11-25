"""LoRA training pieces (src/da3/lora_train.py): gauge of the targets, scale alignment, arms, one step."""

import numpy as np
import pytest
import torch

from da3.lora import LoRAConfig, inject_lora, lora_parameters
from da3.lora_train import align_targets_scale, anchor_loss, api_scale, arm_loss_config, clip_targets, valid_mask
from da3.losses import DA3Loss, normalize_gt
from helpers import tiny_model

H, W = 70, 98
S = 3


def _poses(scale=1.0):
    """w2c (S,4,4): forward motion, view 0 = identity; camera centres multiplied by `scale`."""
    w2c = np.tile(np.eye(4, dtype=np.float32), (S, 1, 1))
    for i in range(S):
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 3] = [0.1 * i, 0.0, 1.5 * i]
        c2w[:3, 3] *= scale
        w2c[i] = np.linalg.inv(c2w)
    return w2c


def _cache(teacher_scale=2.5, frozen_scale=0.7):
    rng = np.random.default_rng(0)
    K = np.tile(np.array([[60, 0, W / 2], [0, 60, H / 2], [0, 0, 1]], np.float32), (S, 1, 1))
    depth = rng.uniform(3, 30, (S, H, W)).astype(np.float32)
    conf = rng.uniform(1, 5, (S, H, W)).astype(np.float32)
    return {"K": K, "w2c": _poses(), "rgb": rng.integers(0, 255, (S, H, W, 3), dtype=np.uint8),
            "teacher_depth": depth * teacher_scale, "teacher_conf": conf, "teacher_head_w2c": _poses(teacher_scale)[:, :3],
            "frozen_depth": depth * frozen_scale, "frozen_conf": conf, "frozen_head_w2c": _poses(frozen_scale)[:, :3],
            "frozen_head_K": K, "_true_depth": depth}


def test_api_scale_matches_the_released_convention():
    assert api_scale(_poses(2.5)[:, :3], _poses()) == pytest.approx(2.5, rel=1e-5)


def test_e3_targets_are_in_the_gauge_of_the_known_poses():
    c = _cache()
    gt = clip_targets("e3", c, (H, W), (H // 2, W // 2), percentile=0, max_depth=1e9)
    t = lambda a: torch.from_numpy(a)[None]
    ref = normalize_gt(t(c["_true_depth"]), t(np.ones((S, H, W), bool)), t(c["K"]), t(c["w2c"]), (H // 2, W // 2))
    assert torch.allclose(gt["depth"], ref["depth"], rtol=1e-4)      # teacher depth / its API scale = metric depth
    assert torch.allclose(gt["pose_enc"], ref["pose_enc"], atol=1e-5)
    assert torch.allclose(gt["ray"], ref["ray"], atol=1e-5)


def test_pose_only_uses_no_teacher_depth():
    c = _cache()
    a = clip_targets("pose_only", c, (H, W), (H // 2, W // 2), percentile=0, max_depth=1e9)
    c["teacher_depth"] = c["teacher_depth"] * 0 + 99.0                  # teacher changes: pose-only must not see it
    b = clip_targets("pose_only", c, (H, W), (H // 2, W // 2), percentile=0, max_depth=1e9)
    for k in ("depth", "ray", "pose_enc"):
        assert torch.equal(a[k], b[k])
    cfg = arm_loss_config("pose_only")
    assert cfg.w_depth == cfg.alpha_grad == cfg.w_point == 0 and cfg.w_ray == cfg.beta_cam == 1
    with pytest.raises(ValueError):
        arm_loss_config("nope")


def test_valid_mask_percentile_and_range():
    d = np.array([1.0, 2.0, 100.0, np.nan, 5.0, 6.0])
    c = np.array([1.0, 5.0, 5.0, 5.0, 2.0, 4.0])
    m = valid_mask(d, c, percentile=50, max_depth=80)
    assert m.tolist() == [False, True, False, False, False, True]


def test_scale_alignment_of_targets():
    c = _cache()
    gt = clip_targets("e3", c, (H, W), (H // 2, W // 2), percentile=0, max_depth=1e9)
    before = {k: gt[k].clone() for k in ("depth", "ray", "pose_enc")}
    s = align_targets_scale(before["depth"] * 3.0, gt)
    assert s == pytest.approx(3.0, rel=1e-5)
    assert torch.allclose(gt["depth"], before["depth"] * 3)
    assert torch.allclose(gt["ray"][..., :3], before["ray"][..., :3] * 3)   # origins scale
    assert torch.equal(gt["ray"][..., 3:], before["ray"][..., 3:])         # directions do not
    assert torch.allclose(gt["pose_enc"][..., :3], before["pose_enc"][..., :3] * 3)
    assert torch.equal(gt["pose_enc"][..., 3:], before["pose_enc"][..., 3:])


@pytest.mark.parametrize("arm", ["e3", "pose_only", "own_output"])
def test_one_training_step_updates_only_the_adapter(arm):
    m = tiny_model("da3-small", grad_checkpointing=True)
    inject_lora(m, LoRAConfig(rank=4))
    base = {n: p.detach().clone() for n, p in m.named_parameters() if ".lora_" not in n}
    m.train()
    c = _cache()
    x = torch.randn(1, S, 3, H, W)
    with torch.no_grad():
        f = m(x, ref_strategy="first")
    h, w = m.ray_hw(H, W)
    gt = clip_targets(arm, c, (H, W), (h, w))
    opt = torch.optim.AdamW(lora_parameters(m), lr=1e-3)
    pred = m(x, ref_strategy="first")
    align_targets_scale(pred["depth"], gt)
    loss, terms = DA3Loss(arm_loss_config(arm))(pred, gt)
    total = loss + 0.1 * anchor_loss(pred, {k: f[k] for k in ("depth", "ray", "pose_enc")})
    assert torch.isfinite(total)
    total.backward()
    opt.step()
    for n, p in m.named_parameters():
        if ".lora_" not in n:
            assert p.grad is None and torch.equal(p, base[n]), n
    assert any(torch.count_nonzero(p) for n, p in m.named_parameters() if n.endswith("lora_B"))
