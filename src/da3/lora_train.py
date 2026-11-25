"""Training pieces of the LoRA arms (enhancement #3, docs/LORA.md §3.4; recipe in configs/lora_analysis.yaml).

Arms (same clips, steps, rank, seeds; only the targets differ):
* e3         : teacher = released DA3 conditioned on the known poses (E_c); depth targets from the teacher,
               pose / ray targets from the known poses. All five DA3 loss terms.
* pose_only  : no teacher depth; only L_M (ray map) + L_C (camera) against the known poses. Isolates
               E3's novel term (the teacher's depth).
* own_output : every target = the frozen unposed DA3's own outputs (no new information): what the
               training pipeline does by itself.
Every arm adds the anchor λ_a · ℓ1(student, frozen unposed DA3) on depth, ray map and camera encoding
(plan v2: computed on the training clips).

Gauge: the student runs unposed with reference view 0 ("first"), the teacher's reference. Teacher depth
is brought to the units of the input poses with the Umeyama scale between its camera-head poses and
the input poses (as the released API does), then `normalize_gt` expresses everything in view 0's frame
divided by the mean ℓ2 point norm (Sec. 3.3). Because the released training normalisation is not
published, a detached per-clip scale s* = median(D̂ / D) multiplies the scale-carrying targets.
"""

from __future__ import annotations

import numpy as np
import torch

from da3.geometry import ORIG
from da3.losses import LossConfig, normalize_gt

ARMS = ("e3", "pose_only", "own_output")


def arm_loss_config(arm: str, lambda_conf: float = 0.2) -> LossConfig:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
    if arm == "pose_only":
        return LossConfig(w_depth=0.0, alpha_grad=0.0, w_point=0.0, w_ray=1.0, beta_cam=1.0, lambda_conf=lambda_conf)
    return LossConfig(lambda_conf=lambda_conf)


def api_scale(pred_w2c: np.ndarray, input_w2c: np.ndarray) -> float:
    """Scale s with pred ≈ s · input (camera centres), exactly as the released API's alignment
    (`bench_adapter.py`: align_sim3_ransac(pred, input), RANSAC only from 10 views). depth / s is in
    the units of the input poses."""
    from da3.eval.metrics import align_sim3_ransac

    _, _, s, _ = align_sim3_ransac(pred_w2c, input_w2c, ransac=len(input_w2c) >= 10, seed=42)
    return float(s)


def valid_mask(depth: np.ndarray, conf: np.ndarray, percentile: float, max_depth: float) -> np.ndarray:
    """Pixels used as targets: finite, positive, <= max_depth, confidence >= the clip's percentile."""
    ok = np.isfinite(depth) & (depth > 0) & (depth <= max_depth)
    if ok.any() and percentile > 0:
        ok &= conf >= np.percentile(conf[ok], percentile)
    return ok


def clip_targets(arm: str, cache: dict, hw, ray_hw, percentile: float = 30.0, max_depth: float = 80.0) -> dict:
    """Normalised targets of one cached clip for one arm (numpy cache -> torch dict, batch dim 1).

    cache keys: K, w2c (known poses, metres, view 0 = identity); teacher_depth/conf/head_w2c (posed pass);
    frozen_depth/conf/head_w2c/head_K (unposed released pass).
    """
    if arm == "own_output":
        depth, conf = cache["frozen_depth"], cache["frozen_conf"]          # own gauge: no rescaling
        K, w2c = cache["frozen_head_K"], cache["frozen_head_w2c"]
    elif arm == "e3":
        depth = cache["teacher_depth"] / api_scale(cache["teacher_head_w2c"], cache["w2c"])
        conf, K, w2c = cache["teacher_conf"], cache["K"], cache["w2c"]
    else:  # pose_only: depth only sets the scale normalisation (own output, brought to the pose units)
        depth = cache["frozen_depth"] / api_scale(cache["frozen_head_w2c"], cache["w2c"])
        conf, K, w2c = cache["frozen_conf"], cache["K"], cache["w2c"]
    depth = depth.astype(np.float32)
    valid = valid_mask(depth, conf.astype(np.float32), percentile, max_depth if arm != "own_output" else np.inf)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None]
    w2c44 = np.tile(np.eye(4, dtype=np.float32), (len(w2c), 1, 1))
    w2c44[:, :3, :] = np.asarray(w2c, np.float32)[:, :3, :]
    return normalize_gt(t(depth), t(valid), t(np.asarray(K, np.float32)), t(w2c44), ray_hw)


@torch.no_grad()
def align_targets_scale(pred_depth: torch.Tensor, gt: dict) -> float:
    """Detached per-clip s* = median(D̂ / D) on valid pixels; scales the targets that carry scale
    (depth, points, ray origins, translations) in place. Returns s*."""
    v = gt["valid"]
    if v.sum() == 0:
        return 1.0
    s = float(torch.median(pred_depth.detach().float()[v] / gt["depth"][v]).clamp(1e-3, 1e3))
    gt["depth"] = gt["depth"] * s
    gt["points"] = gt["points"] * s
    gt["ray"] = gt["ray"].clone()
    gt["ray"][..., ORIG] = gt["ray"][..., ORIG] * s
    gt["pose_enc"] = gt["pose_enc"].clone()
    gt["pose_enc"][..., :3] = gt["pose_enc"][..., :3] * s
    return s


def anchor_loss(pred: dict, frozen: dict) -> torch.Tensor:
    """ℓ1 to the frozen released outputs (same unposed gauge): depth, ray map, camera encoding."""
    d = (pred["depth"].float() - frozen["depth"]).abs().mean()
    r = (pred["ray"].float() - frozen["ray"]).abs().sum(-1).mean()
    c = (pred["pose_enc"].float() - frozen["pose_enc"]).abs().sum(-1).mean()
    return d + r + c


def to_device(gt: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in gt.items()}
