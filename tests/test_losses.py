import torch

from da3.camera import affine_inverse
from da3.losses import DA3Loss, LossConfig, normalize_gt
from helpers import random_camera

H, W = 56, 84


def _scene(S=3):
    K, c2w = random_camera(B=S, H=H, W=W, dtype=torch.float32)
    depth = 1 + 4 * torch.rand(1, S, H, W)
    valid = torch.rand(1, S, H, W) > 0.1
    return depth, valid, K[None], affine_inverse(c2w)[None]


def test_gt_normalization():
    depth, valid, K, w2c = _scene()
    gt = normalize_gt(depth, valid, K, w2c, (32, 48))
    # view 0 is the identity, mean point norm is 1
    assert torch.allclose(gt["w2c"][0, 0], torch.eye(4), atol=1e-5)
    n = gt["points"].norm(dim=-1)[gt["valid"]].mean()
    assert torch.allclose(n, torch.tensor(1.0), atol=1e-5)
    # scaling the scene does not change the normalized targets
    w2c2 = w2c.clone()
    w2c2[..., :3, 3] *= 7.0
    gt2 = normalize_gt(depth * 7.0, valid, K, w2c2, (32, 48))
    assert torch.allclose(gt["depth"], gt2["depth"], atol=1e-4)
    assert torch.allclose(gt["ray"], gt2["ray"], atol=1e-4)


def test_loss_minimal_at_ground_truth():
    depth, valid, K, w2c = _scene()
    gt = normalize_gt(depth, valid, K, w2c, (32, 48))
    conf = torch.ones(1, 3, H, W)
    perfect = {"depth": gt["depth"], "depth_conf": conf, "ray": gt["ray"],
               "ray_conf": torch.ones(1, 3, 32, 48), "pose_enc": gt["pose_enc"]}
    loss_fn = DA3Loss(LossConfig(point_rays="pred"))
    total, terms = loss_fn(perfect, gt)
    for k in ("depth", "ray", "point", "grad", "cam"):
        assert terms[k].abs() < 1e-4, (k, terms[k])
    noisy = dict(perfect, depth=gt["depth"] * 1.1)
    assert loss_fn(noisy, gt)[0] > total


def test_loss_backward_and_invalid_pixels():
    depth, valid, K, w2c = _scene()
    depth[valid.logical_not()] = float("nan")  # invalid GT must not produce NaNs
    gt = normalize_gt(depth, valid, K, w2c, (32, 48))
    pred = {
        "depth": torch.rand(1, 3, H, W, requires_grad=True),
        "depth_conf": 1 + torch.rand(1, 3, H, W),
        "ray": torch.randn(1, 3, 32, 48, 6, requires_grad=True),
        "ray_conf": 1 + torch.rand(1, 3, 32, 48),
        "pose_enc": torch.randn(1, 3, 9, requires_grad=True),
    }
    total, _ = DA3Loss()(pred, gt)
    assert torch.isfinite(total)
    total.backward()
    assert torch.isfinite(pred["depth"].grad).all() and torch.isfinite(pred["ray"].grad).all()
