import pytest
import torch

from da3.backbone.vit import (
    from_cross_view,
    from_within_view,
    gather_views,
    reference_permutation,
    to_cross_view,
    to_within_view,
)
from helpers import tiny_model

H, W = 70, 98  # 5 x 7 patches
BOTH = ["paper-small", "da3-small"]


def test_token_reordering_preserves_views():
    B, S, N, C = 2, 4, 6, 8
    x = torch.randn(B, S, N, C)
    xc = to_cross_view(x)
    assert xc.shape == (B, S * N, C)
    for s in range(S):  # view s occupies a contiguous block of the joint sequence
        assert torch.equal(xc[:, s * N : (s + 1) * N], x[:, s])
    assert torch.equal(from_cross_view(xc, B, S), x)
    xw = to_within_view(x)
    assert xw.shape == (B * S, N, C) and torch.equal(xw[1 * S + 2], x[1, 2])
    assert torch.equal(from_within_view(xw, B, S), x)
    # released reference-view permutation and its inverse
    perm = reference_permutation(torch.tensor([2, 0]), S)
    assert perm.tolist() == [[2, 0, 1, 3], [0, 1, 2, 3]]
    y = gather_views(x, perm)
    assert torch.equal(y[0, 0], x[0, 2]) and torch.equal(y[0, 1], x[0, 0])
    assert torch.equal(gather_views(y, torch.argsort(perm, dim=1)), x)


def test_paper_layout():
    """Paper: L_s = 8 within-view blocks, then L_g = 4 alternating cross / within (2 : 1)."""
    m = tiny_model("paper-small")
    kinds = ["X" if m.backbone.is_cross_view(i) else "W" for i in range(12)]
    assert "".join(kinds) == "WWWWWWWWXWXW"
    assert m.backbone.camera_token.shape[1] == 2  # [reference view, shared by all other views]
    assert m.backbone.n_special == 2               # [camera token, cls] prepended to the patches


@pytest.mark.parametrize("preset", BOTH)
def test_output_shapes_single_and_multi_view(preset):
    m = tiny_model(preset)
    for S in (1, 2, 5):
        with torch.no_grad():
            out = m(torch.randn(2, S, 3, H, W))
        assert out["depth"].shape == (2, S, H, W) and out["depth_conf"].shape == (2, S, H, W)
        h, w = m.ray_hw(H, W)
        assert out["ray"].shape == (2, S, h, w, 6)
        assert out["pose_enc"].shape == (2, S, 9)
        assert (out["depth"] > 0).all() and (out["depth_conf"] > 1).all()
    if preset == "paper-small":
        assert (h, w) == (H // 2, W // 2) and "ray_conf" not in out  # paper: ray at H/2 x W/2


@pytest.mark.parametrize("preset", BOTH)
def test_single_view_reduces_to_monocular(preset):
    """With N_v = 1 the cross-view token layout equals the within-view one (no extra tokens or
    cost), and single-image scenes stacked in a batch do not interact."""
    x = torch.randn(3, 1, 7, 4)
    assert torch.equal(to_cross_view(x), to_within_view(x))
    m = tiny_model(preset)
    imgs = torch.randn(3, 1, 3, H, W)
    with torch.no_grad():
        batched = m(imgs)
        for b in range(3):
            single = m(imgs[b : b + 1])
            for k in ("depth", "ray", "pose_enc"):
                assert torch.allclose(batched[k][b], single[k][0], atol=1e-5), k


@pytest.mark.parametrize("preset", BOTH)
def test_cross_view_is_used_for_multiple_views(preset):
    """For S > 1, changing view 1 must change the prediction of view 0 (information flows)."""
    m = tiny_model(preset)
    x = torch.randn(1, 2, 3, H, W)
    x2 = x.clone()
    x2[:, 1] = torch.randn_like(x2[:, 1])
    with torch.no_grad():
        assert not torch.allclose(m(x)["depth"][:, 0], m(x2)["depth"][:, 0])


@pytest.mark.parametrize("preset,perm", [("paper-small", [0, 2, 3, 1]), ("da3-small", [0, 3, 1, 2])])
def test_permutation_equivariance(preset, perm):
    """Per-view content is preserved by the token reordering: permuting the non-reference views
    (1..S-1, which share a camera token) permutes the outputs accordingly."""
    m = tiny_model(preset)
    x = torch.randn(1, 4, 3, H, W)
    perm = torch.tensor(perm)
    with torch.no_grad():
        a, b = m(x), m(x[:, perm])
    for k in ("depth", "depth_conf", "ray", "pose_enc"):
        assert torch.allclose(a[k][:, perm], b[k], atol=1e-4, rtol=1e-4), k


def test_released_reference_view_reordering_restores_order():
    m = tiny_model("da3-small")
    x = torch.randn(1, 4, 3, H, W)
    with torch.no_grad():
        a = m(x, ref_strategy="middle")  # reference = view 2
        perm = torch.tensor([2, 0, 1, 3])
        b = m(x[:, perm], ref_strategy="first")
    assert a["ref_index"].tolist() == [2]
    assert torch.allclose(a["depth"], b["depth"][:, torch.argsort(perm)], atol=1e-4, rtol=1e-4)


def test_pose_conditioning_and_grad_checkpointing():
    from da3.camera import affine_inverse
    from helpers import random_camera

    m = tiny_model(grad_checkpointing=True).train()
    K, c2w = random_camera(B=3, H=H, W=W, dtype=torch.float32)
    w2c = affine_inverse(c2w)[None]
    x = torch.randn(1, 3, 3, H, W)
    out = m(x, w2c=w2c, K=K[None])
    out["depth"].mean().backward()
    assert m.cam_enc.pose_branch.fc1.weight.grad is not None
    assert m.backbone.camera_token.grad is None  # known poses replace the learnable token
    m.eval()
    with torch.no_grad():
        a = m(x, w2c=w2c, K=K[None])["depth"]
        m.backbone.grad_checkpointing = False
        b = m(x, w2c=w2c, K=K[None])["depth"]
    assert torch.allclose(a, b)
