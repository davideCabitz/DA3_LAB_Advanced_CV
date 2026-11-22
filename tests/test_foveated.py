"""Enhancement #2 (Foveated DA3): mixed-resolution routing, balancing, plans, zoom geometry, adapter."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from da3.camera import affine_inverse
from da3.enhancements.foveated.plan import (
    FoveationSpec,
    backbone_cost,
    equal_budget_res,
    patch_tokens,
    processed_hw,
    select_focus,
    subset_with_focus,
)
from da3.enhancements.foveated.routing import (
    cameras_from_head,
    foveated_forward,
    mass_balance_bias,
    sdpa_key_bias,
)
from helpers import random_camera, tiny_model

H, W = 70, 98          # 5 x 7 patches
HC, WC = 42, 56        # 3 x 4 patches (context views)
BOTH = ["paper-small", "da3-small"]
MAPS = ("depth", "depth_conf", "ray", "ray_conf")


def _stack(out, key):
    return torch.stack(out[key])


def _assert_same_as_original(ref, fov, exact=True, atol=1e-5):
    for k in MAPS:
        if k not in ref:
            continue
        a, b = ref[k][0], _stack(fov, k)
        if exact:
            assert torch.equal(a, b), (k, (a - b).abs().max().item())
        else:
            assert torch.allclose(a, b, atol=atol, rtol=atol), (k, (a - b).abs().max().item())
    a, b = ref["pose_enc"][0], fov["pose_enc"]
    assert (torch.equal(a, b) if exact else torch.allclose(a, b, atol=atol, rtol=atol)), (a - b).abs().max().item()


# ----------------------------------------------------------------------------- routing = DA3.forward
@pytest.mark.parametrize("preset", BOTH)
@pytest.mark.parametrize("S", [1, 2, 4])
def test_uniform_resolution_is_bit_identical(preset, S):
    m = tiny_model(preset)
    x = torch.randn(1, S, 3, H, W)
    with torch.no_grad():
        ref = m(x)
    fov = foveated_forward(m, list(x[0]))
    _assert_same_as_original(ref, fov, exact=True)


@pytest.mark.parametrize("strategy", ["saddle_balanced", "middle"])
def test_uniform_resolution_reference_selection_is_bit_identical(strategy):
    m = tiny_model("da3-small")
    x = torch.randn(1, 5, 3, H, W)
    with torch.no_grad():
        ref = m(x, ref_strategy=strategy)
    fov = foveated_forward(m, list(x[0]), ref_strategy=strategy)
    assert fov["ref_index"] == int(ref["ref_index"][0])
    _assert_same_as_original(ref, fov, exact=True)


@pytest.mark.parametrize("preset", BOTH)
def test_uniform_resolution_posed_is_bit_identical(preset):
    m = tiny_model(preset)
    S = 4
    K, c2w = random_camera(B=S, H=H, W=W, dtype=torch.float32)
    w2c = affine_inverse(c2w)
    x = torch.randn(1, S, 3, H, W)
    with torch.no_grad():
        ref = m(x, w2c=w2c[None], K=K[None])
    fov = foveated_forward(m, list(x[0]), w2c=w2c, K=K)
    _assert_same_as_original(ref, fov, exact=True)


def test_two_groups_of_equal_resolution_match_one_group():
    """Same-size views routed through two separate groups (split within-view batches, concatenated
    cross-view sequence) give the same result up to float round-off."""
    m = tiny_model("da3-small")
    x = torch.randn(1, 5, 3, H, W)
    with torch.no_grad():
        ref = m(x, ref_strategy="saddle_balanced")
    fov = foveated_forward(m, list(x[0]), ref_strategy="saddle_balanced", group_ids=[0, 1, 0, 1, 1])
    assert fov["ref_index"] == int(ref["ref_index"][0])
    _assert_same_as_original(ref, fov, exact=False, atol=1e-4)


# ----------------------------------------------------------------------------- mixed resolutions
def _mixed_views(seed=0):
    g = torch.Generator().manual_seed(seed)
    sizes = [(H, W), (HC, WC), (H, W), (HC, WC), (HC, WC)]
    return [torch.randn(3, h, w, generator=g) for h, w in sizes]


@pytest.mark.parametrize("preset", BOTH)
def test_mixed_resolution_shapes(preset):
    m = tiny_model(preset)
    views = _mixed_views()
    out = foveated_forward(m, views)
    assert out["pose_enc"].shape == (5, 9)
    assert len(out["groups"]) == 2 and out["groups"].members == [[0, 2], [1, 3, 4]]
    for s, v in enumerate(views):
        h, w = v.shape[-2:]
        assert out["depth"][s].shape == (h, w) and (out["depth"][s] > 0).all()
        assert out["ray"][s].shape == (*m.ray_hw(h, w), 6)
        assert torch.isfinite(out["pose_enc"][s]).all()
    w2c, K = cameras_from_head(out["pose_enc"], out["hw"])
    assert w2c.shape == (5, 3, 4) and K.shape == (5, 3, 3)
    assert K[1, 0, 2] == WC / 2 and K[0, 0, 2] == W / 2      # principal point in each view's own pixels


def test_mixed_resolution_views_interact_across_groups():
    """Cross-view attention spans both resolutions: changing a context view changes a focus view."""
    m = tiny_model("da3-small")
    views = _mixed_views()
    a = foveated_forward(m, views)
    views2 = list(views)
    views2[3] = torch.randn_like(views[3])
    b = foveated_forward(m, views2)
    assert not torch.allclose(a["depth"][0], b["depth"][0])


@pytest.mark.parametrize("balance", [False, True])
def test_mixed_resolution_permutation_equivariance(balance):
    """Non-reference views share a camera token and cross-view RoPE positions are constant, so the
    order of views (inside and across resolution groups) only permutes the outputs."""
    m = tiny_model("da3-small")
    views = _mixed_views()
    perm = [0, 4, 3, 2, 1]                      # keep the reference (view 0) first
    a = foveated_forward(m, views, balance=balance)
    b = foveated_forward(m, [views[i] for i in perm], balance=balance)
    for k in ("depth", "ray"):
        for j, i in enumerate(perm):
            assert torch.allclose(a[k][i], b[k][j], atol=1e-4, rtol=1e-4), (k, i)
    assert torch.allclose(a["pose_enc"][perm], b["pose_enc"], atol=1e-4, rtol=1e-4)


def test_reference_candidates_restrict_the_reference_view():
    m = tiny_model("da3-small")
    views = _mixed_views()
    out = foveated_forward(m, views, ref_strategy="saddle_balanced", ref_candidates=[0, 2])
    assert out["ref_index"] in (0, 2)
    out = foveated_forward(m, views, ref_strategy="first", ref_candidates=[2, 0])
    assert out["ref_index"] == 0


def test_balance_is_a_noop_at_equal_resolution():
    m = tiny_model("da3-small")
    x = torch.randn(1, 4, 3, H, W)
    a = foveated_forward(m, list(x[0]))
    b = foveated_forward(m, list(x[0]), balance=True)
    for k in MAPS:
        assert torch.equal(_stack(a, k), _stack(b, k))


def test_mixed_resolution_posed_runs_and_uses_per_view_fov():
    """E_c with views of different sizes: the same camera at a scaled resolution gives the same token."""
    from da3.enhancements.foveated.routing import camera_tokens

    m = tiny_model("da3-small")
    K, c2w = random_camera(B=3, H=H, W=W, dtype=torch.float32)
    w2c = affine_inverse(c2w)
    K_small = K.clone()
    K_small[:, 0] *= WC / W
    K_small[:, 1] *= HC / H
    t_full = camera_tokens(m, w2c, K, [(H, W)] * 3)
    t_mix = camera_tokens(m, w2c, torch.stack([K[0], K_small[1], K_small[2]]), [(H, W), (HC, WC), (HC, WC)])
    assert torch.allclose(t_full, t_mix, atol=1e-5)
    views = [torch.randn(3, H, W), torch.randn(3, HC, WC), torch.randn(3, HC, WC)]
    out = foveated_forward(m, views, w2c=w2c, K=torch.stack([K[0], K_small[1], K_small[2]]))
    assert all(torch.isfinite(d).all() for d in out["depth"])


def test_offload_and_keep():
    m = tiny_model("da3-small")
    out = foveated_forward(m, _mixed_views(), keep=("depth",), offload=True)
    assert set(k for k in MAPS if k in out) == {"depth"}
    assert all(d.device.type == "cpu" for d in out["depth"])


# ----------------------------------------------------------------------------- balanced attention
def test_key_bias_equals_token_replication():
    """Adding log(r) to the logits of a key = replicating that key (and its value) r times."""
    torch.manual_seed(0)
    B, h, d = 1, 2, 16
    q = torch.randn(B, h, 7, d)
    k_a, v_a = torch.randn(B, h, 4, d), torch.randn(B, h, 4, d)   # "focus" view: 4 tokens
    k_b, v_b = torch.randn(B, h, 2, d), torch.randn(B, h, 2, d)   # "context" view: 2 tokens, r = 2
    k = torch.cat([k_a, k_b], 2)
    v = torch.cat([v_a, v_b], 2)
    bias = torch.tensor([0, 0, 0, 0, math.log(2), math.log(2)])
    out = sdpa_key_bias(q, k, v, bias)
    k_rep = torch.cat([k_a, k_b.repeat_interleave(2, dim=2)], 2)
    v_rep = torch.cat([v_a, v_b.repeat_interleave(2, dim=2)], 2)
    ref = F.scaled_dot_product_attention(q, k_rep, v_rep)
    assert torch.allclose(out, ref, atol=1e-5)
    assert torch.allclose(sdpa_key_bias(q, k, v, torch.zeros(6)), F.scaled_dot_product_attention(q, k, v), atol=1e-5)


def test_mass_balance_bias_layout():
    b = mass_balance_bias([1 + 12, 1 + 3], [2, 3], n_special=1, device="cpu")
    assert b.shape == (2 * 13 + 3 * 4,)
    assert torch.all(b[:26] == 0)
    ctx = b[26:].reshape(3, 4)
    assert torch.all(ctx[:, 0] == 0) and torch.allclose(ctx[:, 1:], torch.full((3, 3), math.log(4.0)))


# ----------------------------------------------------------------------------- plans and budgets
def test_select_focus():
    m = select_focus(16, 0.5)
    assert m.sum() == 8 and m[0] and m[-1]
    assert select_focus(100, 0.1).sum() == 10 and select_focus(100, 0.1)[0]
    r1, r2 = select_focus(50, 0.2, strategy="random", seed=3), select_focus(50, 0.2, strategy="random", seed=3)
    assert np.array_equal(r1, r2) and r1[0] and r1.sum() == 10
    assert select_focus(5, 0.01).sum() == 1 and select_focus(5, 1.0).all()
    assert np.array_equal(np.flatnonzero(select_focus(10, count=3, strategy="first")), [0, 1, 2])
    sub = subset_with_focus(20, select_focus(20, 0.1), 6)
    assert sub.sum() == 6 and np.all(sub[select_focus(20, 0.1)])


def test_spec_parse_and_name():
    s = FoveationSpec.parse("fraction=0.25, context_res=280, balance=1")
    assert (s.fraction, s.focus_res, s.context_res, s.balance) == (0.25, 504, 280, True)
    assert s.name == "fov25_504-280_bal"
    assert FoveationSpec.parse("fraction=0.1,strategy=random,seed=2").name == "fov10_504-252_random2"
    with pytest.raises(ValueError):
        FoveationSpec.parse("fraction=0.25,foo=1")


def test_budget_numbers_for_7scenes():
    """640 x 480 frames: 504 -> 36 x 27 patches, 252 -> 18 x 14, 336 -> 24 x 18. With 25% focus views at
    504 and the rest at 252 the token budget equals uniform 336 px."""
    assert processed_hw((480, 640), 504) == (378, 504) and patch_tokens((378, 504)) == 972
    assert processed_hw((480, 640), 252) == (196, 252) and patch_tokens((196, 252)) == 252
    assert patch_tokens(processed_hw((480, 640), 336)) == 432
    fov = 25 * 972 + 75 * 252
    assert fov == 100 * 432
    assert equal_budget_res((480, 640), 100, fov) == 336
    c_fov = backbone_cost([(378, 504)] * 25 + [(196, 252)] * 75)
    c_all = backbone_cost([(378, 504)] * 100)
    assert c_fov["tokens"] == fov and c_fov["gflops"] < c_all["gflops"] / 3


# ----------------------------------------------------------------------------- zoom geometry (H4)
K0 = np.array([[1000.0, 0.0, 600.0], [0.0, 1000.0, 400.0], [0.0, 0.0, 1.0]])   # 1200 x 800 source


def test_zoom_view_of_the_centre_is_a_plain_magnification():
    from da3.enhancements.foveated.zoom import zoom_view

    zv = zoom_view(K0, (450, 300, 300, 200), (200, 300))
    assert np.allclose(zv.R_v, np.eye(3), atol=1e-12)
    assert np.isclose(zv.K[0, 0], 1000.0) and np.allclose(zv.K[:2, 2], [150, 100])


def test_zoom_warp_is_exact_for_an_off_centre_region():
    """Source intensity = smooth function of the viewing ray; the zoom render must reproduce it."""
    from da3.enhancements.foveated.zoom import render_zoom, zoom_to_source, zoom_view

    def pattern(xy):
        u, v = (xy[..., 0] - 600) / 1000, (xy[..., 1] - 400) / 1000
        return 128 + 90 * np.sin(9 * u) * np.cos(7 * v)

    yy, xx = np.meshgrid(np.arange(800) + 0.5, np.arange(1200) + 0.5, indexing="ij")
    img = np.repeat(pattern(np.stack([xx, yy], -1))[..., None], 3, -1).astype(np.uint8)
    zv = zoom_view(K0, (840, 80, 300, 200), (100, 150))                # off-centre: R_v != I
    assert np.degrees(np.arccos(zv.R_v[2, 2])) > 10
    z = render_zoom(img, K0, zv).astype(np.float64)[..., 0]
    truth = pattern(zoom_to_source(zv, K0)).reshape(100, 150)
    assert np.abs(z - truth)[5:-5, 5:-5].mean() < 2.0


def test_zoom_depth_maps_back_to_source_depth():
    """A plane at source depth 5: its zoom-view depth (5 / d_z) maps back to 5 at every box point."""
    from da3.enhancements.foveated.zoom import box_points, depth_from_zoom, zoom_view

    box = (840, 80, 300, 200)
    zv = zoom_view(K0, box, (100, 150))
    uu, vv = np.meshgrid(np.arange(150) + 0.5, np.arange(100) + 0.5)
    rz = np.stack([uu, vv, np.ones_like(uu)], -1) @ np.linalg.inv(zv.K).T
    d = rz @ zv.R_v.T                                                    # source-frame rays
    z_zoom = 5.0 / d[..., 2]
    z_src = depth_from_zoom(z_zoom, zv, K0, box_points(box, (100, 150)))
    assert np.nanmax(np.abs(z_src - 5.0)) < 1e-3 and np.isfinite(z_src).mean() > 0.99


def test_crop_and_global_depth_lookups_follow_the_pixel_grid():
    from da3.enhancements.foveated.zoom import box_points, depth_from_crop, depth_from_global

    box = (300, 200, 600, 400)
    pts = box_points(box, (40, 60))
    hc, wc = 20, 30                                                     # crop prediction size
    xs_src = box[0] + (np.arange(wc) + 0.5) * box[2] / wc
    crop_depth = np.tile(1.0 + 0.01 * xs_src, (hc, 1))
    z = depth_from_crop(crop_depth, box, pts)
    inner = (pts[:, 0] > xs_src[0]) & (pts[:, 0] < xs_src[-1])
    assert np.allclose(z[inner], 1.0 + 0.01 * pts[inner, 0])
    hg, wg = 80, 120                                                    # global prediction of the 800 x 1200 image
    xs_g = (np.arange(wg) + 0.5) * 1200 / wg
    glob = np.tile(2.0 + 0.001 * xs_g, (hg, 1))
    zg = depth_from_global(glob, (800, 1200), pts)
    assert np.allclose(zg, 2.0 + 0.001 * pts[:, 0])


def test_zoom_extrinsics_rotate_the_source_camera():
    from da3.enhancements.foveated.zoom import zoom_view, zoom_w2c

    zv = zoom_view(K0, (840, 80, 300, 200), (100, 150))
    K, c2w = random_camera(B=1, H=800, W=1200)
    w2c = affine_inverse(c2w)[0].numpy()
    X = np.array([0.3, -0.2, 4.0, 1.0])
    Xs = w2c @ X
    Xz = zoom_w2c(w2c, zv.R_v) @ X
    assert np.allclose(zv.R_v @ Xz[:3], Xs[:3])


def test_select_regions_prefers_geometric_detail():
    from da3.enhancements.foveated.zoom import select_regions

    rng = np.random.default_rng(0)
    gt = np.full((400, 600), 5.0, np.float32)
    gt[250:400, 0:150] = 5.0 + rng.random((150, 150)).astype(np.float32)   # detail bottom-left
    gt[:60] = 0.0                                                          # invalid band
    (x0, y0, w, h), = select_regions(gt, zoom=4, n=1, coarse=4)
    assert (w, h) == (150, 100) and x0 < 75 and y0 >= 200


# ----------------------------------------------------------------------------- metrics
def test_depth_and_pose_metrics():
    from da3.enhancements.foveated.metrics import depth_errors, pairwise_pose_errors, view_subset_aucs

    gt = np.random.default_rng(0).uniform(1, 5, (30, 40)).astype(np.float32)
    e = depth_errors(2 * gt, gt)
    assert e["abs_rel"] < 1e-6 and np.isclose(e["scale"], 0.5) and e["delta1"] == 1.0
    assert depth_errors(gt, np.zeros_like(gt)) is None
    K, c2w = random_camera(B=6, H=48, W=64)
    w2c = affine_inverse(c2w).numpy()
    E = pairwise_pose_errors(w2c, w2c)
    aucs = view_subset_aucs(E, select_focus(6, 0.5))
    assert aucs["all"]["pairs"] == 15 and aucs["focus"]["pairs"] == 3 and aucs["cross"]["pairs"] == 9
    assert all(np.isclose(v["auc30"], 1.0) for v in aucs.values())


# ----------------------------------------------------------------------------- harness adapter
def _write_images(folder, n=4, h=120, w=160):
    import cv2

    rng = np.random.default_rng(0)
    files = []
    for i in range(n):
        p = folder / f"{i:03d}.png"
        cv2.imwrite(str(p), rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
        files.append(str(p))
    return files


def test_adapter_with_all_focus_views_equals_the_original_adapter(tmp_path):
    from da3.enhancements.foveated.adapter import FoveatedAPI, MeteredAPI
    from da3.eval.bench_adapter import OursAsOfficialAPI

    files = _write_images(tmp_path)
    m = tiny_model("da3-small")
    ref = OursAsOfficialAPI(m, "cpu", process_res=98).predict(files)
    fov = FoveatedAPI(m, "cpu", spec=FoveationSpec(fraction=1.0, focus_res=98, context_res=56)).predict(files)
    met = MeteredAPI(m, "cpu", process_res=98).predict(files)
    for k in ("depth", "conf", "extrinsics", "intrinsics"):
        assert np.array_equal(ref[k], fov[k]), k
        assert np.array_equal(ref[k], met[k]), k


def test_adapter_mixed_resolution_export_and_meta(tmp_path):
    from da3.enhancements.foveated.adapter import META_FILE, FoveatedAPI

    files = _write_images(tmp_path)
    m = tiny_model("da3-small")
    api = FoveatedAPI(m, "cpu", spec=FoveationSpec(fraction=0.5, focus_res=98, context_res=56))
    pred = api.inference(files, export_dir=str(tmp_path / "exp"))
    assert pred["depth"].shape == (4, 70, 98) and pred["conf"].shape == (4, 70, 98)
    assert np.allclose(pred["intrinsics"][:, :2, 2], [49.0, 35.0])        # every view at the export size
    meta = np.load(tmp_path / "exp" / "exports" / META_FILE)
    assert meta["focus"].tolist() == [True, False, False, True]
    assert meta["view_hw"].tolist() == [[70, 98], [42, 56], [42, 56], [70, 98]]
    assert int(meta["tokens"]) == 2 * 35 + 2 * 12
    assert (tmp_path / "exp" / "exports" / "mini_npz" / "results.npz").exists()
    assert api.summary()["scenes"] == 1


# ----------------------------------------------------------------------------- GT depth decoders
def test_gt_depth_decoders(tmp_path):
    import cv2

    from da3.eval.gt_depth import eth3d_gt_paths, read_gt_depth

    d = np.array([[1.0, 2.0], [0.0, 3.5]])
    cv2.imwrite(str(tmp_path / "h.png"), (d / 100 * 65535).astype(np.uint16))
    cv2.imwrite(str(tmp_path / "m.png"), np.array([[0, 255], [0, 0]], np.uint8))
    h = read_gt_depth("hiroom", str(tmp_path / "h.png"), str(tmp_path / "m.png"))
    assert np.allclose(h, [[1.0, 0.0], [0.0, 3.5]], atol=2e-3)                 # aliasing mask removes (0, 1)
    cv2.imwrite(str(tmp_path / "s.png"), np.array([[1500, 65535], [0, 800]], np.uint16))
    s = read_gt_depth("7scenes", str(tmp_path / "s.png"), hw=(4, 4))
    assert s.shape == (4, 4) and np.isclose(s[0, 0], 1.5) and s[0, 3] == 0 and np.isclose(s[3, 3], 0.8)
    scene = tmp_path / "eth3d" / "pipes"
    img = scene / "images" / "dslr_images" / "DSC_0001.JPG"
    dp, mp = eth3d_gt_paths(str(img))
    assert dp.replace("\\", "/").endswith("pipes/ground_truth_depth/dslr_images/DSC_0001.JPG")
    assert mp.replace("\\", "/").endswith("pipes/masks_for_images/dslr_images/DSC_0001.png")
    Path_ = type(tmp_path)
    Path_(dp).parent.mkdir(parents=True)
    Path_(mp).parent.mkdir(parents=True)
    np.array([[2.0, np.inf], [4.0, 5.0]], np.float32).tofile(dp)
    cv2.imwrite(mp, np.array([[0, 0], [1, 0]], np.uint8))
    e = read_gt_depth("eth3d", dp, mp, shape=(2, 2))
    assert np.allclose(e, [[2.0, 0.0], [0.0, 5.0]])                            # inf and mask == 1 -> invalid
    assert read_gt_depth("eth3d", str(tmp_path / "missing"), None, shape=(2, 2)).sum() == 0


def test_deployable_region_selectors():
    from da3.enhancements.foveated.zoom import select_regions

    rng = np.random.default_rng(0)
    gt = np.full((400, 600), 5.0, np.float32)
    gt[:60] = 0.0                                                          # not evaluable
    image = np.zeros((400, 600, 3), np.uint8)
    image[300:400, 450:600] = rng.integers(0, 255, (100, 150, 3))          # texture bottom-right
    (x0, y0, w, h), = select_regions(gt, zoom=4, n=1, coarse=4, mode="texture", image=image)
    assert x0 >= 375 and y0 >= 225                                         # the textured corner
    a = select_regions(gt, zoom=4, n=3, coarse=4, mode="random", seed=7)
    assert a == select_regions(gt, zoom=4, n=3, coarse=4, mode="random", seed=7) and len(a) == 3
    assert all(b[1] >= 50 for b in a)                                      # never the invalid band
    with pytest.raises(ValueError):
        select_regions(gt, mode="texture")
