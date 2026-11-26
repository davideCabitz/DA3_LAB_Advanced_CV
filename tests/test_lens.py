"""Enhancement #4 (lens-consistent DA3): exact geometry, plans, adapter parity, metrics, records."""

import numpy as np
import pytest

from da3.enhancements.lens import (
    KIND_NORMAL,
    KIND_PORTRAIT,
    KIND_ZOOM,
    AxisAffine,
    LensSpec,
    assign_kinds,
    canonical_resolutions,
    captures_for,
    kappa_from_captures,
    portrait_capture,
    transform_K,
    warp_to,
    zoom_capture,
)
from helpers import tiny_model


def _K(f=500.0, w=640, h=480):
    return np.array([[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]])


def _project(K, X):
    x = X @ K.T
    return x[:, 0] / x[:, 2], x[:, 1] / x[:, 2]


# ----------------------------------------------------------------------------- geometry
def test_affine_compose_and_inverse():
    a = AxisAffine(2.0, 3.0, 0.5, -1.0)
    b = AxisAffine.crop(10, 20).then(AxisAffine.resize((100, 200), (50, 400)))
    c = a.then(b)
    u, v = np.array([1.0, 7.5]), np.array([2.0, -3.0])
    assert np.allclose(c.apply(u, v), b.apply(*a.apply(u, v)))
    assert np.allclose(c.inverse().apply(*c.apply(u, v)), (u, v))
    assert a.then(a.inverse()).is_identity()
    assert np.allclose(c.matrix() @ np.array([1.0, 2.0, 1.0]), [*c.apply(1.0, 2.0), 1.0])


@pytest.mark.parametrize("z", [1.5, 2.0, 3.0])
def test_zoom_capture_intrinsics_are_exact(z):
    rng = np.random.default_rng(0)
    K = _K()
    cap = zoom_capture((480, 640), z)
    X = np.c_[rng.uniform(-0.3, 0.3, (50, 2)), rng.uniform(2.0, 5.0, 50)]
    u, v = _project(K, X)
    uz, vz = cap.affine().apply(u, v)                # where each point lands in the zoomed photo
    Kz = transform_K(K, cap.affine())
    assert np.allclose((uz, vz), _project(Kz, X))
    assert np.isclose(Kz[0, 0] / K[0, 0], cap.zoom, rtol=1e-12) and abs(cap.zoom - z) < 0.01
    assert np.allclose(Kz[:2, 2], [320.0, 240.0])     # exactly centred crop: principal point unchanged
    assert cap.size == (480, 640) and cap.kind == KIND_ZOOM


def test_capture_apply_sizes_and_identity():
    img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    z = zoom_capture((480, 640), 2.0)
    assert z.box == (160, 120, 320, 240) and z.apply(img).shape == (480, 640, 3)
    p = portrait_capture((480, 640))
    assert p.box == (140, 0, 360, 480) and p.apply(img).shape == (480, 360, 3) and p.kind == KIND_PORTRAIT
    assert np.isclose(p.zoom, 1.0)
    spec = LensSpec(arm="unzoomed", fraction=0.3)
    caps = captures_for(spec, [(480, 640)] * 10)
    assert all(c.is_identity() for c in caps) and caps[0].apply(img) is img


def test_warp_back_places_a_zoomed_view_in_the_centre():
    cap = zoom_capture((480, 640), 2.0)
    export = proc = (378, 504)
    A = AxisAffine.resize(export, (480, 640)).then(cap.affine()).then(AxisAffine.resize((480, 640), proc))
    out = warp_to(np.ones(proc, np.float32), A, export)
    ys, xs = np.nonzero(out)
    assert abs(xs.min() - 504 / 4) <= 1 and abs(xs.max() + 1 - 3 * 504 / 4) <= 1
    assert abs(ys.min() - 378 / 4) <= 1 and abs(ys.max() + 1 - 3 * 378 / 4) <= 1
    # values: a ramp over processed columns comes back at the expected positions (nearest neighbour)
    ramp = np.tile(np.arange(proc[1], dtype=np.float32) + 1.0, (proc[0], 1))
    back = warp_to(ramp, A, export)
    xe = np.arange(export[1])
    expect = np.floor(A.ax * (xe + 0.5) + A.bx) + 1.0          # 1-based processed column under each pixel
    row = back[189]
    valid = row > 0
    assert np.abs(row[valid] - expect[valid]).max() <= 1.0
    # intrinsics: a centred processed camera maps back to a centred export camera with focal / zoom
    Kp = np.array([[400.0, 0.0, 252.0], [0.0, 400.0, 189.0], [0.0, 0.0, 1.0]])
    Ke = transform_K(Kp, A.inverse())
    assert np.allclose(Ke[:2, 2], [252.0, 189.0]) and np.isclose(Ke[0, 0], 400.0 / A.ax) and np.isclose(A.ax, 2.0)


def test_warp_identity_returns_the_map_and_content_mask_zeroes_padding():
    m = np.random.default_rng(1).random((7, 9)).astype(np.float32)
    assert warp_to(m, AxisAffine(), (7, 9)) is m
    content = np.zeros((7, 9), np.float32)
    content[2:5, 3:6] = 1
    out = warp_to(m, AxisAffine(), (7, 9), content=content)
    assert np.array_equal(out[2:5, 3:6], m[2:5, 3:6]) and out.sum() == pytest.approx(m[2:5, 3:6].sum())


# ----------------------------------------------------------------------------- plans
def test_assign_kinds():
    k = assign_kinds(10, 0.3)
    assert k[0] == KIND_NORMAL and list(np.flatnonzero(k == KIND_ZOOM)) == [1, 5, 9]
    assert (assign_kinds(10, 0.0) == KIND_NORMAL).all() and (assign_kinds(1, 0.5) == KIND_NORMAL).all()
    kp = assign_kinds(10, 0.3, portrait=0.2)
    assert (kp == KIND_PORTRAIT).sum() == 2 and kp[0] == KIND_NORMAL and not ((kp == KIND_ZOOM) & (kp == KIND_PORTRAIT)).any()


def test_canonical_resolutions():
    assert canonical_resolutions([1, 1, 1], 504) == [504, 504, 504]
    assert canonical_resolutions([1, 2, 1, 2, 1], 504) == [504, 252, 504, 252, 504]
    assert canonical_resolutions([1, 1.1, 0.95, 2.0], 504, deadband=0.15) == [504, 504, 504, 252]
    assert canonical_resolutions([1, 1, 1, 0.5], 504, max_res=900)[3] == 900     # wide view, clipped
    assert canonical_resolutions([1, 3.0, 1], 504, min_res=200)[1] == 200
    with pytest.raises(ValueError):
        canonical_resolutions([1, -1], 504)


def test_oracle_kappa_follows_the_zoom():
    caps = captures_for(LensSpec(arm="canon_oracle", zoom=2.0, fraction=0.3), [(480, 640)] * 10)
    k = kappa_from_captures(caps)
    zi = [i for i, c in enumerate(caps) if c.kind == KIND_ZOOM]
    assert np.allclose(k[zi] / k[0], 2.0)
    assert canonical_resolutions(k, 504) == [252 if i in zi else 504 for i in range(10)]


def test_spec_parse_names_roundtrip():
    s = LensSpec.parse("arm=canon_self,zoom=2,fraction=0.3")
    assert s.arm == "canon_self" and s.protocol_id == "z2_f30" and s.protocol_input == "lens:zoom=2,fraction=0.3"
    assert LensSpec.parse("zoom=1.5,fraction=0.3").protocol_id == "z1p5_f30"
    assert LensSpec.parse(s.to_text()) == s
    for bad in ("arm=nope", "zoom=1", "fraction=1.2", "foo=1"):
        with pytest.raises(ValueError):
            LensSpec.parse(bad)


# ----------------------------------------------------------------------------- adapter
def _write_images(folder, n=5, h=120, w=160):
    import cv2

    rng = np.random.default_rng(0)
    files = []
    for i in range(n):
        p = folder / f"{i:03d}.png"
        cv2.imwrite(str(p), rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
        files.append(str(p))
    return files


@pytest.mark.parametrize("arm", ["unzoomed", "native", "canon_oracle", "canonpad_oracle", "canon_self"])
def test_no_zoomed_view_is_bit_identical_to_the_original_adapter(tmp_path, arm):
    from da3.enhancements.lens.adapter import LensAPI
    from da3.eval.bench_adapter import OursAsOfficialAPI

    files = _write_images(tmp_path)
    m = tiny_model("da3-small")
    ref = OursAsOfficialAPI(m, "cpu", process_res=98).predict(files)
    # canon_self: a random model's focal lengths differ wildly between views; a wide dead band
    # isolates the no-op path (the released model is checked on the GPU by lens_eval.py parity)
    spec = LensSpec(arm=arm, fraction=0.0, res=98, deadband=float("inf") if arm == "canon_self" else 0.15)
    pred = LensAPI(m, "cpu", spec=spec).predict(files)
    for k in ("depth", "conf", "extrinsics", "intrinsics"):
        assert np.array_equal(ref[k], pred[k]), k


def test_zoomed_views_export_and_meta(tmp_path):
    from da3.enhancements.lens.adapter import META_FILE, LensAPI

    files = _write_images(tmp_path, n=6)
    m = tiny_model("da3-small")
    for arm in ("native", "canon_oracle", "canonpad_oracle"):
        api = LensAPI(m, "cpu", spec=LensSpec(arm=arm, zoom=2.0, fraction=0.5, res=98, min_res=28))
        exp = tmp_path / f"exp_{arm}"
        pred = api.inference(files, export_dir=str(exp))
        assert pred["depth"].shape == (6, 70, 98) and pred["conf"].shape == (6, 70, 98)
        meta = np.load(exp / "exports" / META_FILE)
        z = meta["kinds"] == KIND_ZOOM
        assert z.sum() == 3 and not z[0]
        d = pred["depth"]
        assert np.all(d[z][:, :8] == 0) and np.all(d[z][:, :, :12] == 0)     # outside the central crop
        assert np.all(d[z][:, 30:40, 40:58] > 0) and np.all(d[~z] > 0)
        assert np.allclose(pred["intrinsics"][:, 0, 2], 49.0) and np.allclose(pred["intrinsics"][:, 1, 2], 35.0)
        hw = meta["view_hw"]
        if arm == "canon_oracle":
            assert np.all(hw[z].prod(1) < hw[~z].prod(1).min())             # zoomed views on a smaller grid
        else:
            assert len({tuple(h) for h in hw}) == 1                          # one grid for every view
        assert (exp / "exports" / "mini_npz" / "results.npz").exists()
    assert api.summary()["scenes"] == 1


def test_posed_mode_is_rejected(tmp_path):
    from da3.enhancements.lens.adapter import LensAPI

    files = _write_images(tmp_path, n=3)
    with pytest.raises(NotImplementedError):
        LensAPI(tiny_model("da3-small"), "cpu", spec=LensSpec(res=98)).predict(files, np.zeros((3, 3, 4)), np.zeros((3, 3, 3)))


# ----------------------------------------------------------------------------- metrics
def _cams(n=6, seed=0):
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(seed)
    w2c = np.zeros((n, 3, 4))
    for i in range(n):
        R = Rotation.from_euler("y", rng.uniform(-20, 20), degrees=True).as_matrix()
        C = np.array([rng.uniform(-1, 1), 0.0, -4.0 + rng.uniform(-0.5, 0.5)])
        w2c[i, :, :3] = R
        w2c[i, :, 3] = -R @ C
    return w2c


def test_scene_view_metrics_scale_ratio_and_focal():
    from da3.enhancements.lens.metrics import scene_view_metrics

    n, hw = 6, (30, 40)
    w2c = _cams(n)
    K = np.stack([_K(50.0, 40, 30)] * n)
    gt = [np.full(hw, 3.0, np.float32) for _ in range(n)]
    kinds = np.array([0, 1, 0, 1, 0, 0])
    depth = np.stack([g / (1.6 if k == 1 else 1.0) for g, k in zip(gt, kinds)])   # zoomed views 1.6x too close
    meta = {"kinds": kinds, "export_hw": np.array(hw), "orig_hw": np.array([hw] * n), "zoom": np.where(kinds == 1, 2.0, 1.0),
            "z_est": np.where(kinds == 1, 1.8, 1.0)}
    row = scene_view_metrics({"depth": depth, "extrinsics": w2c, "intrinsics": K}, w2c, K, meta, gt)
    assert row["scale_ratio"] == pytest.approx(1.6)
    assert row["auc"]["all"]["auc30"] == pytest.approx(1.0) and row["auc"]["cross"]["pairs"] == 8
    assert row["focal_ratio_zoom"] == pytest.approx(1.0) and row["z_err_zoom"] == pytest.approx(0.1)


def test_reprojection_consistency():
    from da3.enhancements.lens.metrics import reprojection_consistency

    # two cameras looking at a fronto-parallel plane z = 5 (world), shifted sideways
    K = _K(40.0, 40, 30)
    w2c = np.zeros((2, 3, 4))
    w2c[:, :, :3] = np.eye(3)
    w2c[1, 0, 3] = -0.2
    depth = [np.full((30, 40), 5.0), np.full((30, 40), 5.0)]
    good = reprojection_consistency(depth, np.stack([K, K]), w2c, stride=2)
    assert good["median_pct"] < 1e-6 and good["within_1pct"] == pytest.approx(1.0)
    bad = reprojection_consistency([depth[0], depth[1] * 1.1], np.stack([K, K]), w2c, stride=2)
    assert bad["median_pct"] == pytest.approx(100 * np.log(1.1), rel=1e-3)


# ----------------------------------------------------------------------------- records, exporters
def test_lens_records_form_their_own_protocol_group():
    from da3.compare import make_record, protocol, protocol_key

    base = make_record(enhancement="baseline", variant="head", tag="a", setting="official", weights="official:da3-small",
                       protocol=protocol(["hiroom"], "all", 100, ["pose"], "first"))
    lens = make_record(enhancement="baseline", variant="head", tag="b", setting="official", weights="official:da3-small",
                       protocol=protocol(["hiroom"], "all", 100, ["pose"], "first", input="lens:zoom=2,fraction=0.3"))
    assert "input" not in base["protocol"] and protocol_key(base) != protocol_key(lens)
    assert protocol_key(base)[-1] is None


def test_depth_mesh_accepts_views_of_different_sizes():
    from da3.export import depth_mesh

    rng = np.random.default_rng(0)
    pts = []
    for h, w in ((6, 8), (8, 6)):
        xx, yy = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
        pts.append(np.dstack([0.01 * xx, 0.01 * yy, np.full((h, w), 5.0)]))
    rgb = [rng.integers(0, 255, p.shape, dtype=np.uint8) for p in pts]
    conf = [np.ones(p.shape[:2]) for p in pts]
    v, f, c = depth_mesh(pts, rgb, conf, 40.0, step=1)
    assert len(v) == 6 * 8 * 2 and len(f) > 0
    vs, fs, cs = depth_mesh(np.stack([pts[0], pts[0]]), np.stack([rgb[0], rgb[0]]), np.stack([conf[0]] * 2), 40.0, step=1)
    v2, f2, c2 = depth_mesh([pts[0], pts[0]], [rgb[0], rgb[0]], [conf[0]] * 2, 40.0, step=1)
    assert np.array_equal(vs, v2) and np.array_equal(fs, f2)
