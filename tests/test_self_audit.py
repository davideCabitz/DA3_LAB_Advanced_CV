"""Enhancement #1 (Self-Audit): signals, triple collocation, pose sources, metrics, adapter hook."""

import numpy as np
import pytest
import torch

from da3.camera import affine_inverse, quat_to_mat
from da3.enhancements.self_audit import metrics as M
from da3.enhancements.self_audit.models import Ridge, leave_one_group_out, prepare_features
from da3.enhancements.self_audit.pose_sources import fuse, inverse_variance_weight, select
from da3.enhancements.self_audit.signals import (
    ec_fixed_point_residuals,
    head_ray_gap,
    origin_spread,
    per_view_pose_error,
    pinhole_residual_deg,
    rot_angle_deg,
)
from da3.enhancements.self_audit.triple_collocation import fit_power_law, tc_from_triplets
from da3.geometry import make_ray_map
from helpers import random_camera


def _cams(S=6, H=48, W=64, seed=0):
    K, c2w = random_camera(B=S, H=H, W=W, seed=seed)
    return K, c2w, affine_inverse(c2w)


def _rot(axis, deg):
    q = torch.zeros(4, dtype=torch.float64)
    a = np.radians(deg) / 2
    q[:3] = torch.tensor(axis, dtype=torch.float64) / np.linalg.norm(axis) * np.sin(a)
    q[3] = np.cos(a)
    return quat_to_mat(q).numpy()


def test_pinhole_ray_map_has_no_residual_and_no_origin_spread():
    H, W = 48, 64
    K, c2w, _ = _cams(S=3, H=H, W=W)
    ray = make_ray_map(K, c2w, (H, W), ray_hw=(12, 16)).numpy()
    assert np.allclose(origin_spread(ray, None, 1.0), 0, atol=1e-9)
    assert pinhole_residual_deg(ray, None, (H, W)).max() < 1e-3


def test_pinhole_residual_sees_a_distorted_ray_map():
    H, W = 48, 64
    K, c2w, _ = _cams(S=2, H=H, W=W)
    ray = make_ray_map(K, c2w, (H, W), ray_hw=(12, 16)).numpy().copy()
    yy, xx = np.meshgrid(np.linspace(-1, 1, 12), np.linspace(-1, 1, 16), indexing="ij")
    ray[1, ..., 3:6] *= (1 + 0.3 * (xx**2 + yy**2))[..., None]   # scaling keeps directions...
    ray[1, ..., 3] += 0.2 * (xx**2 + yy**2)                        # ...this bends them (not pinhole)
    res = pinhole_residual_deg(ray, None, (H, W))
    assert res[0] < 1e-3 and res[1] > 0.1


def test_head_ray_gap_is_zero_for_identical_cameras_and_finds_a_rotated_view():
    K, _, w2c = _cams(S=6)
    w2c = w2c.numpy()
    g = head_ray_gap(w2c, K.numpy(), w2c, K.numpy())
    assert max(v.max() for v in g.values()) < 1e-5
    bad = w2c.copy()
    C = -bad[2, :3, :3].T @ bad[2, :3, 3]
    bad[2, :3, :3] = _rot([0, 1, 0], 10.0) @ bad[2, :3, :3]   # rotate view 2, keep its centre
    bad[2, :3, 3] = -bad[2, :3, :3] @ C
    g = head_ray_gap(w2c, K.numpy(), bad, K.numpy())
    assert abs(g["gap_rot_deg"][2] - 10.0) < 1e-3
    assert np.delete(g["gap_rot_deg"], 2).max() < 1e-3


def test_ec_residual_ignores_a_global_rotation():
    rng = np.random.default_rng(0)
    K, c2w, w2c = _cams(S=5)
    w2c = w2c.numpy()
    G = _rot([1, 2, 3], 25.0)
    moved = w2c.copy()
    moved[:, :3, :3] = w2c[:, :3, :3] @ G.T      # same cameras in a rotated world frame
    depth = rng.uniform(1, 3, (5, 8, 8))
    ray = make_ray_map(K, c2w, (48, 64), ray_hw=(6, 8)).numpy()
    ray2 = ray.copy()
    ray2[..., 3:6] = ray[..., 3:6] @ G.T
    out = ec_fixed_point_residuals(depth, depth * 2.0, w2c, moved, ray, ray2)
    assert out["ec_depth_change"].max() < 1e-9          # a global depth scale is not a change
    assert out["ec_rot_change_deg"].max() < 1e-4
    assert out["ec_ray_change_deg"].max() < 1e-4


def test_per_view_pose_error_flags_the_wrong_view():
    _, _, w2c = _cams(S=6)
    gt = w2c.numpy()
    pred = gt.copy()
    pred[4, :3, :3] = _rot([1, 0, 0], 20.0) @ pred[4, :3, :3]
    e = per_view_pose_error(pred, gt)
    assert np.argmax(e["rot_err"]) == 4 and e["rot_err"][4] > 15
    assert np.delete(e["rot_err"], 4).max() < 1e-3


def test_triple_collocation_recovers_a_known_sigma_law():
    rng = np.random.default_rng(0)
    N, a, b = 120000, 0.05, 0.7
    c = np.exp(rng.uniform(0, 2, (N, 3)))
    truth = rng.uniform(0, 2, (N, 1))
    x = truth + a * c ** (-b) * rng.standard_normal((N, 3))
    out = tc_from_triplets(x, c, n_bins=8)
    assert abs(out["a"] - a) / a < 0.1
    assert abs(out["b"] - b) < 0.1
    for q in out["bins"]:
        assert abs(q["sigma"] - a * q["c"] ** (-b)) / (a * q["c"] ** (-b)) < 0.15


def test_power_law_fit_exact():
    c = np.array([1.0, 2.0, 4.0, 8.0])
    a, b = fit_power_law(c, 0.1 * c**-0.5, np.ones(4))
    assert abs(a - 0.1) < 1e-9 and abs(b - 0.5) < 1e-9


def test_fusion_and_selection_are_consistent():
    K, _, w2c = _cams(S=5)
    w2c, K = w2c.numpy(), K.numpy()
    for w in (0.0, 0.3, 1.0):
        f_w2c, f_K = fuse(w2c, K, w2c, K, np.full(5, w))
        assert np.allclose(f_w2c[:, :3, :4], w2c[:, :3, :4], atol=1e-6) and np.allclose(f_K, K)
    s_w2c, _ = select(w2c, K, w2c, K, np.array([True, False, True, False, True]))
    assert np.allclose(s_w2c[:, :3, :4], w2c[:, :3, :4], atol=1e-6)
    assert np.allclose(inverse_variance_weight([1.0, 2.0], [1.0, 1.0]), [0.5, 0.8])


def test_detection_metrics():
    err = np.array([1.0, 2.0, 3.0, 20.0, 30.0])
    assert M.auroc(err, err > 15) == 1.0
    assert M.auroc(-err, err > 15) == 0.0
    assert np.isnan(M.auroc(err, err > 100))
    assert abs(M.spearman(err, err) - 1) < 1e-12
    _, _, oracle = M.risk_coverage(err, err)
    _, _, worst = M.risk_coverage(-err, err)
    assert oracle < worst
    e = np.random.default_rng(0).standard_normal(200000) * 0.1
    assert abs(M.coverage(e, np.full_like(e, 0.1), 0.9) - 0.9) < 0.01


def test_ridge_and_leave_one_group_out():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 3))
    y = 2 * X[:, 0] - X[:, 2] + 0.01 * rng.standard_normal(300)
    pred, scheme, fitted = leave_one_group_out(X, y, np.repeat(["a", "b", "c"], 100), np.arange(300))
    assert scheme == "leave-one-dataset-out" and len(fitted) == 3
    assert np.corrcoef(pred, y)[0, 1] > 0.99
    X[0, 1] = np.nan
    assert np.isfinite(Ridge().fit(X, y).predict(X)).all()   # missing values are imputed
    feats = prepare_features({"gap_rot_deg": np.array([0.0, 9.0]), "neg_logconf_depth_mean": np.array([-1.0, 1.0])},
                             ["gap_rot_deg", "neg_logconf_depth_mean"])
    assert np.allclose(feats, [[0.0, -1.0], [np.log(10.0), 1.0]])


def test_rot_angle():
    assert abs(rot_angle_deg(_rot([0, 0, 1], 33.0), np.eye(3)) - 33.0) < 1e-9


def test_adapter_dump_does_not_change_the_exported_results(tmp_path):
    """Benchmark adapter with and without the Self-Audit dumper: identical results.npz, plus a dump."""
    cv2 = pytest.importorskip("cv2")
    from da3.enhancements.self_audit import AuditDumper
    from da3.eval.bench_adapter import OursAsOfficialAPI
    from helpers import tiny_model

    rng = np.random.default_rng(0)
    files = []
    for i in range(3):
        p = tmp_path / f"im{i}.png"
        cv2.imwrite(str(p), rng.integers(0, 255, (60, 80, 3), dtype=np.uint8))
        files.append(str(p))
    model = tiny_model("da3-small")
    outs = {}
    for name, dumper in (("plain", None), ("dump", AuditDumper(ray_solver=None, repass=True))):
        api = OursAsOfficialAPI(model, "cpu", process_res=98, amp=False, dumper=dumper)
        export = tmp_path / name
        api.inference(files, export_dir=str(export))
        outs[name] = dict(np.load(export / "exports" / "mini_npz" / "results.npz"))
    for k in outs["plain"]:
        assert np.array_equal(outs["plain"][k], outs["dump"][k]), k
    assert not (tmp_path / "plain" / "exports" / "audit_dump.npz").exists()
    d = np.load(tmp_path / "dump" / "exports" / "audit_dump.npz")
    for k in ("head_w2c", "head_K", "ray_dlt_w2c", "ray_dlt_K", "ray_ds", "ray_conf_ds", "depth_ds",
              "repass_w2c", "repass_depth_ds", "repass_ray_ds"):
        assert k in d, k
    assert d["head_w2c"].shape == (3, 3, 4) and d["ray_ds"].shape[-1] == 6
    assert np.allclose(d["head_w2c"], outs["plain"]["extrinsics"], atol=1e-5)


def test_adapter_resume_skips_complete_scenes_only(tmp_path):
    """skip_existing: a complete export + dump is reused (model not run); a missing or truncated
    dump is recomputed; no temporary files are left behind."""
    cv2 = pytest.importorskip("cv2")
    from da3.enhancements.self_audit import AuditDumper
    from da3.eval.bench_adapter import OursAsOfficialAPI
    from helpers import tiny_model

    rng = np.random.default_rng(1)
    files = []
    for i in range(2):
        p = tmp_path / f"im{i}.png"
        cv2.imwrite(str(p), rng.integers(0, 255, (60, 80, 3), dtype=np.uint8))
        files.append(str(p))
    model = tiny_model("da3-small")
    export = tmp_path / "scene"
    first = OursAsOfficialAPI(model, "cpu", process_res=98, amp=False, dumper=AuditDumper(ray_solver=None, repass=False))
    first.inference(files, export_dir=str(export))
    res_path = export / "exports" / "mini_npz" / "results.npz"
    dump_path = export / "exports" / "audit_dump.npz"
    saved = dict(np.load(res_path))
    assert not list(export.rglob("*.partial.npz"))

    again = OursAsOfficialAPI(model, "cpu", process_res=98, amp=False, skip_existing=True,
                              dumper=AuditDumper(ray_solver=None, repass=False))
    pred = again.inference(files, export_dir=str(export))
    assert again.skipped == 1 and again.timings == []
    for k in ("depth", "conf", "extrinsics", "intrinsics"):
        assert np.array_equal(pred[k], saved[k]), k

    dump_path.write_bytes(dump_path.read_bytes()[:100])  # truncated copy (interrupted sync)
    third = OursAsOfficialAPI(model, "cpu", process_res=98, amp=False, skip_existing=True,
                              dumper=AuditDumper(ray_solver=None, repass=False))
    third.inference(files, export_dir=str(export))
    assert third.skipped == 0 and len(third.timings) == 1
    assert "head_w2c" in np.load(dump_path)
