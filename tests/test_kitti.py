"""KITTI raw geometry, drive split (leakage guard) and clip metrics, on synthetic data."""

import numpy as np
import pytest

from da3.data.kitti import (
    camera_poses,
    center_crop_box,
    check_split,
    crop_intrinsics,
    drive_id,
    kitti_calibration,
    lidar_depth_map,
    make_split,
    oxts_to_poses,
    read_eigen_test_drives,
)
from da3.eval.clip_metrics import clip_depth_metrics, trajectory_metrics

K = np.array([[721.5, 0, 609.6], [0, 721.5, 172.9], [0, 0, 1.0]])


def _rot(yaw, pitch=0.0, roll=0.0):
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("zyx", [yaw, pitch, roll]).as_matrix()


def _write_calib(d, P2, R_rect, T_cam0_velo, T_velo_imu):
    fmt = lambda a: " ".join(f"{v:.12e}" for v in np.ravel(a))
    (d / "calib_cam_to_cam.txt").write_text(
        f"calib_time: 09-Jan-2012 13:57:47\nR_rect_00: {fmt(R_rect)}\nP_rect_02: {fmt(P2)}\n")
    (d / "calib_velo_to_cam.txt").write_text(f"R: {fmt(T_cam0_velo[:3, :3])}\nT: {fmt(T_cam0_velo[:3, 3])}\n")
    (d / "calib_imu_to_velo.txt").write_text(f"R: {fmt(T_velo_imu[:3, :3])}\nT: {fmt(T_velo_imu[:3, 3])}\n")


def test_calibration_matches_the_projection_matrix(tmp_path):
    P2 = np.c_[K, K @ np.array([0.06, -0.002, 0.003])]
    R_rect = _rot(0.01, 0.002, -0.003)
    T_cv = np.eye(4)
    T_cv[:3, :3] = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]) @ _rot(0.01)
    T_cv[:3, 3] = [-0.004, -0.076, -0.27]
    T_vi = np.eye(4)
    T_vi[:3, 3] = [-0.81, 0.32, -0.8]
    _write_calib(tmp_path, P2, R_rect, T_cv, T_vi)
    c = kitti_calibration(tmp_path)
    Rr = np.eye(4)
    Rr[:3, :3] = R_rect
    pts = np.c_[np.random.default_rng(0).uniform([5, -5, -1], [40, 5, 2], (50, 3)), np.ones(50)]
    a = (P2 @ Rr @ T_cv @ pts.T).T          # devkit projection
    b = (K @ (c["T_cam2_velo"] @ pts.T)[:3]).T
    assert np.allclose(a, b, rtol=1e-10, atol=1e-8)
    assert np.allclose(c["T_cam2_imu"], c["T_cam2_velo"] @ T_vi)


def test_oxts_poses_start_at_identity_and_follow_the_motion():
    pk = np.zeros((3, 6))
    pk[:, 0], pk[:, 1], pk[:, 2] = 49.0, [8.0, 8.0001, 8.0002], 110.0
    pk[2, 5] = np.pi / 2
    T = oxts_to_poses(pk)
    assert np.allclose(T[0], np.eye(4))
    dx = T[1, 0, 3]
    assert 7.0 < dx < 7.5                    # 1e-4 deg of longitude at 49 deg N ~ 7.3 m east
    assert np.allclose(T[2, :3, :3], _rot(np.pi / 2), atol=1e-12)
    c2w = camera_poses(T, np.eye(4))
    assert np.allclose(c2w, T)


def test_lidar_depth_projection_and_nearest_point_wins():
    T = np.eye(4)
    T[:3, :3] = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])  # velo (x fwd, y left, z up) -> cam
    # the 2nd point lies on the same ray as the 1st, twice as far
    cam_pts = np.array([[1.0, 0.5, 10.0], [2.0, 1.0, 20.0], [-2.0, 0.0, 15.0], [0.0, 0.0, -5.0]])
    velo = (np.linalg.inv(T) @ np.c_[cam_pts, np.ones(4)].T).T[:, :3]
    d = lidar_depth_map(velo, T, K, (375, 1242))
    u, v = np.round(K[0, 0] * 0.1 + K[0, 2]).astype(int), np.round(K[1, 1] * 0.05 + K[1, 2]).astype(int)
    assert d[v, u] == pytest.approx(10.0)    # 20 m point behind it is hidden
    u2, v2 = np.round(K[0, 0] * (-2 / 15) + K[0, 2]).astype(int), np.round(K[1, 2]).astype(int)
    assert d[v2, u2] == pytest.approx(15.0)
    assert (d > 0).sum() == 2                # the point behind the camera is dropped


def test_crop_and_intrinsics():
    box = center_crop_box(375, 1242, 1.8)
    assert box == (0, 283, 375, 675)
    Kc = crop_intrinsics(K, box, (280, 504))
    s = 504 / 675
    assert np.allclose(Kc[0], [K[0, 0] * s, 0, (K[0, 2] - 283) * s])
    assert np.allclose(Kc[1], [0, K[1, 1] * 280 / 375, K[1, 2] * 280 / 375])


def test_eigen_list_and_split_guard(tmp_path):
    f = tmp_path / "test_files.txt"
    f.write_text("2011_09_26/2011_09_26_drive_0002_sync 0000000069 l\n"
                 "2011_09_26/2011_09_26_drive_0002_sync 0000000054 l\n"
                 "2011_09_30/2011_09_30_drive_0016_sync 0000000010 l\n")
    eigen = read_eigen_test_drives(f)
    assert eigen == {"2011_09_26_drive_0002", "2011_09_30_drive_0016"}
    assert drive_id("x/2011_09_26_drive_0005_sync") == "2011_09_26_drive_0005"
    with pytest.raises(ValueError, match="Eigen"):
        check_split(["2011_09_26_drive_0002_sync"], [], [], eigen)
    with pytest.raises(ValueError, match="share"):
        check_split(["2011_09_26_drive_0005"], ["2011_09_26_drive_0005"], [], eigen)
    avail = [f"2011_09_26_drive_{i:04d}_sync" for i in (1, 2, 5, 9, 11, 14, 15)] + ["2011_09_30_drive_0016_sync"]
    s1, s2 = (make_split(avail, eigen, n_test=2, n_val=2, seed=0) for _ in range(2))
    assert s1 == s2                          # deterministic
    assert set(s1["test"]) == eigen and len(s1["val"]) == 2
    assert not (set(s1["train"]) | set(s1["val"])) & eigen
    assert len(s1["train"]) + len(s1["val"]) == 6


def test_clip_depth_metrics():
    rng = np.random.default_rng(0)
    gt = rng.uniform(2, 60, (3, 20, 30)) * (rng.random((3, 20, 30)) < 0.3)
    pred = np.where(gt > 0, gt, rng.uniform(2, 60, gt.shape)) / 7.0   # one global scale
    m = clip_depth_metrics(pred, gt)
    assert m["absrel_clip"] == pytest.approx(0, abs=1e-12) and m["delta1_clip"] == 1.0
    assert m["scale_clip"] == pytest.approx(7.0)
    pred2 = pred.copy()
    pred2[1] *= 2.0                          # view 1 at another scale: only the image-scale metric forgives it
    m2 = clip_depth_metrics(pred2, gt)
    assert m2["absrel_image"] == pytest.approx(0, abs=1e-12) and m2["absrel_clip"] > 0.2
    gt_far = gt.copy()
    gt_far[gt_far > 0] = 100.0               # beyond 80 m: nothing valid
    assert np.isnan(clip_depth_metrics(pred, gt_far)["absrel_clip"])


def test_trajectory_metrics_are_sim3_invariant():
    S = 6
    c2w = np.tile(np.eye(4), (S, 1, 1))
    c2w[:, 2, 3] = np.arange(S) * 1.5        # driving forward
    c2w[:, 0, 3] = 0.1 * np.arange(S) ** 2
    for i in range(S):
        c2w[i, :3, :3] = _rot(0.02 * i)
    gt_w2c = np.linalg.inv(c2w)
    G = np.eye(4)
    G[:3, :3], G[:3, 3] = _rot(0.7, 0.2, -0.1), [3, -1, 2]
    pred_c2w = G @ c2w
    pred_c2w[:, :3, 3] = 0.3 * pred_c2w[:, :3, 3]   # Sim(3): rotation, translation, scale 0.3
    m = trajectory_metrics(np.linalg.inv(pred_c2w)[:, :3], gt_w2c)
    assert m["ate"] < 1e-9 and m["rpe_trans"] < 1e-9 and m["rpe_rot"] < 1e-6
    assert m["sim3_scale"] == pytest.approx(1 / 0.3)
    assert m["path_length"] == pytest.approx(np.sum(np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1)))
    noisy = pred_c2w.copy()
    noisy[3, :3, 3] += 0.5
    assert trajectory_metrics(np.linalg.inv(noisy)[:, :3], gt_w2c)["ate"] > 0.1
