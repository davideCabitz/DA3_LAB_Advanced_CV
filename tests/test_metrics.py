import numpy as np
import torch

from da3.camera import affine_inverse
from da3.eval import align_sim3_ransac, pose_metrics, reconstruction_metrics, umeyama, voxel_downsample
from helpers import random_camera


def _w2c(n=8, seed=0):
    _, c2w = random_camera(B=n, seed=seed)
    return affine_inverse(c2w).numpy()


def test_pose_auc_perfect_and_similarity_invariant():
    gt = _w2c()
    assert all(abs(v - 1.0) < 1e-9 for v in pose_metrics(gt, gt).values())
    # the relative-pose metric is invariant to a global similarity transform of the prediction
    S = np.eye(4)
    S[:3, :3] = random_camera(B=1, seed=5)[1][0, :3, :3].numpy()
    S[:3, 3] = [1, 2, 3]
    pred = gt @ np.linalg.inv(S)
    pred[:, :3, 3] *= 3.0
    assert all(abs(v - 1.0) < 1e-9 for v in pose_metrics(pred, gt).values())


def test_pose_auc_degrades_with_noise():
    gt = _w2c(12, seed=1)
    noisy = gt.copy()
    rng = np.random.default_rng(0)
    noisy[:, :3, 3] += rng.normal(0, 0.3, (12, 3))
    m = pose_metrics(noisy, gt)
    assert m["auc03"] < m["auc30"] < 1.0


def test_umeyama_and_ransac_recover_sim3():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 3))
    R = random_camera(B=1, seed=3)[1][0, :3, :3].numpy()
    dst = 2.5 * src @ R.T + np.array([0.3, -1, 2])
    R_, t_, s_ = umeyama(src, dst)
    assert np.allclose(R_, R, atol=1e-9) and np.isclose(s_, 2.5) and np.allclose(t_, [0.3, -1, 2])
    # camera-centre alignment with 3 outlier cameras
    gt = _w2c(12, seed=2)
    c2w = np.linalg.inv(gt)
    c2w[:, :3, 3] = 0.5 * c2w[:, :3, 3]
    c2w[:3, :3, 3] += 10.0                                 # outliers
    _, _, s, aligned = align_sim3_ransac(gt, np.linalg.inv(c2w), ransac=True)
    assert np.isclose(s, 2.0, rtol=1e-6)
    err = np.linalg.norm(np.linalg.inv(aligned)[3:, :3, 3] - np.linalg.inv(gt)[3:, :3, 3], axis=1)
    assert err.max() < 1e-6


def test_reconstruction_metrics():
    rng = np.random.default_rng(0)
    gt = rng.uniform(0, 1, (5000, 3))
    m = reconstruction_metrics(gt.copy(), gt, threshold=0.05)
    assert m["fscore"] == 1.0 and m["overall"] == 0.0
    m2 = reconstruction_metrics(gt + [2.0, 0, 0], gt, threshold=0.05)
    assert m2["fscore"] == 0.0 and m2["acc"] > 0.9
    v = voxel_downsample(np.array([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [0.9, 0.9, 0.9]]), 0.1)
    assert len(v) == 2 and np.allclose(sorted(v[:, 0])[0], 0.015)
