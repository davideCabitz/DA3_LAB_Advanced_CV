"""Cross-check our metric code (src/da3/eval/metrics.py) against the released benchmark functions on
identical random inputs: pose AUC, Umeyama + RANSAC Sim(3), reconstruction F1 / Chamfer.

  python scripts/check_metrics_vs_official.py --official-src /path/Depth-Anything-3/src
"""

import argparse
import json
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-src", required=True)
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()
    sys.path.insert(0, args.official_src)
    from depth_anything_3.bench.utils import compute_pose, evaluate_3d_reconstruction
    from depth_anything_3.utils.pose_align import align_poses_umeyama

    from da3.eval.metrics import align_sim3_ransac, pose_metrics, reconstruction_metrics
    from da3.camera import quat_to_mat

    rng = np.random.default_rng(0)
    diffs = {"auc": 0.0, "sim3_scale": 0.0, "sim3_R": 0.0, "recon": 0.0}
    for _ in range(args.trials):
        n = int(rng.integers(5, 40))
        q = torch.from_numpy(rng.normal(size=(n, 4)))
        R = quat_to_mat(q / q.norm(dim=-1, keepdim=True)).numpy()
        gt = np.tile(np.eye(4), (n, 1, 1))
        gt[:, :3, :3], gt[:, :3, 3] = R, rng.normal(size=(n, 3))
        pred = gt.copy()
        dq = torch.from_numpy(rng.normal(0, 0.05, size=(n, 4)) + np.array([0, 0, 0, 1.0]))
        pred[:, :3, :3] = quat_to_mat(dq / dq.norm(dim=-1, keepdim=True)).numpy() @ pred[:, :3, :3]
        pred[:, :3, 3] = 1.7 * pred[:, :3, 3] + rng.normal(0, 0.1, size=(n, 3))
        o = compute_pose(torch.from_numpy(pred), torch.from_numpy(gt))
        m = pose_metrics(pred, gt)
        diffs["auc"] = max(diffs["auc"], *(abs(float(o[k]) - m[k]) for k in ("auc30", "auc15", "auc05", "auc03")))
        r1, _, s1, _ = align_poses_umeyama(gt.copy(), pred.copy(), return_aligned=True, ransac=n >= 10, random_state=42)
        r2, _, s2, _ = align_sim3_ransac(gt, pred, ransac=n >= 10, seed=42)
        diffs["sim3_scale"] = max(diffs["sim3_scale"], abs(s1 - s2))
        diffs["sim3_R"] = max(diffs["sim3_R"], float(np.abs(r1 - r2).max()))
        P = rng.uniform(0, 1, (4000, 3))
        G = P + rng.normal(0, 0.03, P.shape)
        a = evaluate_3d_reconstruction(P, G, threshold=0.05, down_sample=0.02)
        b = reconstruction_metrics(P, G, threshold=0.05, down_sample=0.02)
        diffs["recon"] = max(diffs["recon"], *(abs(a[k] - b[k]) for k in ("acc", "comp", "precision", "recall", "fscore")))
    print(json.dumps({"max_abs_diff_ours_vs_official": diffs, "trials": args.trials}, indent=2))


if __name__ == "__main__":
    main()
