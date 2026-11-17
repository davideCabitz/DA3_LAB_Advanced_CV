"""Sanity-check a canonical dataset: reproject the depth of frame j into frame i and compare.

A median relative depth error near 0 (and a high inlier ratio) on overlapping pixels means the
converter got K, c2w (OpenCV convention) and the depth unit right.

  python scripts/check_dataset.py <canonical_root> --pairs 20
"""

import argparse
import json
import random

import numpy as np

from da3.data import CanonicalScene, list_scenes


def reproject_error(scene: CanonicalScene, i: int, j: int):
    _, di = scene.load(i)
    _, dj = scene.load(j)
    H, W = dj.shape
    ys, xs = np.nonzero(dj > 0)
    if len(xs) > 20000:
        sel = np.random.default_rng(0).choice(len(xs), 20000, replace=False)
        ys, xs = ys[sel], xs[sel]
    p = np.stack([xs + 0.5, ys + 0.5, np.ones_like(xs, dtype=float)], -1)
    Xj = (p @ np.linalg.inv(scene.K[j]).T) * dj[ys, xs][:, None]
    Xw = Xj @ scene.c2w[j][:3, :3].T + scene.c2w[j][:3, 3]
    w2c_i = np.linalg.inv(scene.c2w[i])
    Xi = Xw @ w2c_i[:3, :3].T + w2c_i[:3, 3]
    front = Xi[:, 2] > 1e-3
    uv = Xi[front] @ scene.K[i].T
    uv = uv[:, :2] / uv[:, 2:]
    u, v = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    z_proj = Xi[front][inside, 2]
    z_obs = di[v[inside], u[inside]]
    ok = z_obs > 0
    if ok.sum() < 100:
        return None
    rel = np.abs(z_proj[ok] - z_obs[ok]) / z_obs[ok]
    return float(np.median(rel)), float((rel < 0.05).mean()), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--max-gap", type=int, default=5)
    args = ap.parse_args()
    rng = random.Random(0)
    scenes = [CanonicalScene(p) for p in list_scenes(args.root)]
    res = []
    for _ in range(args.pairs):
        s = rng.choice(scenes)
        i = rng.randrange(len(s))
        j = min(len(s) - 1, i + rng.randint(1, args.max_gap))
        r = reproject_error(s, i, j)
        if r:
            res.append(r)
    med = np.array([r[0] for r in res])
    inl = np.array([r[1] for r in res])
    out = {"scenes": len(scenes), "pairs": len(res), "median_rel_err": float(np.median(med)),
           "mean_inlier_5pct": float(inl.mean())}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
