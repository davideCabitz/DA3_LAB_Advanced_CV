"""Per-scene depth-uncertainty calibration without ground truth, by triple collocation (F2).

DA3's depth confidence c is trained with c|e| - λ log c, so it ranks pixels but has no calibrated
unit. We fit sigma(c) = a * c^(-b) for the log-depth error of each scene WITHOUT ground truth:

1. sample anchor pixels; each anchor is a 3D point P seen by its view i (predicted depth + camera);
2. project P into the other views; keep views where P is inside the image and not occluded
   (z <= D_j (1 + tau)); pick two such views j, k at random;
3. three measurements of the same quantity, the log-depth of P in camera i:
   x_i = log D_i(p), x_j = z_i of the point that view j sees at P's projection (back-projected with
   D_j and transformed into camera i), x_k likewise;
4. if the three errors are independent, Var(x_a - x_b) = s_a^2 + s_b^2, hence
   s_a^2 = (Var(x_a - x_b) + Var(x_a - x_c) - Var(x_b - x_c)) / 2  (triple collocation).
   Every triplet is used three times (each member as target) and grouped by the target's
   confidence. The identity needs true variances (the partners' errors are scale mixtures, for
   which MAD-based variances are biased), so plain variances are used and gross outliers
   (residual occlusions) are removed by dropping triplets whose largest pairwise |difference|
   exceeds `limit` (0.5 in log-depth, i.e. a factor 1.65). GT comparisons use the same estimator
   (`trimmed_var`);
5. fit log s = log a - b log c over the confidence bins (count-weighted least squares).

Known limits (see the report): errors shared by all views (global scale drift, bending) and the
camera errors themselves are invisible to / absorbed by the consistency check, so sigma tends to
be too low where those dominate.
"""

from __future__ import annotations

import numpy as np

from .signals import to44

LIMIT = 0.5  # log-depth outlier limit (factor 1.65 in depth)


def trimmed_var(x: np.ndarray, limit: float = LIMIT) -> float:
    """Variance of x after dropping values farther than `limit` from the median (gross outliers)."""
    x = np.asarray(x, np.float64)
    x = x[np.abs(x - np.median(x)) <= limit] if len(x) else x
    if len(x) < 2:
        return float("nan")
    return float(np.var(x))


def fit_power_law(c_centers, sigmas, counts) -> tuple[float, float]:
    """Weighted LS fit of log sigma = log a - b log c. Returns (a, b)."""
    c, s, n = (np.asarray(v, np.float64) for v in (c_centers, sigmas, counts))
    ok = np.isfinite(s) & (s > 0) & np.isfinite(c) & (c > 0) & (n > 0)
    if ok.sum() == 0:
        return float("nan"), 0.0
    if ok.sum() == 1:
        return float(s[ok][0]), 0.0
    A = np.stack([np.ones(ok.sum()), -np.log(c[ok])], 1)
    w = np.sqrt(n[ok])
    sol, *_ = np.linalg.lstsq(A * w[:, None], np.log(s[ok]) * w, rcond=None)
    return float(np.exp(sol[0])), float(sol[1])


def sigma_from_conf(conf, a: float, b: float) -> np.ndarray:
    return a * np.clip(np.asarray(conf, np.float64), 1e-6, None) ** (-b)


def conf_bins(conf: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bin edges of log-confidence (unique, at least two edges)."""
    lc = np.log(np.clip(np.asarray(conf, np.float64), 1e-6, None))
    edges = np.unique(np.quantile(lc, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([lc.min() - 1e-6, lc.max() + 1e-6])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def tc_from_triplets(x: np.ndarray, c: np.ndarray, n_bins: int = 8, min_count: int = 100,
                     limit: float = LIMIT) -> dict:
    """Triple collocation on (N, 3) measurements `x` with confidences `c` -> fitted (a, b) + bins."""
    x, c = np.asarray(x, np.float64), np.asarray(c, np.float64)
    ok = np.isfinite(x).all(1) & np.isfinite(c).all(1)
    x, c = x[ok], c[ok]
    pair = np.stack([x[:, 0] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 2]], 1)
    inl = np.abs(pair - np.median(pair, axis=0)).max(1) <= limit
    n_outliers = int((~inl).sum())
    x, c = x[inl], c[inl]
    if len(x) < min_count:
        return {"a": float("nan"), "b": 0.0, "bins": [], "n_triplets": int(len(x)), "n_outliers": n_outliers}
    var = np.var
    roles = [(0, 1, 2), (1, 0, 2), (2, 0, 1)]
    tgt = np.concatenate([x[:, r[0]] for r in roles])
    o1 = np.concatenate([x[:, r[1]] for r in roles])
    o2 = np.concatenate([x[:, r[2]] for r in roles])
    ct = np.concatenate([c[:, r[0]] for r in roles])
    edges = conf_bins(ct, n_bins)
    lct = np.log(np.clip(ct, 1e-6, None))
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (lct > lo) & (lct <= hi)
        n = int(m.sum())
        if n < min_count:
            continue
        v = 0.5 * (var(tgt[m] - o1[m]) + var(tgt[m] - o2[m]) - var(o1[m] - o2[m]))
        bins.append({"c": float(np.exp(np.median(lct[m]))), "lo": float(lo), "hi": float(hi),
                     "sigma": float(np.sqrt(v)) if v > 0 else float("nan"), "count": n})
    a, b = fit_power_law([q["c"] for q in bins], [q["sigma"] for q in bins], [q["count"] for q in bins])
    if not np.isfinite(a):  # all bins degenerate: one pooled estimate, no confidence dependence
        v = 0.5 * (var(tgt - o1) + var(tgt - o2) - var(o1 - o2))
        a, b = (float(np.sqrt(v)) if v > 0 else float("nan")), 0.0
    return {"a": a, "b": b, "bins": bins, "n_triplets": int(len(x)), "n_outliers": n_outliers}


def gather_triplets(depth, conf, w2c, K, n_anchors: int = 20000, tau: float = 0.2, seed: int = 0,
                    chunk: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """(M, 3) log-depth measurements of the same points by three views + their confidences."""
    depth = np.asarray(depth, np.float64)
    conf = np.asarray(conf, np.float64)
    S, H, W = depth.shape
    if S < 3:
        return np.zeros((0, 3)), np.zeros((0, 3))
    T = to44(w2c)
    R, t = T[:, :3, :3], T[:, :3, 3]
    K = np.asarray(K, np.float64)
    Kinv = np.linalg.inv(K)
    rng = np.random.default_rng(seed)
    xs, cs = [], []
    for start in range(0, n_anchors, chunk):
        n = min(chunk, n_anchors - start)
        vi = rng.integers(0, S, n)
        u = rng.integers(0, W, n)
        v = rng.integers(0, H, n)
        d_i = depth[vi, v, u]
        good = np.isfinite(d_i) & (d_i > 0)
        vi, u, v, d_i = vi[good], u[good], v[good], d_i[good]
        n = len(vi)
        if n == 0:
            continue
        p = np.stack([u + 0.5, v + 0.5, np.ones(n)], 1)
        Xc = d_i[:, None] * np.einsum("nij,nj->ni", Kinv[vi], p)
        P = np.einsum("nji,nj->ni", R[vi], Xc - t[vi])                       # world points
        Xall = np.einsum("sab,nb->nsa", R, P) + t[None]                       # (n, S, 3)
        z = Xall[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uvw = np.einsum("sab,nsb->nsa", K, Xall)
            uj = uvw[..., 0] / uvw[..., 2]
            vj = uvw[..., 1] / uvw[..., 2]
        inside = (z > 1e-9) & (uj >= 0) & (uj < W) & (vj >= 0) & (vj < H)
        ui = np.clip(np.nan_to_num(uj, nan=0.0), 0, W - 1).astype(int)
        vi2 = np.clip(np.nan_to_num(vj, nan=0.0), 0, H - 1).astype(int)
        sidx = np.broadcast_to(np.arange(S), (n, S))
        Dj = depth[sidx, vi2, ui]
        Cj = conf[sidx, vi2, ui]
        ok = inside & (Dj > 0) & (z <= Dj * (1 + tau))
        ok[np.arange(n), vi] = False
        keep = ok.sum(1) >= 2
        if not keep.any():
            continue
        score = np.where(ok, rng.random((n, S)), -1.0)
        top2 = np.argsort(-score, axis=1)[:, :2]
        rows = np.arange(n)

        def transfer(jcol):
            j = top2[:, jcol]
            pj = np.stack([ui[rows, j] + 0.5, vi2[rows, j] + 0.5, np.ones(n)], 1)
            Xcj = Dj[rows, j][:, None] * np.einsum("nij,nj->ni", Kinv[j], pj)
            Pj = np.einsum("nji,nj->ni", R[j], Xcj - t[j])
            zi = np.einsum("nab,nb->na", R[vi], Pj)[:, 2] + t[vi][:, 2]
            return zi, Cj[rows, j]

        zj, cj = transfer(0)
        zk, ck = transfer(1)
        keep &= (zj > 0) & (zk > 0)
        if not keep.any():
            continue
        xs.append(np.stack([np.log(d_i[keep]), np.log(zj[keep]), np.log(zk[keep])], 1))
        cs.append(np.stack([conf[vi[keep], v[keep], u[keep]], cj[keep], ck[keep]], 1))
    if not xs:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(xs), np.concatenate(cs)


def fit_scene(depth, conf, w2c, K, n_anchors: int = 20000, n_bins: int = 8, seed: int = 0) -> dict:
    """GT-free sigma(c) for one scene (log-depth units)."""
    x, c = gather_triplets(depth, conf, w2c, K, n_anchors=n_anchors, seed=seed)
    return tc_from_triplets(x, c, n_bins=n_bins)
