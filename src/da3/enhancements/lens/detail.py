"""D1 "nested detail views" of enhancement #4 (docs/LENS.md §13): zoomed photos as detail, not noise.

A zoomed view has two depth maps of the same camera:
* the *coarse* one from the padded pass (equal angular resolution on the full canvas): right scale and pose
  (plan v1: scale ratio 1.043), but the photo was shrunk, so its fine detail is gone;
* the *detail* one from a pass at the photo's native resolution (the released preprocessing): z x the angular
  resolution, but DA3 reads the zoom as "moving closer", so its scale is off by about zoom^0.92.

Both are maps of one camera, so pixel (u, v) of one maps exactly to a known pixel of the other (axis affine).
`fit_scale_shift` aligns the detail map to the coarse one with a robust scale and shift: the paper's own tool for
putting relative depth on a reference (Sec. 4.2, Eq. 8: RANSAC least squares). The aligned detail map then
replaces the coarse one for that view; the camera stays the padded pass's.
"""

from __future__ import annotations

import numpy as np


def fit_scale_shift(src, ref, valid=None, iters: int = 200, thresh: float = 0.05, min_px: int = 32,
                    seed: int = 0, sample: int = 20000) -> dict:
    """Robust (a, b) with ref ≈ a * src + b on the valid pixels.

    RANSAC over 2-pixel samples, hypotheses scored on at most `sample` random valid pixels; inliers have a
    relative residual |a src + b - ref| / ref < thresh; the final (a, b) is the least-squares fit on all inliers
    of the best hypothesis (paper Sec. 4.2, Eq. 8). Falls back to a scale-only median ratio when the fit is
    degenerate (a <= 0) or too few pixels are valid.
    Returns {"a", "b", "inliers" (share of valid pixels), "n" (valid pixels), "mode"}."""
    src = np.asarray(src, np.float64).ravel()
    ref = np.asarray(ref, np.float64).ravel()
    ok = np.isfinite(src) & np.isfinite(ref) & (src > 0) & (ref > 0)
    if valid is not None:
        ok &= np.asarray(valid, bool).ravel()
    s, r = src[ok], ref[ok]
    n = int(len(s))
    if n < max(min_px, 2):
        return {"a": float("nan"), "b": 0.0, "inliers": 0.0, "n": n, "mode": "too_few"}
    rng = np.random.default_rng(seed)
    sub = rng.choice(n, sample, replace=False) if n > sample else np.arange(n)
    ss, rs = s[sub], r[sub]
    best_ab, best_cnt = None, -1
    for _ in range(iters):
        i, j = rng.integers(0, len(ss), 2)
        if abs(ss[i] - ss[j]) < 1e-9 * max(ss[i], ss[j]):
            continue
        a = (rs[i] - rs[j]) / (ss[i] - ss[j])
        if a <= 0:
            continue
        b = rs[i] - a * ss[i]
        c = int((np.abs(a * ss + b - rs) < thresh * rs).sum())
        if c > best_cnt:
            best_ab, best_cnt = (a, b), c
    if best_ab is not None and best_cnt >= 2:
        inl = np.abs(best_ab[0] * s + best_ab[1] - r) < thresh * r
        if inl.sum() >= max(min_px // 2, 2):
            A = np.stack([s[inl], np.ones(int(inl.sum()))], 1)
            (a, b), *_ = np.linalg.lstsq(A, r[inl], rcond=None)
            if a > 0:
                inl = np.abs(a * s + b - r) < thresh * r
                return {"a": float(a), "b": float(b), "inliers": float(inl.mean()), "n": n, "mode": "scale_shift"}
    a = float(np.median(r / s))
    return {"a": a, "b": 0.0, "inliers": float((np.abs(a * s - r) < thresh * r).mean()), "n": n, "mode": "scale"}


def apply_scale_shift(depth, fit: dict, floor: float = 1e-6) -> np.ndarray:
    """a * depth + b on the valid pixels (> 0), clipped at a small positive floor; invalid pixels stay 0."""
    d = np.asarray(depth, np.float32)
    if not np.isfinite(fit.get("a", np.nan)):
        return d
    out = (fit["a"] * d.astype(np.float64) + fit["b"]).astype(np.float32)
    out = np.maximum(out, floor).astype(np.float32)
    return np.where(np.isfinite(d) & (d > 0), out, 0.0).astype(np.float32)
