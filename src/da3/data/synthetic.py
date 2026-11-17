"""Procedural multi-view scenes (textured box room + spheres) with exact depth and poses.

Used only for pipeline tests and the CPU overfit sanity check; never for reported results.
"""

from __future__ import annotations

import numpy as np

from .canonical import write_scene


def _look_at(eye, target, up=(0, -1, 0)):
    """OpenCV camera-to-world (x right, y down, z forward) looking from eye at target."""
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(up, float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([x, y, z], 1)
    c2w[:3, 3] = eye
    return c2w


def _texture(P):
    """Procedural RGB colour at world points P (..., 3)."""
    f = np.stack([
        np.sin(3.1 * P[..., 0]) * np.cos(2.3 * P[..., 1]) + np.sin(5 * P[..., 2]),
        np.sin(4.7 * P[..., 1] + 1) * np.cos(1.9 * P[..., 2]),
        np.cos(2.9 * P[..., 0] - 3.3 * P[..., 2]),
    ], -1)
    checker = ((np.floor(P[..., 0] * 2) + np.floor(P[..., 1] * 2) + np.floor(P[..., 2] * 2)) % 2)[..., None]
    return np.clip(0.5 + 0.3 * f + 0.15 * checker, 0, 1)


def render(K, c2w, H, W, room=(3.0, 1.5, 3.0), spheres=((0.6, 0.2, 0.8, 0.5), (-0.9, 0.5, -0.4, 0.4))):
    ys, xs = np.meshgrid(np.arange(H) + 0.5, np.arange(W) + 0.5, indexing="ij")
    p = np.stack([xs, ys, np.ones_like(xs)], -1)
    d_cam = p @ np.linalg.inv(K).T                     # z = 1
    d = d_cam @ c2w[:3, :3].T
    o = c2w[:3, 3]
    t = np.full((H, W), np.inf)
    half = np.asarray(room)
    for ax in range(3):                                # box interior planes
        for sgn in (-1, 1):
            with np.errstate(divide="ignore", invalid="ignore"):
                ti = (sgn * half[ax] - o[ax]) / d[..., ax]
            ti = np.where(ti > 1e-6, ti, np.inf)
            t = np.minimum(t, ti)
    for cx, cy, cz, r in spheres:
        oc = o - np.array([cx, cy, cz])
        b = (d * oc).sum(-1)
        a = (d * d).sum(-1)
        c = (oc * oc).sum() - r * r
        disc = b * b - a * c
        ts = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / a, np.inf)
        t = np.minimum(t, np.where(ts > 1e-6, ts, np.inf))
    P = o + t[..., None] * d
    rgb = (_texture(P) * 255).astype(np.uint8)
    return rgb, t  # t is z-depth because d_cam has z = 1


def make_scene(out_dir, n_frames=24, H=240, W=320, seed=0):
    rng = np.random.default_rng(seed)
    f = W * (0.7 + 0.3 * rng.random())
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    frames = []
    for i in range(n_frames):
        a = 2 * np.pi * i / n_frames + rng.normal(0, 0.05)
        eye = np.array([1.6 * np.cos(a), 0.3 * rng.normal(), 1.6 * np.sin(a)])
        target = np.array([0.0, 0.2, 0.0]) + rng.normal(0, 0.2, 3)
        c2w = _look_at(eye, target)
        rgb, depth = render(K, c2w, H, W)
        frames.append({"name": f"{i:06d}", "rgb": rgb, "depth": depth, "K": K, "c2w": c2w})
    return write_scene(out_dir, frames, depth_unit=0.001, max_side=None)
