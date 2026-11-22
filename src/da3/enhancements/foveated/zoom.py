"""Region level of enhancement #2 (H4): exact virtual zoom cameras from the full-resolution photo.

A region (box) of a high-resolution photo becomes a narrow-FOV pinhole view by a pure rotation of
the source camera: zoom pixel p_z -> source ray d = R_v K_z^-1 p_z -> source pixel K_0 d. Only the
orientation changes, so the warp is exact for any scene depth. The zoom camera has its principal
point at the image centre (what DA3 assumes), unlike a naive axis-aligned crop of the region, whose
principal point is off-centre. Its camera is known exactly (source camera rotated by R_v, focal
K_z), so it can be given to DA3 through E_c together with the scene views, and DA3's own estimate of
R_v is a free check that it follows the camera input.

Conventions: continuous pixel coordinates with pixel centres at +0.5 (COLMAP / ETH3D, as
da3.geometry); array index = coordinate - 0.5. R_v maps zoom-camera coordinates to source-camera
coordinates (X_src = R_v X_zoom).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ZoomView:
    R_v: np.ndarray   # (3, 3) zoom-camera -> source-camera rotation
    K: np.ndarray     # (3, 3) intrinsics of the zoom image (centred principal point)
    hw: tuple         # (H_z, W_z)
    box: tuple        # (x0, y0, w, h) in source (full-resolution) pixels


def zoom_view(K0: np.ndarray, box, out_hw) -> ZoomView:
    """Virtual camera looking at the centre of `box` (minimal roll w.r.t. the source camera); the
    focal is the largest one whose image still contains the whole box."""
    K0 = np.asarray(K0, np.float64)
    x0, y0, w, h = (float(v) for v in box)
    Hz, Wz = int(out_hw[0]), int(out_hw[1])
    Kinv = np.linalg.inv(K0)
    z = Kinv @ np.array([x0 + w / 2, y0 + h / 2, 1.0])
    z /= np.linalg.norm(z)
    x = np.array([1.0, 0.0, 0.0]) - z[0] * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_v = np.stack([x, y, z], axis=1)
    xs, ys = np.linspace(x0, x0 + w, 9), np.linspace(y0, y0 + h, 9)
    border = np.array([(u, v) for u in xs for v in (y0, y0 + h)] + [(u, v) for u in (x0, x0 + w) for v in ys])
    d = (np.c_[border, np.ones(len(border))] @ Kinv.T) @ R_v          # zoom-camera directions
    u, v = d[:, 0] / d[:, 2], d[:, 1] / d[:, 2]
    f = min(Wz / (2 * np.abs(u).max()), Hz / (2 * np.abs(v).max()))
    K = np.array([[f, 0.0, Wz / 2], [0.0, f, Hz / 2], [0.0, 0.0, 1.0]])
    return ZoomView(R_v=R_v, K=K, hw=(Hz, Wz), box=(int(box[0]), int(box[1]), int(box[2]), int(box[3])))


def zoom_to_source(zv: ZoomView, K0: np.ndarray) -> np.ndarray:
    """(H_z * W_z, 2) source continuous coordinates of the zoom pixel centres."""
    Hz, Wz = zv.hw
    uu, vv = np.meshgrid(np.arange(Wz) + 0.5, np.arange(Hz) + 0.5)
    pz = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)], 1)
    d = pz @ (zv.R_v @ np.linalg.inv(zv.K)).T
    q = d @ np.asarray(K0, np.float64).T
    return q[:, :2] / q[:, 2:3]


def render_zoom(image: np.ndarray, K0: np.ndarray, zv: ZoomView) -> np.ndarray:
    """Warp the full-resolution image into the zoom view (bilinear, after an area pre-filter of the
    needed source region down to the zoom sampling density, so nothing aliases)."""
    Hz, Wz = zv.hw
    H0, W0 = image.shape[:2]
    q = zoom_to_source(zv, K0)
    x_lo, y_lo = max(0, int(np.floor(q[:, 0].min())) - 2), max(0, int(np.floor(q[:, 1].min())) - 2)
    x_hi, y_hi = min(W0, int(np.ceil(q[:, 0].max())) + 2), min(H0, int(np.ceil(q[:, 1].max())) + 2)
    patch = image[y_lo:y_hi, x_lo:x_hi]
    s = float(zv.K[0, 0] / np.asarray(K0)[0, 0])            # zoom pixels per source pixel
    sx = sy = 1.0
    if s < 1.0:
        nw, nh = max(1, int(round(patch.shape[1] * s))), max(1, int(round(patch.shape[0] * s)))
        sx, sy = nw / patch.shape[1], nh / patch.shape[0]
        patch = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)
    mx = ((q[:, 0] - x_lo) * sx - 0.5).reshape(Hz, Wz).astype(np.float32)
    my = ((q[:, 1] - y_lo) * sy - 0.5).reshape(Hz, Wz).astype(np.float32)
    return cv2.remap(patch, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def crop_view(image: np.ndarray, box, out_hw) -> np.ndarray:
    """Naive baseline: the axis-aligned crop of the box, resized (principal point off-centre)."""
    x0, y0, w, h = (int(v) for v in box)
    return cv2.resize(image[y0:y0 + h, x0:x0 + w], (int(out_hw[1]), int(out_hw[0])), interpolation=cv2.INTER_AREA)


def box_points(box, out_hw) -> np.ndarray:
    """(H_z * W_z, 2) evaluation points: a regular grid of continuous source coordinates in the box."""
    x0, y0, w, h = (float(v) for v in box)
    Hz, Wz = int(out_hw[0]), int(out_hw[1])
    xx, yy = np.meshgrid(x0 + (np.arange(Wz) + 0.5) * (w / Wz), y0 + (np.arange(Hz) + 0.5) * (h / Hz))
    return np.stack([xx.ravel(), yy.ravel()], 1)


def sample(m: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear sample of a map (h, w) at continuous coordinates (P, 2); NaN outside the map."""
    m = np.asarray(m, np.float64)
    h, w = m.shape
    x, y = xy[:, 0] - 0.5, xy[:, 1] - 0.5
    inside = (x >= -0.5) & (x <= w - 0.5) & (y >= -0.5) & (y <= h - 0.5)
    x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
    xa, ya = np.floor(x).astype(int), np.floor(y).astype(int)
    xb, yb = np.minimum(xa + 1, w - 1), np.minimum(ya + 1, h - 1)
    wx, wy = x - xa, y - ya
    v = (m[ya, xa] * (1 - wx) * (1 - wy) + m[ya, xb] * wx * (1 - wy) + m[yb, xa] * (1 - wx) * wy
         + m[yb, xb] * wx * wy)
    return np.where(inside, v, np.nan)


def nearest(m: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Value of the pixel containing each continuous coordinate (GT lookup)."""
    h, w = m.shape
    xi = np.clip(np.floor(xy[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.floor(xy[:, 1]).astype(int), 0, h - 1)
    return m[yi, xi]


def source_rays(K0: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """(P, 3) source-camera ray directions with unit z of continuous source coordinates."""
    return np.c_[pts, np.ones(len(pts))] @ np.linalg.inv(np.asarray(K0, np.float64)).T


def depth_from_global(depth: np.ndarray, orig_hw, pts: np.ndarray) -> np.ndarray:
    """Source-view depth predicted on the whole (resized) image, at full-resolution points."""
    h, w = depth.shape
    return sample(depth, pts * np.array([w / orig_hw[1], h / orig_hw[0]]))


def depth_from_crop(depth: np.ndarray, box, pts: np.ndarray) -> np.ndarray:
    """Depth of the naive crop view (same orientation as the source: z is the source z)."""
    x0, y0, w, h = (float(v) for v in box)
    hc, wc = depth.shape
    return sample(depth, np.stack([(pts[:, 0] - x0) * wc / w, (pts[:, 1] - y0) * hc / h], 1))


def depth_from_zoom(depth: np.ndarray, zv: ZoomView, K0: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Zoom-view depth converted to source-camera z at the source points: a point on the source ray
    d0 (unit z) at source depth t has zoom depth t * (R_v^T d0)_z, so t = z_zoom / (R_v^T d0)_z."""
    dz = source_rays(K0, pts) @ zv.R_v
    pz = dz @ zv.K.T
    z = sample(depth, pz[:, :2] / pz[:, 2:3])
    return np.where(dz[:, 2] > 1e-9, z / dz[:, 2], np.nan)


def zoom_w2c(w2c_src: np.ndarray, R_v: np.ndarray) -> np.ndarray:
    """World-to-camera of the zoom view: X_zoom = R_v^T X_src."""
    T = np.eye(4)
    T[:3, :4] = np.asarray(w2c_src, np.float64)[:3, :4]
    Rz = np.eye(4)
    Rz[:3, :3] = np.asarray(R_v, np.float64).T
    return Rz @ T


REGION_SELECTORS = ("oracle", "random", "texture")


def _grad_energy(m: np.ndarray, valid: np.ndarray):
    """Sum of |horizontal| + |vertical| differences between valid neighbours, and their count."""
    e = np.zeros_like(m)
    c = np.zeros_like(m)
    vx, vy = valid[:, 1:] & valid[:, :-1], valid[1:] & valid[:-1]
    e[:, :-1] += np.abs(np.diff(m, axis=1)) * vx
    e[:-1] += np.abs(np.diff(m, axis=0)) * vy
    c[:, :-1] += vx
    c[:-1] += vy
    return e, c


def select_regions(gt: np.ndarray, zoom: float = 4.0, n: int = 1, min_valid: float = 0.6, step: float = 0.5,
                   coarse: int = 8, mode: str = "oracle", image: np.ndarray | None = None,
                   seed: int = 0) -> list[tuple[int, int, int, int]]:
    """Boxes of 1/zoom of the image side, greedy and non-overlapping, among the candidates with at
    least `min_valid` valid GT pixels (needed to evaluate a region at all). Ranking:
    * "oracle" (uses GT): valid fraction x mean |grad log GT depth|, i.e. geometric detail, where
      resolution should matter most; an upper-bound choice;
    * "texture" (deployable, image only): mean |grad| of the grey image;
    * "random" (deployable, unbiased): uniformly random order (seeded)."""
    if mode not in REGION_SELECTORS:
        raise ValueError(f"mode must be one of {REGION_SELECTORS}")
    H0, W0 = gt.shape
    g = gt[coarse // 2::coarse, coarse // 2::coarse]
    valid = g > 0
    if mode == "oracle":
        edge, cnt = _grad_energy(np.log(np.where(valid, g, 1.0)), valid)
    elif mode == "texture":
        if image is None:
            raise ValueError("texture selection needs the image")
        grey = np.asarray(image, np.float64).mean(axis=-1) if image.ndim == 3 else np.asarray(image, np.float64)
        grey = grey[coarse // 2::coarse, coarse // 2::coarse][: g.shape[0], : g.shape[1]]
        edge, cnt = _grad_energy(grey, np.ones_like(grey, bool))
    bw, bh = int(round(W0 / zoom)), int(round(H0 / zoom))
    cands = []
    for y0 in range(0, H0 - bh + 1, max(1, int(bh * step))):
        for x0 in range(0, W0 - bw + 1, max(1, int(bw * step))):
            ys, xs = slice(y0 // coarse, (y0 + bh) // coarse), slice(x0 // coarse, (x0 + bw) // coarse)
            vf = float(valid[ys, xs].mean()) if valid[ys, xs].size else 0.0
            if vf < min_valid:
                continue
            if mode == "random":
                score = 0.0
            else:
                c = cnt[ys, xs].sum()
                score = float(edge[ys, xs].sum() / c) if c > 0 else 0.0
                score *= vf if mode == "oracle" else 1.0
            cands.append((score, (x0, y0, bw, bh)))
    if mode == "random":
        order = np.random.default_rng(seed).permutation(len(cands))
        cands = [cands[i] for i in order]
    else:
        cands = sorted(cands, key=lambda t: -t[0])
    chosen = []
    for _, box in cands:
        if all(box[0] + box[2] <= b[0] or b[0] + b[2] <= box[0] or box[1] + box[3] <= b[1] or b[1] + b[3] <= box[1]
               for b in chosen):
            chosen.append(box)
        if len(chosen) == n:
            break
    return chosen
