"""Image loading / preprocessing shared by inference, parity and evaluation scripts."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def list_images(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS)


def read_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _nearest_multiple(x: int, p: int) -> int:
    down = (x // p) * p
    return max(p, down + p if abs(down + p - x) <= abs(x - down) else down)


def target_size(h: int, w: int, process_res: int = 504, patch: int = 14) -> tuple[int, int]:
    """Final size of the released API default ('upper_bound_resize')."""
    s = process_res / max(h, w)
    nh, nw = (h, w) if max(h, w) == process_res else (max(1, round(h * s)), max(1, round(w * s)))
    return _nearest_multiple(nh, patch), _nearest_multiple(nw, patch)


def preprocess_like_official(img: np.ndarray, process_res: int = 504, patch: int = 14) -> np.ndarray:
    """Two-step resize identical to the released InputProcessor ('upper_bound_resize'):
    longest side -> process_res, then each side to the nearest multiple of 14 (ties up)."""
    h, w = img.shape[:2]
    if max(h, w) != process_res:
        s = process_res / max(h, w)
        nh, nw = max(1, round(h * s)), max(1, round(w * s))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    h, w = img.shape[:2]
    nh, nw = _nearest_multiple(h, patch), _nearest_multiple(w, patch)
    if (nh, nw) != (h, w):
        up = nh > h or nw > w
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC if up else cv2.INTER_AREA)
    return img


def resize_rgb(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if hw[0] * hw[1] < h * w else cv2.INTER_CUBIC
    return cv2.resize(img, (hw[1], hw[0]), interpolation=interp)


def scale_intrinsics(K: np.ndarray, src_hw, dst_hw) -> np.ndarray:
    K = np.array(K, dtype=np.float64, copy=True)
    sy, sx = dst_hw[0] / src_hw[0], dst_hw[1] / src_hw[1]
    K[..., 0, :] *= sx
    K[..., 1, :] *= sy
    return K


def load_views(paths, process_res: int = 504, hw: tuple[int, int] | None = None):
    """Load N images and bring them to one common size. Default: official preprocessing, then
    center-crop to the smallest size if the images had different aspect ratios (as released).
    Returns (uint8 (N,H,W,3), original sizes)."""
    imgs = [read_rgb(p) for p in paths]
    sizes = [im.shape[:2] for im in imgs]
    if hw is not None:
        return np.stack([resize_rgb(im, hw) for im in imgs]), sizes
    out = [preprocess_like_official(im, process_res) for im in imgs]
    mh, mw = min(o.shape[0] for o in out), min(o.shape[1] for o in out)
    # same offsets as torchvision CenterCrop
    out = [o[int(round((o.shape[0] - mh) / 2.0)) :][:mh, int(round((o.shape[1] - mw) / 2.0)) :][:, :mw]
           for o in out]
    return np.stack(out), sizes


def to_model_input(rgb_uint8: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (1, N, 3, H, W) ImageNet-normalized float tensor."""
    from da3.model import normalize_images

    x = torch.from_numpy(rgb_uint8).float().div(255.0).permute(0, 3, 1, 2)
    return normalize_images(x)[None]
