"""Exact image geometry of enhancement #4 (lens-consistent DA3).

Every operation between the dataset image and the tensor DA3 sees (crop, zoom, resize, pad) is an
axis-aligned affine map of continuous pixel coordinates (pixel i covers [i, i + 1), its centre is at
i + 0.5; DA3's intrinsics use the same convention, a centred principal point is at (W / 2, H / 2)).
Composing these maps carries intrinsics and depth maps between frames with no approximation beyond
the final nearest-neighbour resampling of the depth (the released harness resamples depth with
nearest neighbours too).

* `AxisAffine`: u' = ax u + bx, v' = ay v + by. `a.then(b)` = first a, then b.
* `transform_K(K, A)`: intrinsics of the same camera after the image is mapped by A (K' = T_A K).
* `warp_to(src, A, dst_hw)`: resample a map living in frame "src" onto frame "dst", given A: dst -> src.
* `Capture`: how one *photo* is made from the dataset image: a crop box, then a resize. A zoomed
  photo is a centred crop of 1/z of each side resized back to the full size (focal length x z).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

KIND_NORMAL, KIND_ZOOM, KIND_PORTRAIT = 0, 1, 2
KIND_NAMES = {KIND_NORMAL: "normal", KIND_ZOOM: "zoom", KIND_PORTRAIT: "portrait"}
MEAN_RGB = (124, 116, 104)   # ImageNet mean in uint8: padding with it gives ~0 after normalisation


@dataclass(frozen=True)
class AxisAffine:
    """u' = ax * u + bx, v' = ay * v + by (continuous pixel coordinates)."""

    ax: float = 1.0
    bx: float = 0.0
    ay: float = 1.0
    by: float = 0.0

    def then(self, other: "AxisAffine") -> "AxisAffine":
        """First self, then other."""
        return AxisAffine(other.ax * self.ax, other.ax * self.bx + other.bx,
                          other.ay * self.ay, other.ay * self.by + other.by)

    def inverse(self) -> "AxisAffine":
        return AxisAffine(1.0 / self.ax, -self.bx / self.ax, 1.0 / self.ay, -self.by / self.ay)

    def apply(self, u, v):
        return self.ax * np.asarray(u, np.float64) + self.bx, self.ay * np.asarray(v, np.float64) + self.by

    def matrix(self) -> np.ndarray:
        """3x3 homogeneous matrix T with [u', v', 1] = T [u, v, 1]."""
        return np.array([[self.ax, 0.0, self.bx], [0.0, self.ay, self.by], [0.0, 0.0, 1.0]])

    def is_identity(self, tol: float = 1e-9) -> bool:
        return (abs(self.ax - 1.0) < tol and abs(self.ay - 1.0) < tol
                and abs(self.bx) < tol and abs(self.by) < tol)

    def to_list(self) -> list:
        return [float(self.ax), float(self.bx), float(self.ay), float(self.by)]

    # ---- elementary maps
    @staticmethod
    def resize(src_hw, dst_hw) -> "AxisAffine":
        """Image of size src_hw resized to dst_hw (both (h, w))."""
        return AxisAffine(float(dst_hw[1]) / float(src_hw[1]), 0.0, float(dst_hw[0]) / float(src_hw[0]), 0.0)

    @staticmethod
    def crop(x0: float, y0: float) -> "AxisAffine":
        """Crop whose top-left corner is at pixel (x0, y0) of the source."""
        return AxisAffine(1.0, -float(x0), 1.0, -float(y0))

    @staticmethod
    def pad(px: float, py: float) -> "AxisAffine":
        """Paste into a canvas at offset (px, py)."""
        return AxisAffine(1.0, float(px), 1.0, float(py))


def transform_K(K: np.ndarray, A: AxisAffine) -> np.ndarray:
    """Intrinsics (3, 3) of the same camera after its image is mapped by A: u' = ax (fx X/Z + cx) + bx."""
    return A.matrix() @ np.asarray(K, np.float64)


def warp_to(src: np.ndarray, A_dst_to_src: AxisAffine, dst_hw, content: np.ndarray | None = None,
            interp: int = cv2.INTER_NEAREST) -> np.ndarray:
    """Resample the map `src` (h, w) onto a (dst_h, dst_w) grid. A_dst_to_src maps continuous dst
    coordinates to continuous src coordinates. Pixels that land outside src, or outside `content`
    (a 0/1 mask in src, e.g. the photo inside a padded canvas), get 0 (= invalid depth). With an
    identity map and equal sizes the map is returned unchanged."""
    src = np.asarray(src)
    if content is None and A_dst_to_src.is_identity() and tuple(src.shape[:2]) == tuple(dst_hw):
        return src
    s = src.astype(np.float32)
    if content is not None:
        s = s * np.asarray(content, np.float32)
    a = A_dst_to_src
    # OpenCV pixel indices have centres at integers: i_src = ax (i_dst + 0.5) + bx - 0.5
    M = np.array([[a.ax, 0.0, a.ax * 0.5 + a.bx - 0.5], [0.0, a.ay, a.ay * 0.5 + a.by - 0.5]], np.float64)
    return cv2.warpAffine(s, M, (int(dst_hw[1]), int(dst_hw[0])), flags=interp | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


# ----------------------------------------------------------------------------- photos from dataset images
@dataclass(frozen=True)
class Capture:
    """One photo made from a dataset image of size `src_hw`: crop box (x0, y0, w, h), then resize to
    `size` (h, w). focal_scale = photo pixels per source pixel (x axis); for a zoom it is the focal
    multiplier z."""

    src_hw: tuple
    box: tuple
    size: tuple
    kind: int = KIND_NORMAL

    @property
    def focal_scale(self) -> float:
        return float(self.size[1]) / float(self.box[2])

    @property
    def zoom(self) -> float:
        """Focal length of the photo / focal length of the source camera (x axis)."""
        return self.focal_scale

    def affine(self) -> AxisAffine:
        """Source image -> photo."""
        x0, y0, w, h = self.box
        return AxisAffine.crop(x0, y0).then(AxisAffine.resize((h, w), self.size))

    def is_identity(self) -> bool:
        return tuple(self.box) == (0, 0, int(self.src_hw[1]), int(self.src_hw[0])) and \
            tuple(self.size) == tuple(int(t) for t in self.src_hw)

    def apply(self, img: np.ndarray) -> np.ndarray:
        if self.is_identity():
            return img
        x0, y0, w, h = self.box
        crop = img[y0:y0 + h, x0:x0 + w]
        if (h, w) == tuple(self.size):
            return np.ascontiguousarray(crop)
        up = self.size[0] * self.size[1] > h * w
        return cv2.resize(crop, (int(self.size[1]), int(self.size[0])),
                          interpolation=cv2.INTER_CUBIC if up else cv2.INTER_AREA)


def identity_capture(hw) -> Capture:
    h, w = int(hw[0]), int(hw[1])
    return Capture(src_hw=(h, w), box=(0, 0, w, h), size=(h, w), kind=KIND_NORMAL)


def _centred_side(n: int, target: float) -> int:
    """Whole-pixel crop side closest to `target` with an even margin n - side, so the crop is exactly
    centred (its principal point stays at the image centre)."""
    b = int(round(target))
    if (n - b) % 2:
        b += 1 if target > b else -1
    return int(min(n, max(2, b)))


def zoom_capture(hw, z: float) -> Capture:
    """Centred crop of ~1/z of each side, resized back to the full size (focal length x z, principal
    point exactly centred). The box is whole pixels; the exact factors live in the affine (zoom)."""
    h, w = int(hw[0]), int(hw[1])
    if z <= 1.0:
        raise ValueError(f"zoom factor must be > 1, got {z}")
    bw, bh = _centred_side(w, w / z), _centred_side(h, h / z)
    return Capture(src_hw=(h, w), box=((w - bw) // 2, (h - bh) // 2, bw, bh), size=(h, w), kind=KIND_ZOOM)


def portrait_capture(hw, aspect: float = 3.0 / 4.0) -> Capture:
    """Centred portrait crop (width = aspect x height) of a landscape image, not resized. On
    benchmark images this also narrows the field of view (a real portrait photo cannot be made from
    a landscape frame); see docs/LENS.md."""
    h, w = int(hw[0]), int(hw[1])
    bw = _centred_side(w, min(w, h * aspect))
    return Capture(src_hw=(h, w), box=((w - bw) // 2, 0, bw, h), size=(h, bw), kind=KIND_PORTRAIT)
