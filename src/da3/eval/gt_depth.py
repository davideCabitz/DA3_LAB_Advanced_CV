"""Per-frame ground-truth depth of the DA3-BENCH datasets, decoded like the released loaders.

Depth in metres, 0 = invalid. File locations come from the released dataset objects
(`depth_anything_3.bench.registries.MV_REGISTRY.get(data)().get_data(scene)`), passed in by the
caller (src/da3 never imports the official package):

* HiRoom: 16-bit PNG, value / 65535 * 100 m; the aliasing mask (> 0) marks pixels to exclude;
* 7Scenes: 16-bit PNG in mm, 65535 = invalid. It is Kinect depth, NOT registered to the RGB camera,
  so per-pixel errors include that offset (compare methods with each other, not with HiRoom);
* ETH3D: raw float32 at the full DSLR resolution (ground_truth_depth/dslr_images/<image>), inf =
  invalid; masks_for_images/dslr_images/<image>.png == 1 marks occluded / invalid pixels. The maps
  belong to the distorted JPGs; like the released harness we treat those images as pinhole.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

GT_DEPTH_DATASETS = ("hiroom", "7scenes", "eth3d")


def eth3d_gt_paths(image_file: str) -> tuple[str, str]:
    """(depth, mask) paths of an ETH3D image <scene>/images/dslr_images/<name>.JPG."""
    scene_dir, rel = image_file.replace("\\", "/").split("/images/", 1)
    name = os.path.basename(rel)
    return (os.path.join(scene_dir, "ground_truth_depth", "dslr_images", name),
            os.path.join(scene_dir, "masks_for_images", "dslr_images", os.path.splitext(name)[0] + ".png"))


def gt_depth_index(data: str, scene_data) -> dict:
    """{image_file: (depth_path, mask_path | None, (h, w) | None)} for one scene (released loader output)."""
    files = [str(f) for f in scene_data.image_files]
    aux = scene_data.aux
    if data == "hiroom":
        alias = list(aux.get("aliasing_mask_files", [None] * len(files)))
        return {f: (d, a, None) for f, d, a in zip(files, aux.gt_depth_files, alias)}
    if data == "7scenes":
        return {f: (d, None, None) for f, d in zip(files, aux.gt_depth_files)}
    if data == "eth3d":
        hs, ws = aux.get("heights", []), aux.get("widths", [])
        out = {}
        for i, f in enumerate(files):
            d, m = eth3d_gt_paths(f)
            shape = (int(hs[i]), int(ws[i])) if i < len(hs) else None
            out[f] = (d, m, shape)
        return out
    raise ValueError(f"no per-frame GT depth for {data}")


def read_gt_depth(data: str, path: str, mask_path: str | None = None, shape=None, hw=None) -> np.ndarray:
    """GT depth (metres, invalid = 0) of one frame; resized (nearest) to hw = (H, W) if given.
    shape: (h, w) of the raw ETH3D map (required for ETH3D). Missing file -> all zeros (or None if
    neither hw nor shape is known)."""
    if not path or not os.path.exists(path):
        size = hw or shape
        return np.zeros(size, np.float32) if size is not None else None
    if data == "eth3d":
        if shape is None:
            raise ValueError("ETH3D raw depth needs the image shape")
        d = np.fromfile(path, dtype=np.float32).reshape(shape)
        d = np.where(np.isfinite(d) & (d > 0), d, 0.0).astype(np.float32)
        if mask_path and os.path.exists(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None and m.shape == d.shape:
                d[m == 1] = 0.0
    else:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            size = hw or shape
            return np.zeros(size, np.float32) if size is not None else None
        raw = raw.astype(np.float64)
        if raw.ndim == 3:
            raw = raw[..., 0]
        if data == "hiroom":
            d = raw / 65535.0 * 100.0
            if mask_path and os.path.exists(mask_path):
                m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if m is not None:
                    d[(m if m.ndim == 2 else m[..., 0]) > 0] = 0.0
        elif data == "7scenes":
            raw[raw == 65535] = 0
            d = raw / 1000.0
        else:
            raise ValueError(f"no GT depth decoder for {data}")
        d = d.astype(np.float32)
    if hw is not None and tuple(d.shape) != tuple(hw):
        d = cv2.resize(d, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return d
