"""Canonical on-disk multi-view format used by all training datasets.

    <root>/<scene>/rgb/<frame>.jpg|png
    <root>/<scene>/depth/<frame>.png      uint16, metres = value * depth_unit, 0 = invalid
    <root>/<scene>/cameras.json           {"depth_unit": 0.001, "frames": [
                                             {"name": "000000", "K": 3x3, "c2w": 4x4 (OpenCV)}, ...]}

Per-dataset converters (scripts/prepare_*.py) write this format from the raw downloads, so the
training loader does not depend on any dataset's native layout.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def write_scene(
    out_dir: str | Path,
    frames: list[dict],
    depth_unit: float,
    max_side: int | None = 640,
    jpeg_quality: int = 95,
):
    """frames: [{"name", "rgb" (H,W,3 uint8 RGB), "depth" (H,W float metres, <=0/nan invalid),
    "K" (3,3), "c2w" (4,4)}]. Images are downscaled so the longest side is <= max_side."""
    out = Path(out_dir)
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    meta = {"depth_unit": depth_unit, "frames": []}
    for f in frames:
        rgb, depth, K = f["rgb"], f["depth"].astype(np.float32), np.array(f["K"], np.float64)
        h, w = rgb.shape[:2]
        if max_side and max(h, w) > max_side:
            s = max_side / max(h, w)
            nw, nh = round(w * s), round(h * s)
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST)
            K[0] *= nw / w
            K[1] *= nh / h
        d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0) / depth_unit
        d = np.where((d > 0) & (d < 65535), np.round(d), 0).astype(np.uint16)
        cv2.imwrite(str(out / "rgb" / f"{f['name']}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        cv2.imwrite(str(out / "depth" / f"{f['name']}.png"), d)
        meta["frames"].append({"name": f["name"], "K": K.tolist(), "c2w": np.asarray(f["c2w"]).tolist()})
    (out / "cameras.json").write_text(json.dumps(meta))
    return len(meta["frames"])


class CanonicalScene:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        meta = json.loads((self.path / "cameras.json").read_text())
        self.depth_unit = float(meta["depth_unit"])
        self.frames = meta["frames"]
        self.K = np.array([f["K"] for f in self.frames], np.float64)
        self.c2w = np.array([f["c2w"] for f in self.frames], np.float64)

    def __len__(self):
        return len(self.frames)

    def load(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        name = self.frames[i]["name"]
        rgb_path = self.path / "rgb" / f"{name}.jpg"
        if not rgb_path.exists():
            rgb_path = rgb_path.with_suffix(".png")
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        d = cv2.imread(str(self.path / "depth" / f"{name}.png"), cv2.IMREAD_UNCHANGED)
        return rgb, d.astype(np.float32) * self.depth_unit


def list_scenes(root: str | Path, min_frames: int = 2) -> list[Path]:
    root = Path(root)
    scenes = sorted(p.parent for p in root.rglob("cameras.json"))
    return [s for s in scenes if len(json.loads((s / "cameras.json").read_text())["frames"]) >= min_frames]
