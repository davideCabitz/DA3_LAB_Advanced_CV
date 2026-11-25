"""scripts/compare_reconstructions.py: a reconstruction compared with a Sim(3)-transformed copy of itself."""

import json
import runpy
import sys
from pathlib import Path

import cv2
import numpy as np

from da3.export import write_ply

REPO = Path(__file__).resolve().parents[1]


def _folder(d: Path, c2w, depth, pts, rgb):
    for sub in ("depth", "conf", "images"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(len(c2w)):
        np.save(d / "depth" / f"v{i}.npy", depth[i].astype(np.float32))
        np.save(d / "conf" / f"v{i}.npy", np.full(depth[i].shape, 2.0, np.float32))
        cv2.imwrite(str(d / "images" / f"v{i}.jpg"), np.full((*depth[i].shape, 3), 128, np.uint8))
        frames.append({"file": f"v{i}.jpg", "used": {"w2c": np.linalg.inv(c2w[i])[:3].tolist()}})
    (d / "cameras.json").write_text(json.dumps({"image_size_hw": list(depth.shape[1:]), "frames": frames}))
    write_ply(d / "points.ply", pts, rgb)


def test_identical_up_to_sim3_gives_zero_difference(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    c2w = np.tile(np.eye(4), (4, 1, 1))
    c2w[:, 0, 3] = [0, 1, 2, 3.5]
    c2w[:, 2, 3] = [0, 0.3, -0.2, 0.1]
    depth = rng.uniform(2, 8, (4, 20, 30))
    pts = rng.uniform(-3, 3, (5000, 3)) + [1.5, 0, 6]
    rgb = rng.integers(0, 255, (5000, 3), dtype=np.uint8)
    _folder(tmp_path / "ref", c2w, depth, pts, rgb)
    s = 0.4
    R = cv2.Rodrigues(np.array([[0.3], [-0.2], [0.5]]))[0]
    t = np.array([2.0, -1.0, 0.5])
    c2w_t = c2w.copy()
    c2w_t[:, :3, :3] = R @ c2w[:, :3, :3]
    c2w_t[:, :3, 3] = (s * (R @ c2w[:, :3, 3].T)).T + t
    _folder(tmp_path / "test", c2w_t, depth * s, (s * (R @ pts.T)).T + t, rgb)
    monkeypatch.setattr(sys, "argv", ["compare_reconstructions.py", "--ref", str(tmp_path / "ref"),
                                      "--test", f"same={tmp_path / 'test'}", "--out", str(tmp_path / "cmp")])
    runpy.run_path(str(REPO / "scripts" / "compare_reconstructions.py"), run_name="__main__")
    m = json.loads((tmp_path / "cmp" / "compare.json").read_text())["tests"]["same"]
    assert m["depth_reldiff_median"] < 1e-5 and m["rot_diff_deg_max"] < 1e-3 and m["f1"] > 0.999
    assert m["chamfer_rel_to_size"] < 1e-5 and abs(m["sim3_scale"] - 1 / s) < 1e-6
    assert (tmp_path / "cmp" / "side_by_side.mp4").stat().st_size > 0
    assert (tmp_path / "cmp" / "aligned" / "same.ply").exists() and (tmp_path / "cmp" / "depth_diff_same.png").exists()
