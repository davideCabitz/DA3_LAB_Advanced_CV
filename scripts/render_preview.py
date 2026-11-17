"""Render a quick 3D preview (two viewpoints) of an infer.py output: coloured points + camera frusta.

  python scripts/render_preview.py outputs/demo/<scene>      -> <scene>/preview_3d.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def read_ply(path):
    with open(path, "rb") as f:
        n = 0
        while (line := f.readline().decode().strip()) != "end_header":
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        a = np.frombuffer(f.read(n * dt.itemsize), dtype=dt)
    return np.stack([a["x"], a["y"], a["z"]], 1), np.stack([a["r"], a["g"], a["b"]], 1) / 255.0


def main(d):
    d = Path(d)
    P, C = read_ply(d / "points.ply")
    idx = np.random.default_rng(0).choice(len(P), min(60000, len(P)), replace=False)
    P, C = P[idx], C[idx]
    P = P * np.array([1, -1, -1])  # OpenCV -> y up, z towards viewer (for plotting)
    meta = json.loads((d / "transforms.json").read_text())
    W, H = meta["w"], meta["h"]
    fig = plt.figure(figsize=(14, 6))
    for k, (elev, azim) in enumerate([(20, -60), (80, -90)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.scatter(P[:, 0], P[:, 2], P[:, 1], c=C, s=0.3, depthshade=False)
        scale = 0.1 * np.ptp(P, axis=0).max()
        for fr in meta["frames"]:
            c2w = np.linalg.inv(np.array(fr["w2c_opencv"]))
            K = np.array(fr["K"])
            corners = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]], float) @ np.linalg.inv(K).T * scale
            pts = np.vstack([np.zeros(3), corners]) @ c2w[:3, :3].T + c2w[:3, 3]
            pts = pts * np.array([1, -1, -1])
            for a, b in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]:
                ax.plot(*[[pts[a, j], pts[b, j]] for j in (0, 2, 1)], color="red", lw=0.8)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x"), ax.set_ylabel("-z (depth)"), ax.set_zlabel("-y (up)")
        ax.set_box_aspect(np.ptp(P[:, [0, 2, 1]], axis=0))
    fig.suptitle(f"{d.name}: {len(meta['frames'])} views, cameras in red")
    fig.tight_layout()
    fig.savefig(d / "preview_3d.png", dpi=110)
    print("wrote", d / "preview_3d.png")


if __name__ == "__main__":
    main(sys.argv[1])
