"""Convert TartanAir trajectories to the canonical training format (src/da3/data/canonical.py).

Native TartanAir layout (also used by most Kaggle mirrors):
  <traj>/image_left/000000_left.png
  <traj>/depth_left/000000_left_depth.npy      float32 metres
  <traj>/pose_left.txt                         tx ty tz qx qy qz qw per line (NED: x fwd, y right, z down)
Intrinsics are fixed: fx = fy = 320, cx = 320, cy = 240 (640 x 480).

Any directory under --raw that contains image_left/ + depth_left/ + pose_left.txt is treated as one
trajectory. Paper App. A data processing is applied: the `amusement` environment is skipped and the
depth of carwelding / hospital / ocean / office / office2 is clipped at 80 / 30 / 1000 / 30 / 30 m
(clipped pixels are marked invalid). Depth >= --max-depth (sky) is invalid as well.

  python scripts/prepare_tartanair.py --raw /kaggle/input/<dataset> --out /tmp/canonical/tartanair --stride 2
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from da3.data import write_scene

K_TARTAN = np.array([[320.0, 0, 320.0], [0, 320.0, 240.0], [0, 0, 1]])
# OpenCV camera axes expressed in TartanAir's NED body axes: body = M @ cv
NED_FROM_CV = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
APPENDIX_A_CLIP = {"carwelding": 80.0, "hospital": 30.0, "ocean": 1000.0, "office": 30.0, "office2": 30.0}
APPENDIX_A_SKIP = {"amusement"}


def find_trajectories(raw: Path):
    return sorted(p.parent for p in raw.rglob("pose_left.txt")
                  if (p.parent / "image_left").is_dir() and (p.parent / "depth_left").is_dir())


def env_name(traj: Path, raw: Path) -> str:
    return traj.relative_to(raw).parts[0].lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=400, help="per trajectory, after striding")
    ap.add_argument("--max-depth", type=float, default=1000.0)
    ap.add_argument("--max-side", type=int, default=640)
    args = ap.parse_args()
    raw = Path(args.raw)
    trajs = find_trajectories(raw)
    print(f"found {len(trajs)} trajectories")
    total = 0
    for traj in trajs:
        env = env_name(traj, raw)
        if env in APPENDIX_A_SKIP:
            print("skip (App. A):", traj)
            continue
        clip = APPENDIX_A_CLIP.get(env, args.max_depth)
        poses = np.loadtxt(traj / "pose_left.txt")
        imgs = sorted((traj / "image_left").glob("*_left.png"))
        frames = []
        for i in range(0, min(len(imgs), len(poses)), args.stride)[: args.max_frames]:
            stem = imgs[i].name.replace("_left.png", "")
            dpath = traj / "depth_left" / f"{stem}_left_depth.npy"
            if not dpath.exists():
                continue
            rgb = cv2.cvtColor(cv2.imread(str(imgs[i])), cv2.COLOR_BGR2RGB)
            depth = np.load(dpath).astype(np.float32)
            depth = np.where(depth < min(clip, args.max_depth), depth, 0)
            t, q = poses[i, :3], poses[i, 3:7]
            c2w = np.eye(4)
            c2w[:3, :3] = Rotation.from_quat(q).as_matrix() @ NED_FROM_CV
            c2w[:3, 3] = t
            frames.append({"name": stem, "rgb": rgb, "depth": depth, "K": K_TARTAN, "c2w": c2w})
        if frames:
            name = "_".join(traj.relative_to(raw).parts)
            n = write_scene(Path(args.out) / name, frames, depth_unit=0.02, max_side=args.max_side)
            total += n
            print(f"{name}: {n} frames")
    print("total frames:", total)


if __name__ == "__main__":
    main()
