"""Convert a Virtual KITTI 2 subset to the canonical training format (src/da3/data/canonical.py).

Raw layout (official vkitti_2.0.3 tars; extract only what you need, e.g. with
`curl -sL <url> | tar -x --wildcards '*Scene01/clone/frames/rgb/Camera_0/*'`):
  <raw>/SceneXX/<variant>/frames/rgb/Camera_0/rgb_00000.jpg
  <raw>/SceneXX/<variant>/frames/depth/Camera_0/depth_00000.png   uint16 centimetres, 65535 = sky
  <raw>/SceneXX/<variant>/{intrinsic,extrinsic}.txt               (textgt tar)
extrinsic.txt rows are world-to-camera matrices in the OpenCV camera convention.

  python scripts/prepare_vkitti2.py --raw /tmp/vkitti2_raw --out /tmp/canonical/vkitti2 \
      --scenes Scene01 Scene02 Scene06 --variants clone --stride 2
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from da3.data import write_scene


def read_table(path):
    rows = [l.split() for l in Path(path).read_text().splitlines()[1:] if l.strip()]
    return np.array(rows, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", nargs="+", default=["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"])
    ap.add_argument("--variants", nargs="+", default=["clone"])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--stride", type=int, default=2, help="keep every k-th frame")
    ap.add_argument("--max-side", type=int, default=1008)
    args = ap.parse_args()

    total = 0
    for scene in args.scenes:
        for var in args.variants:
            base = Path(args.raw) / scene / var
            intr, extr = read_table(base / "intrinsic.txt"), read_table(base / "extrinsic.txt")
            intr = intr[intr[:, 1] == args.camera]
            extr = extr[extr[:, 1] == args.camera]
            frames = []
            for row_i, row_e in zip(intr[:: args.stride], extr[:: args.stride]):
                f = int(row_i[0])
                rgb_p = base / "frames" / "rgb" / f"Camera_{args.camera}" / f"rgb_{f:05d}.jpg"
                dep_p = base / "frames" / "depth" / f"Camera_{args.camera}" / f"depth_{f:05d}.png"
                if not (rgb_p.exists() and dep_p.exists()):
                    continue
                rgb = cv2.cvtColor(cv2.imread(str(rgb_p)), cv2.COLOR_BGR2RGB)
                d = cv2.imread(str(dep_p), cv2.IMREAD_UNCHANGED).astype(np.float32)
                depth = np.where(d >= 65535, 0, d / 100.0)               # cm -> m, sky invalid
                K = np.array([[row_i[2], 0, row_i[4]], [0, row_i[3], row_i[5]], [0, 0, 1]])
                w2c = row_e[2:].reshape(4, 4)
                frames.append({"name": f"{f:05d}", "rgb": rgb, "depth": depth, "K": K, "c2w": np.linalg.inv(w2c)})
            if frames:
                n = write_scene(Path(args.out) / f"{scene}_{var}", frames, depth_unit=0.01, max_side=args.max_side)
                total += n
                print(f"{scene}/{var}: {n} frames")
    print("total frames:", total)


if __name__ == "__main__":
    main()
