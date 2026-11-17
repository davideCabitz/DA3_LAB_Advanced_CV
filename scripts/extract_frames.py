"""Extract N evenly spaced frames from a video into a folder (sample inputs for parity / demo)."""

import argparse
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--start", type=float, default=0.0, help="fraction of the video to skip")
    ap.add_argument("--end", type=float, default=1.0)
    args = ap.parse_args()
    cap = cv2.VideoCapture(args.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lo, hi = int(n * args.start), max(int(n * args.end) - 1, 0)
    idx = [lo + round(i * (hi - lo) / max(args.frames - 1, 1)) for i in range(args.frames)]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"could not read frame {i} of {n}")
        cv2.imwrite(str(out / f"{k:03d}.png"), frame)
    print(f"wrote {len(idx)} frames (of {n}) to {out}")


if __name__ == "__main__":
    main()
