"""Reference reconstruction with the OFFICIAL DA3 code and any released model, exported exactly like the
online demo (scene.glb: coloured points + camera wireframes). Default model: the demo's own,
DA3NESTED-GIANT-LARGE-1.1 (1.40B parameters, CC BY-NC 4.0).

  python scripts/official_reference.py --official-src /kaggle/working/Depth-Anything-3/src \
      --images /kaggle/input/<dataset>/<folder> --out /kaggle/working/demo/official_<name>

REFERENCE ONLY. It is a different code path with different (larger) weights, so it is never the baseline of a
"released DA3 vs enhancement" comparison. Those always run the same weights through our code (da3-small / base
/ large: scripts/infer.py, scripts/lens_demo.py). Use it to show what the largest released model gives on the
same photos, labelled as such.

Writes <out>/scene.glb (+ the official depth visualisations), <out>/cameras.json (the model's own cameras,
processed-image pixels) and <out>/reference.json (model, settings, time).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--official-src", required=True, help="src/ of the official Depth-Anything-3 clone")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--conf-percentile", type=float, default=40.0, help="the official GLB export's default (40)")
    ap.add_argument("--max-points", type=int, default=1_000_000, help="the official GLB export's default")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.official_src).resolve()))
    sys.path.insert(0, str(REPO / "src"))
    from depth_anything_3.api import DepthAnything3

    from da3.io_utils import list_images

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        # T4 / P100: the API picks bfloat16 whenever torch reports it (emulated, slow); use real float16
        torch.cuda.is_bf16_supported = lambda *a, **k: False
    paths = [str(p) for p in list_images(args.images)]
    if len(paths) < 2:
        raise SystemExit(f"need >= 2 images in {args.images}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[official] {args.model} on {len(paths)} photos (reference only, official code)")
    model = DepthAnything3.from_pretrained(args.model).to(args.device).eval()
    t0 = time.time()
    with torch.no_grad():
        pred = model.inference(paths, process_res=args.process_res, export_dir=str(out), export_format="glb",
                               conf_thresh_percentile=args.conf_percentile, num_max_points=args.max_points,
                               show_cameras=True)
    dt = time.time() - t0
    frames = []
    ext, ixt = getattr(pred, "extrinsics", None), getattr(pred, "intrinsics", None)
    if ext is not None and ixt is not None:
        for p, e, k in zip(paths, np.asarray(ext), np.asarray(ixt)):
            frames.append({"file": Path(p).name, "w2c": np.asarray(e, np.float64).tolist(), "K": np.asarray(k, np.float64).tolist()})
    (out / "cameras.json").write_text(json.dumps({"frames": frames, "pixels": "processed image",
                                                  "source": f"{args.model} camera head (official code)"}, indent=1))
    meta = {"model": args.model, "code": "official DA3 API (pinned clone)", "images": len(paths),
            "process_res": args.process_res, "conf_thresh_percentile": args.conf_percentile,
            "num_max_points": args.max_points, "time_s": dt,
            "note": "REFERENCE ONLY: different code and weights from our comparisons; license CC BY-NC 4.0 for "
                    "the Giant / Nested models"}
    (out / "reference.json").write_text(json.dumps(meta, indent=2))
    glb = out / "scene.glb"
    print(f"[official] done in {dt:.1f} s -> {glb if glb.exists() else out}")
    if args.zip:
        print("package:", shutil.make_archive(str(out), "zip", root_dir=out))


if __name__ == "__main__":
    main()
