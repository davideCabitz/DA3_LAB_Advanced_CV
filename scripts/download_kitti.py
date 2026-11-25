"""Download only the KITTI raw drives the pre-registered split needs (after registering at cvlibs.net and
accepting the KITTI terms; the URLs are the ones of KITTI's own raw_data_downloader.sh).

1. Candidate drives: the Eigen TEST drives (monodepth2 splits/eigen/test_files.txt) and the Eigen TRAIN
   drives (splits/eigen_full/train_files.txt) - the latter are disjoint from the test drives by construction.
2. Size rule (fixed before any result, plan v2): drives whose synced zip is larger than --max-gb are
   dropped (HEAD request), so one session can hold them.
3. The split is drawn with configs/lora_analysis.yaml (n_test / n_val / n_train, split_seed) over the
   remaining candidates -> <out>/planned_split.json; prepare_kitti.py uses it with --split-from.
4. Only those drives (+ the calibration of their dates) are downloaded; from each zip only image_02,
   oxts and velodyne_points are extracted; the zip is deleted. Resumable (one marker per drive).

  python scripts/download_kitti.py --out /tmp/kitti_raw --eigen-test-list test_files.txt \
      --eigen-train-list train_files.txt
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import yaml

from da3.data.kitti import drive_id, make_split, read_eigen_test_drives

BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
KEEP = ("/image_02/", "/oxts/", "/velodyne_points/")
REPO = Path(__file__).resolve().parents[1]


def zip_url(d: str) -> str:
    return f"{BASE}/{d}/{d}_sync.zip"


def calib_url(date: str) -> str:
    return f"{BASE}/{date}_calib.zip"


def size_gb(url: str) -> float:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers.get("Content-Length", 0)) / 2**30
    except Exception:
        return float("nan")


def fetch(url: str, dst: Path):
    tmp = dst.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=16 << 20)
    tmp.replace(dst)


def extract(zpath: Path, out: Path, keep=KEEP):
    with zipfile.ZipFile(zpath) as z:
        for m in z.namelist():
            if keep is None or any(k in "/" + m for k in keep) or m.endswith("timestamps.txt"):
                z.extract(m, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--eigen-test-list", required=True)
    ap.add_argument("--eigen-train-list", required=True)
    ap.add_argument("--plan", default=str(REPO / "configs" / "lora_analysis.yaml"))
    ap.add_argument("--max-gb", type=float, default=4.0, help="drop drives whose zip is larger (fixed rule)")
    ap.add_argument("--only", nargs="*", default=None, help="download only these roles, e.g. val (go/no-go first)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    k = yaml.safe_load(Path(args.plan).read_text())["kitti"]
    eigen_test = read_eigen_test_drives(args.eigen_test_list)
    eigen_train = read_eigen_test_drives(args.eigen_train_list) - eigen_test   # same line format

    plan_file = out / "planned_split.json"
    if plan_file.exists():
        split = json.loads(plan_file.read_text())
        print("using", plan_file)
    else:
        sizes = {d: size_gb(zip_url(d)) for d in sorted(eigen_test | eigen_train)}
        ok = [d for d, s in sizes.items() if s == s and s <= args.max_gb]
        print(f"{len(sizes)} candidate drives, {len(ok)} with a zip <= {args.max_gb} GB "
              f"(unreachable: {sum(s != s for s in sizes.values())})", flush=True)
        if not ok:
            raise SystemExit("no drive reachable: check the internet connection / the KITTI terms")
        split = make_split(ok, eigen_test, k["n_test_drives"], k["n_val_drives"], k["split_seed"], k.get("n_train_drives"))
        split.update(max_gb=args.max_gb, sizes_gb=sizes)
        plan_file.write_text(json.dumps(split, indent=2))
    roles = args.only or ["val", "test", "train"]
    drives = [d for r in roles for d in split[r]]
    total = sum(split["sizes_gb"].get(d, 0) for d in drives)
    print(f"downloading {len(drives)} drives ({total:.1f} GB of zips): " + ", ".join(f"{r} {len(split[r])}" for r in roles), flush=True)
    for date in sorted({d[:10] for d in drives}):
        if not (out / date / "calib_cam_to_cam.txt").exists():
            z = out / f"{date}_calib.zip"
            fetch(calib_url(date), z)
            extract(z, out, keep=None)
            z.unlink()
    for i, d in enumerate(drives):
        mark = out / f".done_{d}"
        if mark.exists():
            continue
        z = out / f"{d}_sync.zip"
        fetch(zip_url(d), z)
        extract(z, out)
        z.unlink()
        mark.touch()
        print(f"[{i + 1}/{len(drives)}] {d} ({split['sizes_gb'].get(d, 0):.2f} GB zip)", flush=True)
    print("done:", out)


if __name__ == "__main__":
    main()
