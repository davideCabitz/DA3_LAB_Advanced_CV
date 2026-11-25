"""Convert KITTI raw drives to the canonical multi-view format + the pre-registered drive split.

Per drive (camera 2 = left colour, rectified, full resolution, no resizing):
  <out>/<drive>/rgb/<frame>.jpg
  <out>/<drive>/depth/<frame>.png      projected LiDAR, uint16, metres = value / 256, 0 = no point
  <out>/<drive>/cameras.json           K and c2w (OpenCV) from calib + OXTS (src/da3/data/kitti.py)
and <out>/split.json: train / val / test drives (configs/lora_analysis.yaml `kitti`), with the
Eigen-test leakage check. The LiDAR depth is for EVALUATION only; LoRA training never reads it.

  python scripts/prepare_kitti.py --raw /kaggle/input/kitti-raw --out /tmp/canonical/kitti \
      --eigen-test-list /tmp/eigen_test_files.txt

The Eigen list is the standard `test_files.txt` (e.g. monodepth2: splits/eigen/test_files.txt).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from da3.data.kitti import (
    camera_poses,
    check_split,
    drive_id,
    kitti_calibration,
    lidar_depth_map,
    make_split,
    oxts_to_poses,
    read_eigen_test_drives,
    read_oxts,
    read_velodyne,
)

DEPTH_UNIT = 1.0 / 256.0
REPO = Path(__file__).resolve().parents[1]


def find_drives(raw: Path) -> list[Path]:
    """One folder per drive (the one holding image_02/data), duplicates of a drive id dropped."""
    found = {}
    for p in sorted(raw.rglob("*_drive_*_sync")):
        if (p / "image_02" / "data").is_dir():
            found.setdefault(drive_id(p.name), p)
    return [found[k] for k in sorted(found)]


def find_calibrations(raw: Path) -> dict[str, Path]:
    """date -> folder with calib_*.txt. Kaggle unpacks each KITTI zip into its own folder, so the
    calibration is often NOT the drive's parent; it is found by its date folder name."""
    out = {}
    for f in raw.rglob("calib_cam_to_cam.txt"):
        out.setdefault(f.parent.name, f.parent)
    return out


def convert_drive(drive_dir: Path, out_dir: Path, calib_dir: Path, stride: int = 1, jpeg_quality: int = 95) -> dict:
    calib = kitti_calibration(calib_dir)
    imgs = sorted((drive_dir / "image_02" / "data").glob("*.png"))
    poses = camera_poses(oxts_to_poses(read_oxts(drive_dir)), calib["T_cam2_imu"])
    if len(poses) != len(imgs):
        raise ValueError(f"{drive_dir.name}: {len(imgs)} images but {len(poses)} OXTS packets")
    (out_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)
    meta = {"depth_unit": DEPTH_UNIT, "source": "kitti_raw", "drive": drive_id(drive_dir.name),
            "depth_kind": "lidar_eval_only", "frames": []}
    n_lidar = 0
    for i in range(0, len(imgs), stride):
        name = imgs[i].stem
        bgr = cv2.imread(str(imgs[i]), cv2.IMREAD_COLOR)
        cv2.imwrite(str(out_dir / "rgb" / f"{name}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        velo = drive_dir / "velodyne_points" / "data" / f"{name}.bin"
        depth = np.zeros(bgr.shape[:2], np.float32)
        if velo.exists():
            depth = lidar_depth_map(read_velodyne(velo), calib["T_cam2_velo"], calib["K"], bgr.shape[:2])
            n_lidar += 1
        d = np.where((depth > 0) & (depth / DEPTH_UNIT < 65535), np.round(depth / DEPTH_UNIT), 0).astype(np.uint16)
        cv2.imwrite(str(out_dir / "depth" / f"{name}.png"), d)
        meta["frames"].append({"name": name, "K": calib["K"].tolist(), "c2w": poses[i].tolist()})
    (out_dir / "cameras.json").write_text(json.dumps(meta))
    return {"frames": len(meta["frames"]), "with_lidar": n_lidar}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eigen-test-list", required=True, help="Eigen test_files.txt (leakage guard)")
    ap.add_argument("--plan", default=str(REPO / "configs" / "lora_analysis.yaml"))
    ap.add_argument("--stride", type=int, default=1, help="keep every k-th frame (1 = all)")
    ap.add_argument("--split-only", action="store_true", help="write split.json for drives already converted")
    ap.add_argument("--split-from", default=None,
                    help="use this split (planned_split.json of download_kitti.py) instead of drawing one; leakage-checked")
    ap.add_argument("--skip-existing", action="store_true",
                    help="resume: keep drives whose cameras.json exists (written last, so it marks a complete drive)")
    args = ap.parse_args()

    import yaml

    k = yaml.safe_load(Path(args.plan).read_text())["kitti"]
    eigen = read_eigen_test_drives(args.eigen_test_list)
    raw, out = Path(args.raw), Path(args.out)
    drives = find_drives(raw) if not args.split_only else sorted(p for p in out.iterdir() if (p / "cameras.json").exists())
    if not drives:
        raise SystemExit(f"no *_drive_*_sync folders with image_02/data under {raw}")
    # the split is drawn from the drive NAMES first, so only the drives it uses are converted
    names = [d.name for d in drives]
    if args.split_from:
        planned = json.loads(Path(args.split_from).read_text())
        avail = {drive_id(n) for n in names}
        roles = {r: [d for d in planned[r] if d in avail] for r in ("train", "val", "test")}
        missing = sorted(set(planned["train"] + planned["val"] + planned["test"]) - avail)
        split = check_split(roles["train"], roles["val"], roles["test"], eigen)
        split.update(planned_from=str(args.split_from), not_downloaded=missing)
        if missing:
            print("[note] planned but not available (yet):", missing)
    else:
        split = make_split(names, eigen, k["n_test_drives"], k["n_val_drives"], k["split_seed"], k.get("n_train_drives"))
    needed = set(split["train"]) | set(split["val"]) | set(split["test"])
    stats = {}
    if not args.split_only:
        calibs = find_calibrations(raw)
        todo = [d for d in drives if drive_id(d.name) in needed]
        missing = sorted({drive_id(d.name)[:10] for d in todo} - set(calibs))
        if missing:
            raise SystemExit(f"calibration files (calib_cam_to_cam.txt, ...) missing for dates {missing}: "
                             f"also provide <date>_calib.zip from the KITTI raw page")
        print(f"{len(drives)} drives found, {len(todo)} used by the split "
              f"(train {len(split['train'])}, val {len(split['val'])}, test {len(split['test'])})", flush=True)
        for d in todo:
            if args.skip_existing and (out / drive_id(d.name) / "cameras.json").exists():
                meta = json.loads((out / drive_id(d.name) / "cameras.json").read_text())
                stats[drive_id(d.name)] = {"frames": len(meta["frames"]), "skipped": "already converted"}
                print(drive_id(d.name), "already converted, skipped", flush=True)
                continue
            stats[drive_id(d.name)] = convert_drive(d, out / drive_id(d.name), calibs[drive_id(d.name)[:10]], args.stride)
            print(drive_id(d.name), stats[drive_id(d.name)], "(Eigen test drive)" if drive_id(d.name) in eigen else "",
                  flush=True)
    split["eigen_list_sha256"] = hashlib.sha256(Path(args.eigen_test_list).read_bytes()).hexdigest()
    split["n_eigen_drives_in_list"] = len(eigen)
    split["stats"] = stats
    (out / "split.json").write_text(json.dumps(split, indent=2))
    print(json.dumps({k: v for k, v in split.items() if k != "stats"}, indent=2))
    if not split["test"]:
        print("[note] no Eigen test drive downloaded: fine for the go/no-go (it uses non-test drives), "
              "required later for the held-out evaluation")


if __name__ == "__main__":
    main()
