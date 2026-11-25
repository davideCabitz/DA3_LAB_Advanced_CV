"""KITTI raw: calibration, OXTS poses, LiDAR depth, drive split (enhancement #3, docs/LORA.md).

Raw layout (synced + rectified download from cvlibs.net, after registration):

    <raw>/<date>/calib_cam_to_cam.txt, calib_velo_to_cam.txt, calib_imu_to_velo.txt
    <raw>/<date>/<date>_drive_XXXX_sync/image_02/data/0000000000.png
    <raw>/<date>/<date>_drive_XXXX_sync/oxts/data/0000000000.txt
    <raw>/<date>/<date>_drive_XXXX_sync/velodyne_points/data/0000000000.bin

Conventions (as in the KITTI devkit / pykitti):
* camera 2 = left colour camera, rectified; OpenCV axes (x right, y down, z forward); K = P_rect_02[:, :3];
* T_cam2_velo = T2 · R_rect_00 · T_cam0_velo with T2 = [I | K⁻¹ P_rect_02[:, 3]] (exactly P_rect_02);
* OXTS → world: Mercator projection with the scale of the drive's first latitude,
  R = Rz(yaw) Ry(pitch) Rx(roll); c2w(cam2) = T_w_imu · (T_cam2_velo · T_velo_imu)⁻¹;
* LiDAR depth = z of the points in camera 2, nearest point per pixel; used for EVALUATION only.

Leakage guard: the KITTI Eigen test drives (Eigen et al. 2014; the list is read from the standard
`test_files.txt`, e.g. monodepth2's splits/eigen/test_files.txt) must never be used to train or tune.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

EARTH_RADIUS = 6378137.0
DRIVE_RE = re.compile(r"(\d{4}_\d{2}_\d{2}_drive_\d{4})")


# ----------------------------------------------------------------------------- calibration
def read_calib_file(path: str | Path) -> dict[str, np.ndarray]:
    out = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        try:
            out[key.strip()] = np.array([float(v) for v in val.split()], np.float64)
        except ValueError:
            pass  # e.g. calib_time
    return out


def _rt(c: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = c["R"].reshape(3, 3), c["T"]
    return T


def kitti_calibration(date_dir: str | Path) -> dict[str, np.ndarray]:
    """K (3x3) of rectified camera 2, T_cam2_velo, T_cam2_imu (4x4)."""
    d = Path(date_dir)
    cam = read_calib_file(d / "calib_cam_to_cam.txt")
    T_cam0_velo = _rt(read_calib_file(d / "calib_velo_to_cam.txt"))
    T_velo_imu = _rt(read_calib_file(d / "calib_imu_to_velo.txt"))
    P2 = cam["P_rect_02"].reshape(3, 4)
    R_rect = np.eye(4)
    R_rect[:3, :3] = cam["R_rect_00"].reshape(3, 3)
    K = P2[:, :3].copy()
    T2 = np.eye(4)
    T2[:3, 3] = np.linalg.solve(K, P2[:, 3])
    T_cam2_velo = T2 @ R_rect @ T_cam0_velo
    return {"K": K, "P_rect_02": P2, "T_cam2_velo": T_cam2_velo, "T_cam2_imu": T_cam2_velo @ T_velo_imu}


# ----------------------------------------------------------------------------- OXTS poses
def _rot(axis: str, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def oxts_to_poses(packets: np.ndarray) -> np.ndarray:
    """(N, >=6) [lat, lon, alt, roll, pitch, yaw, ...] -> (N, 4, 4) T_w_imu, world = first IMU pose."""
    packets = np.asarray(packets, np.float64)
    scale = np.cos(packets[0, 0] * np.pi / 180.0)
    T = np.tile(np.eye(4), (len(packets), 1, 1))
    for i, (lat, lon, alt, roll, pitch, yaw) in enumerate(packets[:, :6]):
        T[i, 0, 3] = scale * lon * np.pi * EARTH_RADIUS / 180.0
        T[i, 1, 3] = scale * EARTH_RADIUS * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
        T[i, 2, 3] = alt
        T[i, :3, :3] = _rot("z", yaw) @ _rot("y", pitch) @ _rot("x", roll)
    return np.linalg.inv(T[0]) @ T


def read_oxts(drive_dir: str | Path) -> np.ndarray:
    files = sorted((Path(drive_dir) / "oxts" / "data").glob("*.txt"))
    return np.array([[float(v) for v in f.read_text().split()] for f in files], np.float64)


def camera_poses(T_w_imu: np.ndarray, T_cam2_imu: np.ndarray) -> np.ndarray:
    """(N, 4, 4) camera-2 c2w (OpenCV), world = first IMU frame."""
    return T_w_imu @ np.linalg.inv(T_cam2_imu)


# ----------------------------------------------------------------------------- LiDAR depth
def lidar_depth_map(points: np.ndarray, T_cam2_velo: np.ndarray, K: np.ndarray, hw: tuple[int, int],
                    min_depth: float = 0.1) -> np.ndarray:
    """(N, >=3) velodyne points -> (H, W) float32 depth in metres of camera 2, 0 = no point.
    The nearest point wins when several fall into one pixel."""
    H, W = hw
    p = np.c_[points[:, :3], np.ones(len(points))] @ T_cam2_velo.T
    p = p[p[:, 2] > min_depth]
    uv = (p[:, :3] / p[:, 2:3]) @ K.T
    u, v = np.round(uv[:, 0]).astype(np.int64), np.round(uv[:, 1]).astype(np.int64)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[ok], v[ok], p[ok, 2]
    order = np.argsort(-z)                       # far first, so the nearest point is written last
    depth = np.zeros((H, W), np.float32)
    depth[v[order], u[order]] = z[order]
    return depth


def read_velodyne(path: str | Path) -> np.ndarray:
    return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)


# ----------------------------------------------------------------------------- crops
def center_crop_box(h: int, w: int, aspect: float = 1.8) -> tuple[int, int, int, int]:
    """(y0, x0, ch, cw): the largest centred crop with width / height = aspect."""
    if w / h > aspect:
        ch, cw = h, int(round(h * aspect))
    else:
        ch, cw = int(round(w / aspect)), w
    return (h - ch) // 2, (w - cw) // 2, ch, cw


def crop_intrinsics(K: np.ndarray, box, out_hw) -> np.ndarray:
    """K of the crop `box` (y0, x0, ch, cw) resized to out_hw."""
    y0, x0, ch, cw = box
    K = np.array(K, np.float64, copy=True)
    K[0, 2] -= x0
    K[1, 2] -= y0
    K[0] *= out_hw[1] / cw
    K[1] *= out_hw[0] / ch
    return K


# ----------------------------------------------------------------------------- drive split
def drive_id(name: str) -> str:
    m = DRIVE_RE.search(str(name).replace("\\", "/"))
    if not m:
        raise ValueError(f"not a KITTI raw drive name: {name}")
    return m.group(1)


def read_eigen_test_drives(test_files: str | Path) -> set[str]:
    """Drive ids from an Eigen test file list (lines like '2011_09_26/2011_09_26_drive_0002_sync 69 l')."""
    drives = {drive_id(line.split()[0]) for line in Path(test_files).read_text().splitlines() if line.strip()}
    if not drives:
        raise ValueError(f"{test_files}: no drives found")
    return drives


def check_split(train: list[str], val: list[str], test: list[str], eigen_test: set[str]) -> dict:
    """Raises if any training / validation drive is an Eigen test drive or the sets overlap."""
    tr, va, te = ({drive_id(d) for d in s} for s in (train, val, test))
    problems = []
    if tr & eigen_test:
        problems.append(f"training drives in the Eigen test list: {sorted(tr & eigen_test)}")
    if va & eigen_test:
        problems.append(f"validation drives in the Eigen test list: {sorted(va & eigen_test)}")
    for a, b, na, nb in ((tr, va, "train", "val"), (tr, te, "train", "test"), (va, te, "val", "test")):
        if a & b:
            problems.append(f"{na} and {nb} share drives: {sorted(a & b)}")
    if problems:
        raise ValueError("KITTI split leaks: " + "; ".join(problems))
    return {"train": sorted(tr), "val": sorted(va), "test": sorted(te), "eigen_overlap_train_val": []}


def draw_drives(candidates, n: int, seed: int = 0) -> list[str]:
    """n drives drawn with `seed`, stratified by recording date (a proxy for the KITTI route)."""
    pool = sorted({drive_id(d) for d in candidates})
    if len(pool) <= n:
        return pool
    rng = np.random.default_rng(seed)
    by_date: dict[str, list[str]] = {}
    for d in pool:
        by_date.setdefault(d[:10], []).append(d)
    quota = {k: int(np.floor(n * len(v) / len(pool))) for k, v in by_date.items()}
    rest = n - sum(quota.values())
    for k in sorted(by_date, key=lambda k: -(n * len(by_date[k]) / len(pool) - quota[k]))[:rest]:
        quota[k] += 1
    out = []
    for k in sorted(by_date):
        out += [str(d) for d in rng.choice(sorted(by_date[k]), size=quota[k], replace=False)]
    return sorted(out)


def make_split(available, eigen_test: set[str], n_test: int, n_val: int, seed: int = 0,
               n_train: int | None = None) -> dict:
    """Pre-registered split (configs/lora_analysis.yaml `kitti`):
    test  = n_test drives drawn from the available Eigen test drives;
    val   = n_val drives drawn from the available non-Eigen drives;
    train = n_train drives drawn from the remaining non-Eigen drives (all of them if n_train is None).
    Drives not drawn are left unused (and need not be converted)."""
    avail = {drive_id(d) for d in available}
    test = draw_drives(avail & eigen_test, n_test, seed)
    val = draw_drives(avail - eigen_test, n_val, seed)
    rest = avail - eigen_test - set(val)
    train = sorted(rest) if n_train is None else draw_drives(rest, n_train, seed)
    split = check_split(train, val, test, eigen_test)
    split["unused_eigen"] = sorted((avail & eigen_test) - set(test))
    split["unused_other"] = sorted(rest - set(train))
    split["seed"] = int(seed)
    return split


# ----------------------------------------------------------------------------- clips (shared by training and evaluation)
CLIP_HW = (280, 504)        # 1.8:1, multiples of 14 (configs/lora_analysis.yaml train.resolution)


def clip_starts(n_frames: int, views: int, stride: int, n_clips: int) -> list[int]:
    """Up to n_clips evenly spaced, non-overlapping windows of `views` frames taken every `stride` frames."""
    span = (views - 1) * stride + 1
    if n_frames < span:
        return []
    k = min(n_clips, n_frames // span)
    return [int(round(x)) for x in np.linspace(0, n_frames - span, k)]


def load_clip(scene, idx: list[int], hw=CLIP_HW, aspect: float = 1.8):
    """Frames `idx` of a CanonicalScene: centre-cropped to `aspect`, RGB resized to hw.
    Returns rgb (S,H,W,3) uint8, LiDAR depth of the crop at full resolution (S,ch,cw), K of the resized
    crop (S,3,3), w2c (S,4,4) relative to the first frame (OpenCV)."""
    from da3.io_utils import resize_rgb

    rgbs, gts, Ks, w2cs = [], [], [], []
    for i in idx:
        rgb, depth = scene.load(i)
        y0, x0, ch, cw = box = center_crop_box(*rgb.shape[:2], aspect)
        rgbs.append(resize_rgb(rgb[y0:y0 + ch, x0:x0 + cw], hw))
        gts.append(depth[y0:y0 + ch, x0:x0 + cw])
        Ks.append(crop_intrinsics(scene.K[i], box, hw))
        w2cs.append(np.linalg.inv(scene.c2w[i]))
    w2c = np.stack(w2cs)
    return np.stack(rgbs), np.stack(gts), np.stack(Ks), w2c @ np.linalg.inv(w2c[0])
