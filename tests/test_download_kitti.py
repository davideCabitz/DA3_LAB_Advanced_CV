"""scripts/download_kitti.py (network mocked with local zips) + prepare_kitti.py --split-from."""

import json
import runpy
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
TEST = ["2011_09_26_drive_0002", "2011_09_26_drive_0009"]
TRAIN = ["2011_09_26_drive_0001", "2011_09_26_drive_0005", "2011_09_26_drive_0011", "2011_09_26_drive_0014"]
BIG = "2011_09_26_drive_0015"


def _drive_zip(path: Path, d: str, n=3):
    rng = np.random.default_rng(0)
    with zipfile.ZipFile(path, "w") as z:
        for i in range(n):
            ok, png = cv2.imencode(".png", rng.integers(0, 255, (375, 1242, 3), dtype=np.uint8))
            z.writestr(f"2011_09_26/{d}_sync/image_02/data/{i:010d}.png", png.tobytes())
            z.writestr(f"2011_09_26/{d}_sync/image_03/data/{i:010d}.png", b"x")    # must NOT be extracted
            z.writestr(f"2011_09_26/{d}_sync/oxts/data/{i:010d}.txt", " ".join(["49.0", f"{8 + 1e-5 * i}", "110"] + ["0"] * 27))
            pts = np.c_[rng.uniform(5, 40, (500, 1)), rng.uniform(-5, 5, (500, 2)), np.ones((500, 1))].astype(np.float32)
            z.writestr(f"2011_09_26/{d}_sync/velodyne_points/data/{i:010d}.bin", pts.tobytes())


def _calib_zip(path: Path):
    f = lambda a: " ".join(map(str, np.ravel(a)))
    K = np.array([[721.5, 0, 609.6], [0, 721.5, 172.9], [0, 0, 1]])
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("2011_09_26/calib_cam_to_cam.txt", f"R_rect_00: {f(np.eye(3))}\nP_rect_02: {f(np.c_[K, [44.9, 0.2, 0.003]])}\n")
        z.writestr("2011_09_26/calib_velo_to_cam.txt", f"R: {f([[0, -1, 0], [0, 0, -1], [1, 0, 0]])}\nT: 0 -0.08 -0.27\n")
        z.writestr("2011_09_26/calib_imu_to_velo.txt", f"R: {f(np.eye(3))}\nT: -0.8 0.3 -0.8\n")


def test_download_then_convert(tmp_path, monkeypatch):
    src = tmp_path / "server"
    src.mkdir()
    for d in TEST + TRAIN:
        _drive_zip(src / f"{d}_sync.zip", d)
    _calib_zip(src / "2011_09_26_calib.zip")
    (tmp_path / "test.txt").write_text("".join(f"2011_09_26/{d}_sync 1 l\n" for d in TEST))
    (tmp_path / "train.txt").write_text("".join(f"2011_09_26/{d}_sync 1 l\n" for d in TRAIN + [BIG, TEST[0]]))
    mod = runpy.run_path(str(REPO / "scripts" / "download_kitti.py"), run_name="dl")
    g = mod["main"].__globals__
    monkeypatch.setitem(g, "size_gb", lambda url: 9.0 if BIG in url else 0.1)
    fetched = []
    def fetch(url, dst):
        fetched.append(url.rsplit("/", 1)[-1])
        dst.write_bytes((src / url.rsplit("/", 1)[-1]).read_bytes())
    monkeypatch.setitem(g, "fetch", fetch)
    raw = tmp_path / "raw"
    monkeypatch.setattr(sys, "argv", ["dl", "--out", str(raw), "--eigen-test-list", str(tmp_path / "test.txt"),
                                      "--eigen-train-list", str(tmp_path / "train.txt")])
    g["main"]()
    plan = json.loads((raw / "planned_split.json").read_text())
    assert set(plan["test"]) == set(TEST) and BIG not in plan["train"] + plan["val"]
    assert not (set(plan["train"]) | set(plan["val"])) & set(TEST)
    assert not list(raw.rglob("image_03")) and list(raw.rglob("velodyne_points")) and not list(raw.glob("*.zip"))
    n = len(fetched)
    g["main"]()                                   # resume: nothing is fetched again
    assert len(fetched) == n
    canon = tmp_path / "canon"
    monkeypatch.setattr(sys, "argv", ["prepare_kitti.py", "--raw", str(raw), "--out", str(canon), "--eigen-test-list",
                                      str(tmp_path / "test.txt"), "--split-from", str(raw / "planned_split.json")])
    runpy.run_path(str(REPO / "scripts" / "prepare_kitti.py"), run_name="__main__")
    split = json.loads((canon / "split.json").read_text())
    assert split["test"] == sorted(plan["test"]) and split["val"] == sorted(plan["val"]) and split["train"] == sorted(plan["train"])
    meta = json.loads((canon / TEST[0] / "cameras.json").read_text())
    assert len(meta["frames"]) == 3 and np.allclose(np.array(meta["frames"][0]["c2w"])[:3, :3] @ [0, 0, 1], [1, 0, 0], atol=1e-6)
