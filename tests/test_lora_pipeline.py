"""End-to-end check of the LoRA scripts on synthetic KITTI-like drives with a tiny random model
(the released weights are replaced by a random ViT-S; nothing here is a result).

prepare split -> train_lora (cache, 2 steps, snapshot, resume, skip when finished) -> kitti_eval with
the released / posed arms and two adapters."""

import json
import runpy
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
DRIVES = ["2011_09_26_drive_0001", "2011_09_26_drive_0005", "2011_09_26_drive_0011"]


def _write_drive(root: Path, name: str, n=9, h=375, w=1242, seed=0):
    rng = np.random.default_rng(seed)
    d = root / name
    (d / "rgb").mkdir(parents=True)
    (d / "depth").mkdir()
    K = [[721.5, 0, 609.6], [0, 721.5, 172.9], [0, 0, 1]]
    frames = []
    for i in range(n):
        cv2.imwrite(str(d / "rgb" / f"{i:010d}.jpg"), rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
        depth = np.where(rng.random((h, w)) < 0.05, rng.uniform(3, 60, (h, w)), 0)
        cv2.imwrite(str(d / "depth" / f"{i:010d}.png"), np.round(depth * 256).astype(np.uint16))
        c2w = np.eye(4)
        c2w[2, 3] = 1.0 * i
        frames.append({"name": f"{i:010d}", "K": K, "c2w": c2w.tolist()})
    (d / "cameras.json").write_text(json.dumps({"depth_unit": 1 / 256, "frames": frames}))


@pytest.fixture()
def setup(tmp_path, monkeypatch):
    import da3.checkpoint_io as cio
    from helpers import tiny_model

    root = tmp_path / "kitti"
    for k, d in enumerate(DRIVES):
        _write_drive(root, d, seed=k)
    split = {"train": DRIVES[:2], "val": [DRIVES[2]], "test": [DRIVES[2]]}
    (root / "split.json").write_text(json.dumps(split))
    fake_weights = tmp_path / "model.safetensors"
    fake_weights.write_bytes(b"released-weights-stand-in")
    monkeypatch.setattr(cio, "official_model", lambda preset="da3-small", path=None, **kw: (tiny_model("da3-small", **kw), {}))
    monkeypatch.setattr(cio, "download_official", lambda preset, local_dir="weights": fake_weights)
    plan = yaml.safe_load((REPO / "configs" / "lora_analysis.yaml").read_text())
    plan["train"].update(views=3, steps=2, warmup=1)
    plan["kitti"]["clip_stride_frames"] = 2
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan))
    return tmp_path, root, plan_path


def _run(script, args, monkeypatch):
    monkeypatch.setattr(sys, "argv", [script] + [str(a) for a in args])
    runpy.run_path(str(REPO / "scripts" / script), run_name="__main__")


def test_train_and_evaluate_end_to_end(setup, monkeypatch):
    tmp, root, plan = setup
    common = ["--root", root, "--plan", plan, "--cache", tmp / "cache", "--out", tmp / "runs", "--device", "cpu",
              "--snapshot-every", 1, "--ckpt-every", 1]
    _run("train_lora.py", common + ["--arm", "e3", "--seed", 0, "--time-budget-h", 1e-9], monkeypatch)  # stops after 1 step
    run = tmp / "runs" / "da3-small_e3_s0"
    assert (run / "ckpt" / "last.pt").exists() and not (run / "adapter.pt").exists()
    _run("train_lora.py", common + ["--arm", "e3", "--seed", 0, "--resume"], monkeypatch)
    assert (run / "adapter.pt").exists() and (run / "snapshots" / "step_00000.pt").exists()
    assert (run / "snapshots" / "step_00002.pt").exists()
    summary = json.loads((run / "train_summary.json").read_text())
    assert summary["final_step"] == 2 and summary["plan_version"] == 2 and DRIVES[2] not in summary["train_drives"]
    log = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [1, 2] and all(np.isfinite(r["total"]) for r in log)
    _run("train_lora.py", common + ["--arm", "pose_only", "--seed", 1], monkeypatch)
    _run("train_lora.py", common + ["--arm", "e3", "--seed", 0], monkeypatch)   # finished -> skipped
    assert len(list((tmp / "cache").glob("clip_*.npz"))) >= 2

    out = tmp / "eval" / "test.json"
    _run("kitti_eval.py", ["--root", root, "--drives", "test", "--arms", "released", "posed", "--views", 3, "--stride", 2,
                           "--clips-per-drive", 2, "--device", "cpu", "--fp32", "--out", out,
                           "--adapters", f"e3_s0={run / 'adapter.pt'}",
                           f"po_s1={tmp / 'runs' / 'da3-small_pose_only_s1' / 'adapter.pt'}"], monkeypatch)
    res = json.loads(out.read_text())
    arms = {r["arm"] for r in res["rows"]}
    assert arms == {"released", "posed", "e3_s0", "po_s1"}
    n = {a: sum(r["arm"] == a for r in res["rows"]) for a in arms}
    assert len(set(n.values())) == 1                       # identical clips for every arm
    assert set(res["adapters"]) == {"e3_s0", "po_s1"} and res["adapters"]["e3_s0"]["arm"] == "e3"
    assert "e3_s0" in res["summary"]["vs_released"] and "gate" in res["summary"]
