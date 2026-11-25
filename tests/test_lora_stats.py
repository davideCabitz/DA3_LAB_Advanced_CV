"""Pre-registered LoRA statistics (src/da3/lora_stats.py) and scripts/lora_analysis.py on synthetic data."""

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from da3.lora_stats import (
    FAIL,
    INCONCLUSIVE,
    NULL,
    PASS,
    decide_superiority,
    holm,
    non_inferiority,
    sign_flip_p,
    superiority_ratio,
    unit_weights,
)

REPO = Path(__file__).resolve().parents[1]


def test_exact_sign_flip_p_values():
    assert sign_flip_p(-np.ones(7), alternative="less") == pytest.approx(1 / 128)   # all 2^7 patterns
    assert sign_flip_p(np.ones(7), alternative="less") == 1.0
    e = np.array([-3.0, -2.0, -1.0, 0.5])
    assert sign_flip_p(e, alternative="less") == pytest.approx(2 / 16)              # only the obs. and -0.5 variant


def test_unit_weights_give_equal_dataset_weight():
    w = unit_weights(["a", "a", "a", "b"])
    assert w.sum() == pytest.approx(1) and w[3] == pytest.approx(0.5) and w[0] == pytest.approx(1 / 6)


def test_superiority_decisions():
    rng = np.random.default_rng(0)
    A = rng.uniform(0.1, 0.2, 12)
    assert decide_superiority(superiority_ratio(A, A * 0.8, n_boot=2000), 0.025) == PASS
    assert decide_superiority(superiority_ratio(A, A * 1.2, n_boot=2000), 0.025) == FAIL
    assert decide_superiority(superiority_ratio(A, A * (1 + rng.normal(0, 0.002, 12)), n_boot=2000), 0.025) == NULL
    noisy = A * (1 + rng.normal(0, 0.3, 12))
    assert decide_superiority(superiority_ratio(A, noisy, n_boot=2000), 0.025) in (INCONCLUSIVE, NULL, FAIL, PASS)


def test_non_inferiority_absolute_and_relative():
    rng = np.random.default_rng(1)
    g = ["h"] * 30 + ["s"] * 7
    A = rng.uniform(10, 40, 37)
    assert non_inferiority(A, A - 0.1, g, margin_abs=1.0, n_boot=2000)["decision"] == PASS
    assert non_inferiority(A, A - 5.0, g, margin_abs=1.0, n_boot=2000)["decision"] == FAIL
    C = rng.uniform(1, 3, 22)
    assert non_inferiority(C, C * 1.001, ["d"] * 22, margin_rel=0.03, n_boot=2000)["decision"] == PASS
    assert non_inferiority(C, C * 1.2, ["d"] * 22, margin_rel=0.03, n_boot=2000)["decision"] == FAIL


def test_holm():
    r = holm({"a": 0.001, "b": 0.02, "c": 0.04}, 0.05)
    assert r["a"][1] and r["b"][1] and r["c"][1]
    r = holm({"a": 0.001, "b": 0.03, "c": 0.04}, 0.05)
    assert r["a"][1] and not r["b"][1] and not r["c"][1]
    assert r["b"][0] == pytest.approx(0.06)


def test_analysis_script_end_to_end(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    drives = [f"2011_09_26_drive_{i:04d}" for i in range(12)]
    metrics = ["absrel_clip", "delta1_clip", "absrel_image", "delta1_image", "ate", "rpe_trans", "rpe_rot", "auc03", "auc30"]
    base = {d: {m: float(rng.uniform(0.1, 0.3)) for m in metrics} for d in drives}
    def arm(f):
        return {d: {m: v * f * (1 + rng.normal(0, 0.01)) for m, v in base[d].items()} for d in drives}
    per_drive = {"released": base, "e3_s0": arm(0.85), "e3_s1": arm(0.86), "pose_only_s0": arm(0.97),
                 "pose_only_s1": arm(0.98), "own_output_s0": arm(1.0), "own_output_s1": arm(1.0)}
    kitti = {"args": {"preset": "da3-small", "views": 8}, "drives": drives, "summary": {"per_drive": per_drive},
             "adapters": {a: {"path": a, "arm": a.rsplit("_s", 1)[0]} for a in per_drive if a != "released"}}
    (tmp_path / "k.json").write_text(json.dumps(kitti))
    def rec(shift):
        per = {}
        for d, n in (("hiroom", 30), ("7scenes", 7), ("dtu64", 22)):
            per[f"{d}_pose"] = {f"s{i}": {"auc03": 0.2 + 0.001 * i + shift, "auc30": 0.8 + shift} for i in range(n)}
            per[f"{d}_recon_unposed"] = {f"s{i}": {"fscore": 0.5 + shift, "overall": 2.0 - shift} for i in range(n)}
        return {"sec6_per_scene": per}
    (tmp_path / "r.json").write_text(json.dumps(rec(0.0)))
    (tmp_path / "e0.json").write_text(json.dumps(rec(-0.0005)))
    (tmp_path / "e1.json").write_text(json.dumps(rec(0.0005)))
    monkeypatch.setattr(sys, "argv", ["lora_analysis.py", "--kitti", str(tmp_path / "k.json"),
                                      "--sec6-released", str(tmp_path / "r.json"),
                                      "--sec6-e3", str(tmp_path / "e0.json"), str(tmp_path / "e1.json"),
                                      "--out", str(tmp_path / "out"), "--results-root", str(tmp_path / "res")])
    runpy.run_path(str(REPO / "scripts" / "lora_analysis.py"), run_name="__main__")
    a = json.loads((tmp_path / "out" / "analysis.json").read_text())
    assert a["H1"]["decision"] == PASS and a["H1"]["seed_robust"]
    sec = {r["id"]: r for r in a["secondary"]}
    assert sec["S4"]["decision"] == PASS and sec["S5"]["decision"] == PASS
    ni = {r["id"]: r["decision"] for r in a["non_inferiority"]}
    assert ni["N1"] == ni["N2"] == ni["N3"] == ni["N4"] == PASS and ni["N5"] == ni["N6"] == "NOT EVALUATED"
    assert a["no_regression_claim"] == PASS
    assert len(a["records"]) == 7 and (tmp_path / "out" / "summary.md").exists()
