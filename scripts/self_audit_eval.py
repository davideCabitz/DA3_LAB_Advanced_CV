"""Enhancement #1, DA3 Self-Audit: original DA3 vs DA3 + Self-Audit, from one baseline run.

Input: a baseline run of scripts/eval_bench.py made with `--dump-audit` (the harness exports +
audit_dump.npz per scene, under <work-root>/workspace/evaluation/<baseline-tag>). No GPU needed.

Three comparisons (docs/da3_enhancement_report.md §4.2), each against what the original DA3 offers:

A. Failure detection, per view. Original DA3: its own depth confidence as reliability score.
   Self-Audit: a ridge model on the audit signals (da3.enhancements.self_audit.signals) predicting
   log(1 + per-view pose error), fitted leave-one-dataset-out (never on the dataset it is tested
   on). Metrics: AUROC for views with error > 15° / > 5°, Spearman, risk–coverage (AURC).
B. Pose source, paper protocol (Sec. 6). Original DA3: camera-head poses (API default) and
   ray-map poses (official RANSAC solver; our DLT as a second row). Self-Audit: per-view choice and
   per-view inverse-variance fusion of head and ray poses, weights from the predicted errors;
   oracle choice (uses GT) as upper bound. Depth is identical in all rows (copied from the
   baseline exports), so the rows differ only in the cameras. Scored by the released harness
   (`eval_bench.py --eval-only`) and, for pose AUC, also by our metric code.
C. Depth uncertainty (datasets with per-frame GT depth: HiRoom, 7Scenes). Original DA3: confidence
   calibrated with GT on the OTHER datasets (sigma = a c^-b). Self-Audit: per-scene triple
   collocation, no GT at all. Upper bound: per-scene fit with GT. Metrics: empirical coverage of
   50% / 90% intervals and its error, Gaussian NLL of the log-depth residual; AUSE of the
   confidence ranking (identical for all three, since they are monotone in c).

Plus the go/no-go gates of the report, figures, a per-view CSV and a Markdown summary. Results are
saved as standard records (da3.compare) under --results-root.

  python scripts/self_audit_eval.py --official-src /kaggle/working/Depth-Anything-3/src \
      --bench-root /tmp/da3bench --work-root /kaggle/working/eval_self_audit \
      --baseline-tag sa_da3-small_hiroom-7scenes-dtu64 --weights official:da3-small \
      --datasets hiroom 7scenes dtu64
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GT_DEPTH_DATASETS = ("hiroom", "7scenes")
NAMES = {"hiroom": "HiRoom", "eth3d": "ETH3D", "dtu": "DTU", "dtu64": "DTU-64", "7scenes": "7Scenes",
         "scannetpp": "ScanNet++", "all": "All"}
VARIANTS = {  # variant -> enhancement family
    "ray_official": "baseline", "ray_dlt": "baseline",
    "audit_select": "self_audit", "audit_fuse": "self_audit", "oracle_select": "self_audit",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-src", required=True)
    ap.add_argument("--bench-root", required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--baseline-tag", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--max-scenes", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--results-root", default=str(REPO / "experiments" / "results"))
    ap.add_argument("--out-dir", default=None, help="figures / CSV / summary (default: outputs/self_audit/<tag>)")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--skip-sec6", action="store_true", help="no harness replays (A, C and our pose AUC only)")
    ap.add_argument("--recon-variants", nargs="*", default=["ray_official", "audit_select", "audit_fuse"],
                    help="variants also scored on reconstruction (TSDF, slow); the others get pose metrics only")
    ap.add_argument("--keep-replay-exports", action="store_true",
                    help="keep the copied exports of each variant (they duplicate the depth: large)")
    ap.add_argument("--fusion-workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true",
                    help="skip the harness replays whose result record already exists (interrupted run)")
    ap.add_argument("--tc-anchors", type=int, default=20000)
    ap.add_argument("--max-depth-samples", type=int, default=200000)
    return ap.parse_args()


# ----------------------------------------------------------------------------- data loading
def collect_scenes(work_dir: Path, datasets) -> list[dict]:
    from da3.eval.harness import load_scene_exports, unposed_scenes

    scenes = []
    for data in datasets:
        found = unposed_scenes(work_dir, data, require="audit_dump.npz")
        for scene, exports in found.items():
            sc = load_scene_exports(exports)
            sc.update({"data": data, "scene": scene, "exports": exports})
            scenes.append(sc)
        print(f"[self-audit] {data}: {len(found)} scenes with an audit dump")
    return scenes


_DATASETS = {}


def gt_depth_index(data: str, scene: str) -> dict:
    """{image_file: (gt_depth_path, aliasing_mask_path | None)} from the released dataset loader."""
    from depth_anything_3.bench.registries import MV_REGISTRY

    if data not in _DATASETS:
        _DATASETS[data] = MV_REGISTRY.get(data)()
    sd = _DATASETS[data].get_data(scene)
    depth_files = sd.aux.get("gt_depth_files", [])
    alias = sd.aux.get("aliasing_mask_files", [None] * len(depth_files))
    return {str(img): (dp, al) for img, dp, al in zip(sd.image_files, depth_files, alias)}


def read_gt_depth(data: str, path: str, alias_path: str | None, hw) -> np.ndarray:
    """GT depth in metres resized (nearest) to the prediction size; invalid pixels = 0. Decoding as
    in the released loaders (HiRoom: uint16 / 65535 * 100 m, aliasing mask excluded; 7Scenes:
    uint16 mm, 65535 = invalid)."""
    import cv2

    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return np.zeros(hw, np.float64)
    raw = raw.astype(np.float64)
    if data == "hiroom":
        d = raw / 65535.0 * 100.0
        if alias_path and os.path.exists(alias_path):
            m = cv2.imread(alias_path, cv2.IMREAD_UNCHANGED)
            if m is not None:
                d[(m if m.ndim == 2 else m[..., 0]) > 0] = 0.0
    elif data == "7scenes":
        raw[raw == 65535] = 0
        d = raw / 1000.0
    else:
        raise ValueError(f"no GT depth decoder for {data}")
    return cv2.resize(d, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)


# ----------------------------------------------------------------------------- A: failure detection
def detection_metrics(score, err, rot_err) -> dict:
    from da3.enhancements.self_audit.metrics import auroc, risk_coverage, spearman

    _, _, aurc = risk_coverage(score, err)
    return {"views": int(np.isfinite(err).sum()),
            "AUROC>15°": auroc(score, rot_err > 15), "AUROC>5°": auroc(score, rot_err > 5),
            "Spearman": spearman(score, err), "AURC (deg)": aurc}


def part_a(table: dict, features: list[str], ec_available: bool) -> dict:
    from da3.enhancements.self_audit.models import Ridge, leave_one_group_out, prepare_features
    from da3.enhancements.self_audit.signals import BASELINE_SCORE, EC_FEATURES

    y = np.log1p(table["err_head"])
    X = prepare_features(table, features)
    pred, scheme, _ = leave_one_group_out(X, y, table["data"], table["scene_id"])
    res = {"scheme": scheme, "audit_score": pred, "baseline_score": table[BASELINE_SCORE]}
    if ec_available:
        f_noec = [f for f in features if f not in EC_FEATURES]
        res["audit_score_noec"], _, _ = leave_one_group_out(prepare_features(table, f_noec), y, table["data"],
                                                            table["scene_id"])
    res["coefficients"] = Ridge(1.0).fit(X, y).to_dict(features)

    groups = {}
    for d in list(np.unique(table["data"])) + ["all"]:
        m = np.ones(len(y), bool) if d == "all" else table["data"] == d
        err, rot = table["err_head"][m], table["rot_err_head"][m]
        methods = {"DA3 confidence (original)": res["baseline_score"][m], "Self-Audit": pred[m]}
        if ec_available:
            methods["Self-Audit w/o E_c residual"] = res["audit_score_noec"][m]
        for name, s in methods.items():
            groups[f"{NAMES.get(d, d)} · {name}"] = detection_metrics(s, err, rot)
        oracle = detection_metrics(err, err, rot)
        groups[f"{NAMES.get(d, d)} · oracle (sorted by true error)"] = {"views": oracle["views"],
                                                                        "AURC (deg)": oracle["AURC (deg)"]}
        groups[f"{NAMES.get(d, d)} · DA3 confidence (original)"]["gross failures"] = int((rot > 15).sum())
    res["groups"] = groups

    signals = {}
    from da3.enhancements.self_audit.metrics import auroc, spearman

    for f in features:
        s = table[f]
        signals[f] = {"Spearman": spearman(s, table["err_head"]), "AUROC>15°": auroc(s, table["rot_err_head"] > 15),
                      "AUROC>5°": auroc(s, table["rot_err_head"] > 5),
                      "available (%)": 100.0 * float(np.isfinite(s).mean())}
    res["signals"] = signals
    return res


# ----------------------------------------------------------------------------- B: pose source
def part_b(scenes: list[dict], table: dict, features: list[str], a_res: dict) -> dict:
    from da3.enhancements.self_audit.models import leave_one_group_out, prepare_features
    from da3.enhancements.self_audit.pose_sources import fuse, inverse_variance_weight, oracle_choice, select
    from da3.eval.metrics import pose_metrics

    X = prepare_features(table, features)
    pred_ray, _, _ = leave_one_group_out(X, np.log1p(table["err_ray"]), table["data"], table["scene_id"])
    pred_head = a_res["audit_score"]
    e_head, e_ray = np.expm1(pred_head), np.expm1(pred_ray)

    poses, per_scene_auc = {v: {} for v in VARIANTS}, []
    for k, sc in enumerate(scenes):
        m = table["scene_id"] == k
        d = sc["dump"]
        hw, hK = d["head_w2c"], d["head_K"]
        off = "ray_off_w2c" in d
        rw, rK = (d["ray_off_w2c"], d["ray_off_K"]) if off else (d["ray_dlt_w2c"], d["ray_dlt_K"])
        eh, er = e_head[m], e_ray[m]
        finite = np.isfinite(eh) & np.isfinite(er)
        use_ray = np.where(finite, er < eh, False)
        w_ray = np.where(finite, inverse_variance_weight(eh, er), 0.0)
        poses["ray_official"][k] = (rw, rK)
        poses["ray_dlt"][k] = (d["ray_dlt_w2c"], d["ray_dlt_K"])
        poses["audit_select"][k] = select(hw, hK, rw, rK, use_ray)
        poses["audit_fuse"][k] = fuse(hw, hK, rw, rK, w_ray)
        poses["oracle_select"][k] = select(hw, hK, rw, rK, oracle_choice(table["err_head"][m], table["err_ray"][m]))
        row = {"data": sc["data"], "official_ray": off, "chose_ray": float(use_ray.mean()),
               "head": pose_metrics(hw, sc["gt_extrinsics"])}
        for v, store in poses.items():
            row[v] = pose_metrics(store[k][0], sc["gt_extrinsics"])
        per_scene_auc.append(row)

    groups = {}
    for dname in sorted({r["data"] for r in per_scene_auc}) + ["all"]:
        rows = [r for r in per_scene_auc if dname == "all" or r["data"] == dname]
        for v in ["head"] + list(VARIANTS):
            label = {"head": "camera head (original, API default)", "ray_official": "ray map, official RANSAC (original option)",
                     "ray_dlt": "ray map, our DLT", "audit_select": "Self-Audit per-view choice",
                     "audit_fuse": "Self-Audit per-view fusion", "oracle_select": "oracle per-view choice (uses GT)"}[v]
            g = {"scenes": len(rows), "AUC@3": 100 * np.mean([r[v]["auc03"] for r in rows]),
                 "AUC@30": 100 * np.mean([r[v]["auc30"] for r in rows])}
            if v == "audit_select":
                g["views using ray pose (%)"] = 100 * np.mean([r["chose_ray"] for r in rows])
            groups[f"{NAMES.get(dname, dname)} · {label}"] = g
    n_fallback = sum(not r["official_ray"] for r in per_scene_auc)
    return {"poses": poses, "groups": groups, "per_scene": per_scene_auc, "official_ray_missing": n_fallback}


def run_replays(args, scenes, poses, work_root: Path) -> list[str]:
    """Score every pose-source variant with the released harness (same depth, other cameras).
    Reconstruction (TSDF, slow) only for --recon-variants; the others get pose metrics only. Each
    variant's copied exports are deleted after scoring unless --keep-replay-exports (disk)."""
    import shutil

    from da3.eval.harness import evaluation_dir, write_variant_exports

    tags = []
    for v in args.variants:
        if v not in VARIANTS:
            print(f"[self-audit] unknown variant {v}, skipped")
            continue
        modes = ("pose", "recon_unposed") if v in args.recon_variants else ("pose",)
        vtag = f"{args.baseline_tag}__{v}"
        done = Path(args.results_root) / VARIANTS[v] / f"{vtag}.json"
        if args.resume and done.exists():
            try:
                json.loads(done.read_text(encoding="utf-8"))
                print(f"[self-audit] harness replay: {vtag} already scored ({done}), skipped")
                tags.append(vtag)
                continue
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass  # truncated copy: score it again
        vdir = evaluation_dir(work_root, vtag)
        for k, sc in enumerate(scenes):
            w2c, K = poses[v][k]
            write_variant_exports(sc["exports"], vdir, sc["data"], sc["scene"], w2c, K)
        workers = args.fusion_workers
        while True:
            cmd = [sys.executable, str(REPO / "scripts" / "eval_bench.py"), "--official-src", args.official_src,
                   "--bench-root", args.bench_root, "--work-root", str(work_root), "--weights", args.weights,
                   "--datasets", *args.datasets, "--modes", *modes, "--max-frames", str(args.max_frames),
                   "--max-scenes", str(args.max_scenes), "--eval-only", "--tag", vtag,
                   "--enhancement", VARIANTS[v], "--variant", v, "--baseline-tag", args.baseline_tag,
                   "--results-root", args.results_root, "--fusion-workers", str(workers),
                   "--notes", f"replay of {args.baseline_tag}: same depth, cameras from '{v}'"]
            print(f"[self-audit] harness replay: {vtag} ({' + '.join(modes)}, {workers} fusion workers)")
            rc = subprocess.run(cmd, cwd=str(REPO)).returncode
            if rc in (-9, 137) and workers > 1:  # killed: out of RAM (parallel TSDF fusions); fewer workers
                workers = max(1, workers // 2)  # (the worker count does not change the scores)
                print(f"[self-audit] replay {vtag} ran out of memory; retrying with {workers} fusion worker(s)")
                continue
            if rc != 0:
                raise subprocess.CalledProcessError(rc, cmd)
            break
        tags.append(vtag)
        if not args.keep_replay_exports:
            shutil.rmtree(vdir, ignore_errors=True)  # the record (with per-scene metrics) is already saved
    return tags


# ----------------------------------------------------------------------------- C: depth uncertainty
def binned_sigma(e, c, edges) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from da3.enhancements.self_audit.triple_collocation import trimmed_var

    lc = np.log(np.clip(c, 1e-6, None))
    cs, ss, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (lc > lo) & (lc <= hi)
        if m.sum() < 50:
            continue
        cs.append(float(np.exp(np.median(lc[m]))))
        ss.append(float(np.sqrt(trimmed_var(e[m]))))
        ns.append(int(m.sum()))
    return np.array(cs), np.array(ss), np.array(ns)


def fit_gt(e, c, n_bins=8) -> tuple[float, float]:
    from da3.enhancements.self_audit.triple_collocation import conf_bins, fit_power_law

    cs, ss, ns = binned_sigma(e, c, conf_bins(c, n_bins))
    return fit_power_law(cs, ss, ns)


def part_c(scenes: list[dict], args) -> dict:
    from da3.enhancements.self_audit import metrics as M
    from da3.enhancements.self_audit.triple_collocation import fit_scene, trimmed_var, sigma_from_conf

    rng = np.random.default_rng(0)
    per_scene = []
    for sc in scenes:
        if sc["data"] not in GT_DEPTH_DATASETS:
            continue
        D, Cf = sc["results"]["depth"].astype(np.float64), sc["results"]["conf"].astype(np.float64)
        S, H, W = D.shape
        try:
            index = gt_depth_index(sc["data"], sc["scene"])
        except Exception as e:  # dataset loader missing / layout change: report and skip
            print(f"[self-audit] no GT depth for {sc['data']}/{sc['scene']}: {e!r}")
            continue
        G = np.zeros_like(D)
        for i, img in enumerate(sc["image_files"]):
            if img in index:
                G[i] = read_gt_depth(sc["data"], index[img][0], index[img][1], (H, W))
        valid = (G > 0) & (D > 0) & np.isfinite(G) & np.isfinite(D)
        if valid.sum() < 1000:
            continue
        shift = np.median(np.log(G[valid]) - np.log(D[valid]))
        # random order, so that any prefix used later (pooled fits, curves) is an unbiased subsample
        idx = rng.permutation(np.flatnonzero(valid.ravel()))[: args.max_depth_samples]
        e = (np.log(D.ravel()[idx]) + shift - np.log(G.ravel()[idx]))
        c = Cf.ravel()[idx]
        tc = fit_scene(D, Cf, sc["dump"]["head_w2c"], sc["dump"]["head_K"], n_anchors=args.tc_anchors)
        a_or, b_or = fit_gt(e, c)
        # TC vs GT sigma in TC's own confidence bins (gate G3)
        ratios = []
        lc = np.log(np.clip(c, 1e-6, None))
        for q in tc["bins"]:
            m = (lc > q["lo"]) & (lc <= q["hi"])
            if m.sum() >= 50 and np.isfinite(q["sigma"]):
                gt_s = float(np.sqrt(trimmed_var(e[m])))
                ratios.append({"c": q["c"], "tc_sigma": q["sigma"], "gt_sigma": gt_s,
                               "ratio": q["sigma"] / gt_s if gt_s > 0 else float("nan")})
        per_scene.append({"data": sc["data"], "scene": sc["scene"], "e": e, "c": c, "tc": tc,
                          "oracle": (a_or, b_or), "ratios": ratios, "ause": M.ause(np.abs(e), c)})
        print(f"[self-audit] depth {sc['data']}/{sc['scene']}: TC a={tc['a']:.4f} b={tc['b']:.3f} "
              f"({tc['n_triplets']} triplets), GT-fit a={a_or:.4f} b={b_or:.3f}")

    # global GT fit from the OTHER datasets (the calibration the original DA3 would need)
    gt_datas = sorted({p["data"] for p in per_scene})
    for p in per_scene:
        others = [q for q in per_scene if (q["data"] != p["data"] if len(gt_datas) >= 2 else q is not p)]
        if not others:
            p["global"] = (float("nan"), 0.0)
            continue
        e = np.concatenate([q["e"][:50000] for q in others])
        c = np.concatenate([q["c"][:50000] for q in others])
        p["global"] = fit_gt(e, c)
    scheme = "other datasets" if len(gt_datas) >= 2 else "other scenes of the same dataset (one GT dataset only)"

    methods = {"DA3 confidence + GT calibration from " + ("other datasets" if len(gt_datas) >= 2 else "other scenes"): "global",
               "Self-Audit triple collocation (no GT)": "tc", "upper bound: per-scene GT fit": "oracle"}
    groups, curves = {}, {}
    for dname in gt_datas + (["all"] if len(gt_datas) > 1 else []):
        rows = [p for p in per_scene if dname == "all" or p["data"] == dname]
        for label, key in methods.items():
            vals = {"cov50": [], "cov90": [], "nll": []}
            ee, ss = [], []
            for p in rows:
                a, b = (p["tc"]["a"], p["tc"]["b"]) if key == "tc" else p[key]
                if not np.isfinite(a):
                    continue
                s = sigma_from_conf(p["c"], a, b)
                vals["cov50"].append(M.coverage(p["e"], s, 0.5))
                vals["cov90"].append(M.coverage(p["e"], s, 0.9))
                vals["nll"].append(M.gaussian_nll(p["e"], s))
                ee.append(p["e"][:20000])
                ss.append(s[:20000])
            if not vals["cov90"]:
                continue
            groups[f"{NAMES.get(dname, dname)} · {label}"] = {
                "scenes": len(vals["cov90"]),
                "coverage@50%": float(np.mean(vals["cov50"])),
                "|error|@50%": float(np.mean(np.abs(np.subtract(vals["cov50"], 0.5)))),
                "coverage@90%": float(np.mean(vals["cov90"])),
                "|error|@90%": float(np.mean(np.abs(np.subtract(vals["cov90"], 0.9)))),
                "NLL": float(np.mean(vals["nll"])),
            }
            curves.setdefault(dname, {})[label] = M.reliability_curve(np.concatenate(ee), np.concatenate(ss))
        groups[f"{NAMES.get(dname, dname)} · AUSE of DA3 confidence (same for all)"] = {
            "scenes": len(rows), "AUSE (rel.)": float(np.nanmean([p["ause"] for p in rows]))}
    return {"per_scene": per_scene, "groups": groups, "curves": curves, "global_scheme": scheme}


# ----------------------------------------------------------------------------- gates
def gates(table, a_res, b_res, c_res, ec_available) -> dict:
    from da3.enhancements.self_audit.metrics import auroc, spearman

    out = {}
    passed, eligible, detail = 0, 0, []
    for d in np.unique(table["data"]):
        m = table["data"] == d
        rot, err = table["rot_err_head"][m], table["err_head"][m]
        if (rot > 15).sum() == 0 or (rot <= 15).sum() == 0:
            detail.append(f"{NAMES.get(d, d)}: no gross failures")
            continue
        eligible += 1
        best = []
        for f in ("gap_rot_deg", "pinhole_resid_deg"):
            s = table[f][m]
            best.append((spearman(s, rot), auroc(s, rot > 15), f))
        ok = any(sp >= 0.3 and au >= 0.7 for sp, au, _ in best)
        passed += ok
        detail.append(f"{NAMES.get(d, d)}: " + ", ".join(f"{f} ρ={sp:.2f} AUROC={au:.2f}" for sp, au, f in best))
    out["G1 detector signals"] = {
        "result": ("PASS" if passed >= 2 else "FAIL") if eligible >= 2 else "n.a.",
        "value": f"{passed}/{eligible} datasets; " + "; ".join(detail),
        "criterion": "Spearman(head–ray gap or pinhole residual, rotation error) ≥ 0.3 and AUROC(>15°) ≥ 0.7 on ≥ 2 datasets"}

    wins, n, det = 0, 0, []
    for d in sorted({r["data"] for r in b_res["per_scene"]}):
        rows = [r for r in b_res["per_scene"] if r["data"] == d]
        auc = {v: 100 * np.mean([r[v]["auc03"] for r in rows]) for v in ("head", "ray_official", "audit_fuse", "oracle_select")}
        best = max(auc["head"], auc["ray_official"])
        ok = (auc["oracle_select"] - best >= 2.0) or (auc["audit_fuse"] - best >= 0.5)
        n += 1
        wins += ok
        det.append(f"{NAMES.get(d, d)}: best single {best:.1f}, fused {auc['audit_fuse']:.1f}, oracle {auc['oracle_select']:.1f}")
    out["G2 pose-source choice"] = {
        "result": ("PASS" if wins > n / 2 else "FAIL") if n else "n.a.", "value": f"{wins}/{n} datasets; " + "; ".join(det),
        "criterion": "oracle − better single source ≥ 2 AUC@3, or fusion ≥ +0.5 AUC@3, on a majority of datasets"}

    ratios = [r["ratio"] for p in c_res["per_scene"] if p["data"] == "hiroom" for r in p["ratios"] if np.isfinite(r["ratio"])]
    src = "HiRoom"
    if not ratios:
        ratios = [r["ratio"] for p in c_res["per_scene"] for r in p["ratios"] if np.isfinite(r["ratio"])]
        src = "all GT-depth datasets"
    if ratios:
        frac = float(np.mean((np.array(ratios) >= 0.7) & (np.array(ratios) <= 1.4)))
        out["G3 GT-free calibration (F2)"] = {
            "result": "PASS" if frac >= 0.75 else "FAIL",
            "value": f"{100 * frac:.0f}% of {len(ratios)} confidence bins within [0.7, 1.4] ({src}); median ratio {np.median(ratios):.2f}",
            "criterion": "triple-collocation sigma / true sigma within 0.7–1.4 in ≥ 75% of confidence bins"}
    else:
        out["G3 GT-free calibration (F2)"] = {"result": "n.a.", "value": "no dataset with GT depth", "criterion": ""}

    if ec_available:
        rot = table["rot_err_head"]
        thr = 15 if ((rot > 15).any() and (rot <= 15).any()) else 5
        a1, a0 = auroc(a_res["audit_score"], rot > thr), auroc(a_res["audit_score_noec"], rot > thr)
        out["G4 E_c fixed-point residual"] = {
            "result": "PASS" if (a1 - a0) >= 0.03 else "FAIL",
            "value": f"AUROC(>{thr}°) {a1:.3f} with vs {a0:.3f} without the E_c signals (Δ {a1 - a0:+.3f})",
            "criterion": "adding the E_c residual raises the audit AUROC by ≥ 0.03"}
    else:
        out["G4 E_c fixed-point residual"] = {"result": "n.a.", "value": "run without the E_c re-pass", "criterion": ""}
    return out


# ----------------------------------------------------------------------------- figures / files
def make_figures(out_dir: Path, table, a_res, c_res, scenes) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from da3.compare.plots import curves
    from da3.enhancements.self_audit.metrics import risk_coverage

    files = []
    datas = list(np.unique(table["data"])) + ["all"]
    fig, axes = plt.subplots(1, len(datas), figsize=(4.4 * len(datas), 3.6), squeeze=False)
    for ax, d in zip(axes[0], datas):
        m = np.ones(len(table["data"]), bool) if d == "all" else table["data"] == d
        err = table["err_head"][m]
        named = {}
        for name, s in (("DA3 confidence (original)", a_res["baseline_score"][m]), ("Self-Audit", a_res["audit_score"][m]),
                        ("oracle", err)):
            cov, risk, aurc = risk_coverage(s, err)
            named[f"{name} (AURC {aurc:.1f})"] = (cov, risk)
        curves(named, "coverage (fraction of views kept)", "mean pose error of kept views (deg, ≤30)",
               f"{NAMES.get(d, d)}: risk–coverage", ax=ax)
    fig.tight_layout()
    p = out_dir / "risk_coverage.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p.name)

    if c_res["curves"]:
        ds = list(c_res["curves"])
        fig, axes = plt.subplots(1, len(ds), figsize=(4.4 * len(ds), 3.6), squeeze=False)
        for ax, d in zip(axes[0], ds):
            curves(c_res["curves"][d], "nominal coverage", "empirical coverage", f"{NAMES.get(d, d)}: depth intervals",
                   ax=ax, diagonal=True)
        fig.tight_layout()
        p = out_dir / "depth_reliability.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        files.append(p.name)

        tc = [(r["gt_sigma"], r["tc_sigma"], q["data"]) for q in c_res["per_scene"] for r in q["ratios"]]
        if tc:
            fig, ax = plt.subplots(figsize=(4.4, 3.8))
            for d in sorted({t[2] for t in tc}):
                g = np.array([(a, b) for a, b, dd in tc if dd == d])
                ax.scatter(g[:, 0], g[:, 1], s=10, alpha=0.6, label=NAMES.get(d, d))
            lo = min(min(t[0] for t in tc), min(t[1] for t in tc))
            hi = max(max(t[0] for t in tc), max(t[1] for t in tc))
            xs = np.array([lo, hi])
            ax.plot(xs, xs, "k--", lw=0.8, label="ideal")
            ax.fill_between(xs, 0.7 * xs, 1.4 * xs, color="gray", alpha=0.15, label="gate band 0.7–1.4")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("true sigma of log-depth error (GT)")
            ax.set_ylabel("triple-collocation sigma (no GT)")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            p = out_dir / "tc_vs_gt_sigma.png"
            fig.savefig(p, dpi=130)
            plt.close(fig)
            files.append(p.name)

    coefs = a_res["coefficients"]["standardised_coefficients"]
    fig, ax = plt.subplots(figsize=(5.2, 0.32 * len(coefs) + 1))
    names = list(coefs)
    ax.barh(names, [coefs[n] for n in names], color=["C3" if coefs[n] > 0 else "C0" for n in names])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("standardised ridge coefficient (→ predicts larger pose error)")
    ax.set_title("What the audit relies on (fit on all views)", fontsize=10)
    fig.tight_layout()
    p = out_dir / "audit_coefficients.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p.name)

    try:  # most / least suspicious views according to the audit
        import cv2

        score = a_res["audit_score"]
        finite = np.flatnonzero(np.isfinite(score))
        order = finite[np.argsort(-score[finite])]
        picks = list(order[:5]) + list(order[::-1][:3])
        fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
        for ax, i in zip(axes.ravel(), picks):
            img = cv2.imread(table["image"][i])
            if img is not None:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            kind = "flagged" if i in order[:5] else "trusted"
            ax.set_title(f"{kind}: {NAMES.get(table['data'][i], table['data'][i])}\n"
                         f"true error {table['err_head'][i]:.1f}°, audit {np.expm1(score[i]):.1f}°", fontsize=8)
            ax.axis("off")
        fig.suptitle("Self-Audit: 5 most suspicious views (top) / 3 most trusted (bottom right)", fontsize=10)
        fig.tight_layout()
        p = out_dir / "audit_examples.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        files.append(p.name)
    except Exception as e:  # images not reachable: the figure is optional
        print(f"[self-audit] example figure skipped: {e!r}")
    return files


def write_csv(path: Path, table: dict, a_res: dict, features):
    cols = ["data", "scene", "view", "image"] + list(features) + ["err_head", "rot_err_head", "err_ray"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols + ["baseline_score", "audit_score_deg"])
        for i in range(len(table["data"])):
            w.writerow([table[c][i] for c in cols] + [a_res["baseline_score"][i], float(np.expm1(a_res["audit_score"][i]))])


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    args.official_src = str(Path(args.official_src).resolve())
    args.bench_root = str(Path(args.bench_root).resolve())
    args.results_root = str(Path(args.results_root).resolve())
    out_dir = Path(args.out_dir or REPO / "outputs" / "self_audit" / args.baseline_tag).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from da3.compare import make_record, protocol, save_record
    from da3.compare.schema import load_records
    from da3.compare.tables import extra_markdown, sec6_markdown
    from da3.enhancements.self_audit.signals import EC_FEATURES, FEATURES, compute_signals, per_view_pose_error
    from da3.eval.harness import evaluation_dir, prepare_workspace
    from da3.train_utils import environment_info, git_commit

    work_root = prepare_workspace(args.work_root, args.bench_root)
    os.chdir(work_root)
    sys.path.insert(0, args.official_src)
    work_dir = evaluation_dir(work_root, args.baseline_tag)

    scenes = collect_scenes(work_dir, args.datasets)
    if not scenes:
        raise SystemExit(f"no scene with audit_dump.npz under {work_dir}: run eval_bench.py --dump-audit first")

    # per-view table
    cols = {k: [] for k in ["data", "scene", "scene_id", "view", "image", "err_head", "rot_err_head", "err_ray"] + FEATURES}
    for k, sc in enumerate(scenes):
        sig = compute_signals(sc["results"], sc["dump"])
        d = sc["dump"]
        e_head = per_view_pose_error(d["head_w2c"], sc["gt_extrinsics"])
        ray = d["ray_off_w2c"] if "ray_off_w2c" in d else d["ray_dlt_w2c"]
        e_ray = per_view_pose_error(ray, sc["gt_extrinsics"])
        S = len(d["head_w2c"])
        cols["data"] += [sc["data"]] * S
        cols["scene"] += [sc["scene"]] * S
        cols["scene_id"] += [k] * S
        cols["view"] += list(range(S))
        cols["image"] += list(sc["image_files"])
        cols["err_head"] += list(e_head["err"])
        cols["rot_err_head"] += list(e_head["rot_err"])
        cols["err_ray"] += list(e_ray["err"])
        for f in FEATURES:
            cols[f] += list(sig[f])
    table = {k: np.asarray(v) for k, v in cols.items()}
    ec_available = bool(np.isfinite(table[EC_FEATURES[0]]).any())
    features = [f for f in FEATURES if np.isfinite(table[f]).any()]
    print(f"[self-audit] {len(table['data'])} views, features: {features}")

    a_res = part_a(table, features, ec_available)
    b_res = part_b(scenes, table, features, a_res)
    c_res = part_c(scenes, args)
    g = gates(table, a_res, b_res, c_res, ec_available)

    files = make_figures(out_dir, table, a_res, c_res, scenes)  # before the replays: kept even if one fails
    write_csv(out_dir / "per_view.csv", table, a_res, features)
    files.append("per_view.csv")

    # The replays run the harness (TSDF fusion workers) in a subprocess: free the per-scene depth /
    # confidence / dump arrays first, they are no longer needed (the replays re-read the exports).
    scenes = [{k: sc[k] for k in ("data", "scene", "exports")} for sc in scenes]
    import gc

    gc.collect()
    replay_tags = []
    if not args.skip_sec6:
        replay_tags = run_replays(args, scenes, b_res["poses"], work_root)

    base = [r for r in load_records(args.results_root) if r["tag"] == args.baseline_tag]
    if base:
        proto = base[0]["protocol"]
    else:
        print("[self-audit] baseline record not found under --results-root; protocol rebuilt from the arguments")
        proto = protocol(args.datasets, "all" if not args.max_scenes else sorted({s["scene"] for s in scenes}),
                         args.max_frames, ["pose", "recon_unposed"], "first")
    extra = {
        "failure_detection": a_res["groups"],
        "signals": a_res["signals"],
        "pose_source": b_res["groups"],
        "depth_uncertainty": c_res["groups"],
        "gates": g,
        "audit_model": {"all": {"scheme": a_res["scheme"], **{f"coef {k}": v for k, v in
                                                            a_res["coefficients"]["standardised_coefficients"].items()}}},
    }
    rec = make_record(
        enhancement="self_audit", variant="audit_metrics", tag=f"{args.baseline_tag}__self_audit",
        setting="official" if args.weights.startswith("official:") else "reduced-scale", weights=args.weights,
        protocol=proto, extra=extra, baseline_tag=args.baseline_tag, commit=git_commit(REPO), env=environment_info(),
        notes=(f"audit fitted {a_res['scheme']}; depth calibration baseline from {c_res['global_scheme']}; "
               f"official ray poses missing in {b_res['official_ray_missing']} scene(s) (DLT used there); "
               f"replays: {', '.join(replay_tags) or 'none'}"),
        files=[str(out_dir / f) for f in files],
    )
    rec_path = save_record(rec, args.results_root)

    # Markdown summary: Sec. 6 group of this baseline + the Self-Audit tables
    group = [r for r in load_records(args.results_root)
             if r["tag"] == args.baseline_tag or r.get("baseline_tag") == args.baseline_tag]
    md = [f"# Self-Audit results · {args.weights} · baseline run `{args.baseline_tag}`\n",
          "## B. Paper protocol (Sec. 6): only the cameras differ, depth is identical\n", sec6_markdown(group),
          "## A. Failure detection (per view)\n", extra_markdown(group, "self_audit", "failure_detection"),
          "## B'. Pose source (our metric code, per dataset)\n", extra_markdown(group, "self_audit", "pose_source"),
          "## C. Depth uncertainty (log-depth intervals)\n", extra_markdown(group, "self_audit", "depth_uncertainty"),
          "## Single signals (pooled)\n", extra_markdown(group, "self_audit", "signals"),
          "## Go/no-go gates (report §4.2)\n", extra_markdown(group, "self_audit", "gates")]
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(f"{k}: {v['result']} | {v['value']}" for k, v in g.items()))
    print("wrote", rec_path, "and", out_dir)


if __name__ == "__main__":
    main()
