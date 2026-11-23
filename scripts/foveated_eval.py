"""Enhancement #2, Foveated DA3: mixed per-view resolutions in one pass (docs/da3_enhancement_report.md §4.3).

Sub-commands, in the order notebook 06_foveated runs them (GPU parts first, analysis last):

  parity     GPU  routing check with the released weights: all views at one resolution -> outputs
                  bit-identical to DA3.forward (unposed fp32 / fp16, released reference selection,
                  posed through E_c); plus cross-group and permutation checks at mixed resolution
  probe      GPU  gates 1 and 3 on HiRoom / 7Scenes, N views per scene (half of them focus views):
                  uniform-resolution curve (504 ... 252 px) and cross-scale probe (focus 504 px +
                  context 252 / 280 px, plain and attention-mass balanced) vs all-504 / all-252
  scaling    GPU  large-view protocol on full 7Scenes sequences (100 ... 1000 views): time, peak
                  memory, OOM, accuracy of the focus views; foveated vs all-504, uniform at equal
                  token budget, all-252 and fewer full-resolution views at equal budget
  sec6views  CPU  per-view metrics of a Sec. 6 run exported by eval_bench.py: focus vs context views,
                  depth vs GT, pose AUC on view subsets, efficiency meta (run before deleting exports)
  zoom       GPU  region level (H4), gate 2: exact virtual zoom cameras from the full-resolution ETH3D
                  photos, given through E_c together with the scene views, vs global 504 / 896 px
                  and a naive crop
  analyse    CPU  gates, tables, figures, standard result records (<results-root>/foveated/),
                  summary.md

Setting: [official] (released weights, network unchanged). The probe, scaling and zoom protocols
are new (not Sec. 6) and never enter the Sec. 6 tables; Sec. 6 runs go through scripts/eval_bench.py
(--process-res / --meter / --foveated) and the released harness. Depth is scale-free: every depth
error is median-aligned per view (or per region), plus a per-scene alignment for consistency.

Statistics follow the pre-registered plan (configs/foveated_analysis.yaml, explained in
docs/FOVEATED_ANALYSIS_PLAN.md; its SHA-256 is stored with every result): scene = unit of analysis,
paired scene-bootstrap CIs, three-way gates (PASS / FAIL / INCONCLUSIVE) with the margins at which
each decision changes, margins raised only by the treatment-blind A/A noise floor (capped), one
primary configuration per gate, secondary ones with simultaneous CIs, selection only on a dev split.

  python scripts/foveated_eval.py probe --official-src /kaggle/working/Depth-Anything-3/src \
      --bench-root /tmp/da3bench --work-root /tmp/fov_work --weights official:da3-small \
      --datasets hiroom 7scenes --out /kaggle/working/foveated/probe_da3-small.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import sys
import time
import zipfile
import zlib
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "configs" / "foveated_analysis.yaml"   # pre-registered analysis plan
NAMES = {"hiroom": "HiRoom", "7scenes": "7Scenes", "eth3d": "ETH3D", "dtu64": "DTU-64", "dtu": "DTU",
         "scannetpp": "ScanNet++", "all": "All"}


# ============================================================================= shared helpers
def jdump(obj, path: Path):
    def conv(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=conv), encoding="utf-8")
    print("wrote", path)


def jload(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_model(weights: str, device):
    if weights.startswith("official:"):
        from da3.checkpoint_io import official_model

        model, _ = official_model(weights.split(":", 1)[1])
    else:
        from da3 import DA3

        state = torch.load(weights, map_location="cpu", weights_only=False)
        model = DA3(state["config"]["model"])
        model.load_state_dict(state["model"])
    return model.eval().to(device)


def model_name(weights: str) -> str:
    return weights.split(":", 1)[1] if weights.startswith("official:") else Path(weights).stem


def open_harness(args):
    """chdir into the harness workspace (it resolves dataset paths from the CWD) and import the
    released dataset registry. Call after every path argument has been made absolute."""
    from da3.eval.harness import prepare_workspace

    work_root = prepare_workspace(args.work_root, args.bench_root)
    os.chdir(work_root)
    sys.path.insert(0, str(Path(args.official_src).resolve()))
    from depth_anything_3.bench.registries import MV_REGISTRY

    return MV_REGISTRY, work_root


def absolutise(args, *names):
    for n in names:
        v = getattr(args, n, None)
        if v:
            setattr(args, n, str(Path(v).resolve()))


def harness_sample(n: int, max_frames: int = 100) -> list[int]:
    """The released harness' frame sampling (Evaluator._sample_frames): seed 42, sorted."""
    if max_frames <= 0 or n <= max_frames:
        return list(range(n))
    random.seed(42)
    idx = list(range(n))
    random.shuffle(idx)
    return sorted(idx[:max_frames])


def evenly(n: int, k: int) -> np.ndarray:
    return np.unique(np.round(np.linspace(0, n - 1, min(n, k))).astype(int))


def degrade_to(img: np.ndarray, res: int) -> np.ndarray:
    """The image as a `res`-px view would see it, brought back to its original size (released
    preprocessing at `res`, then bicubic upsampling). Diagnostic: fed at the focus resolution it has
    the focus views' token grid and patch footprint but only `res`-px information."""
    import cv2

    from da3.io_utils import preprocess_like_official

    h, w = img.shape[:2]
    return cv2.resize(preprocess_like_official(img, res), (w, h), interpolation=cv2.INTER_CUBIC)


def prep_views(images, res_per_view):
    """Released preprocessing of every uint8 RGB image at its own resolution (center-crop to a
    common size within a resolution, as io_utils.load_views) -> list of (3, H, W) tensors."""
    from da3.io_utils import preprocess_like_official, to_model_input

    out = [None] * len(images)
    for r in sorted(set(res_per_view)):
        idx = [i for i, rr in enumerate(res_per_view) if rr == r]
        proc = [preprocess_like_official(images[i], r) for i in idx]
        mh, mw = min(o.shape[0] for o in proc), min(o.shape[1] for o in proc)
        proc = [o[int(round((o.shape[0] - mh) / 2.0)):][:mh, int(round((o.shape[1] - mw) / 2.0)):][:, :mw] for o in proc]
        x = to_model_input(np.stack(proc))[0]
        for j, i in enumerate(idx):
            out[i] = x[j]
    return out


def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_pass(model, views, device, amp: bool = True, w2c=None, K=None, keep=("depth",), offload=True, **kw) -> dict:
    """One foveated_forward (any mix of resolutions) with head cameras, time, peak memory, OOM flag.
    The views are stacked per resolution and uploaded before the clock starts (one copy on the GPU,
    released after the patch embedding); time = forward pass + camera decoding."""
    from da3.enhancements.foveated import cameras_from_head, foveated_forward
    from da3.enhancements.foveated.routing import prepare_views

    free_cuda()
    oom = False
    res = {}
    try:
        prep = prepare_views(views, device)
        del views
        cuda_sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            out = foveated_forward(model, prep, w2c=w2c, K=K, keep=keep, offload=offload, **kw)
        w2c_p, K_p = cameras_from_head(out["pose_enc"], out["hw"])
        cuda_sync(device)
        res = {"time_s": time.time() - t0,
               "peak_mem_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else float("nan"),
               "out": out, "w2c": w2c_p.float().cpu().numpy(), "K": K_p.float().cpu().numpy(), "hw": out["hw"]}
    except RuntimeError as e:   # torch.cuda.OutOfMemoryError is a RuntimeError
        if "out of memory" not in str(e).lower():
            raise
        oom = True
    prep = None
    if oom:
        free_cuda()
        return {"oom": True}
    res["oom"] = False
    return res


def warmup(model, device):
    v = [torch.zeros(3, 112, 112), torch.zeros(3, 56, 56)]
    run_pass(model, v, device)
    free_cuda()


def depth_rows(preds, gts, groups=None):
    """Per-view errors: median-aligned per view, one scale per scene, and (diagnostic) one scale per
    group of views (`groups`: one label per view, e.g. the focus mask; default: one group). Also the
    per-view median-alignment scale median(gt) / median(pred): its ratio between two groups of the
    same pass measures how far apart their depth scales are (1 = consistent)."""
    from da3.enhancements.foveated.metrics import depth_errors, pooled_scale, resize_pred

    preds = [None if (p is None or g is None) else resize_pred(p, g.shape) for p, g in zip(preds, gts)]
    s = pooled_scale(preds, gts)
    labels = [0] * len(preds) if groups is None else [int(g) for g in groups]
    s_group = {lab: pooled_scale([p for p, l in zip(preds, labels) if l == lab],
                                 [g for g, l in zip(gts, labels) if l == lab]) for lab in set(labels)}
    out = {"abs_rel": [], "delta1": [], "delta105": [], "abs_rel_scene": [], "abs_rel_group": [], "scale": []}
    for p, g, lab in zip(preds, gts, labels):
        e = depth_errors(p, g) if p is not None else None
        es = depth_errors(p, g, scale=s) if (p is not None and np.isfinite(s)) else None
        sg = s_group[lab]
        eg = depth_errors(p, g, scale=sg) if (p is not None and np.isfinite(sg)) else None
        out["abs_rel"].append(e["abs_rel"] if e else float("nan"))
        out["delta1"].append(e["delta1"] if e else float("nan"))
        out["delta105"].append(e["delta105"] if e else float("nan"))
        out["abs_rel_scene"].append(es["abs_rel"] if es else float("nan"))
        out["abs_rel_group"].append(eg["abs_rel"] if eg else float("nan"))
        out["scale"].append(e["scale"] if e else float("nan"))
    return out


def nanmean(x):
    x = np.asarray(x, np.float64)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


# ============================================================================= parity (GPU)
def cmd_parity(args):
    from da3.camera import to_homogeneous
    from da3.enhancements.foveated import foveated_forward
    from da3.io_utils import list_images, load_views, to_model_input
    from da3.model import DA3

    device = torch.device(args.device)
    model = load_model(args.weights, device)
    files = list_images(args.images)[: args.views]
    if len(files) < 3:
        raise SystemExit(f"need >= 3 images in {args.images}")
    rgb, _ = load_views(files, args.res)
    x = to_model_input(rgb).to(device)
    views = list(x[0])
    keys = ("depth", "depth_conf", "ray", "ray_conf", "pose_enc")
    checks = {}

    def compare(name, ref, fov, need_exact=True, tol=1e-5, same_ref=None):
        diffs, rels, exact = {}, [], True
        for k in keys:
            if k not in ref or (k != "pose_enc" and k not in fov):
                continue
            a = ref[k][0].float()
            b = fov[k].float() if k == "pose_enc" else torch.stack([t.float() for t in fov[k]])
            d = float((a - b).abs().max())
            exact &= bool(torch.equal(a, b))
            diffs[k] = d
            rels.append(d / max(float(a.abs().max()), 1e-12))
        rel = max(rels)
        ok = (exact or rel <= tol) if need_exact else rel <= tol
        if same_ref is not None:
            ok &= same_ref
        checks[name] = {"bit_identical": exact, "max_rel_diff": rel, "max_abs_diff": diffs, "pass": bool(ok),
                        "criterion": "bit-identical (max rel. diff <= 1e-5 tolerated)" if need_exact
                        else f"max rel. diff <= {tol:g}"}
        print(f"[parity] {name}: bit-identical={exact} max_rel={rel:.2e} -> {'PASS' if ok else 'FAIL'}")

    with torch.no_grad():
        for amp in (False, True):
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                ref = model(x)
                fov = foveated_forward(model, views)
            compare(f"equal resolution, unposed ({'fp16 autocast' if amp else 'fp32'})", ref, fov)
        ref = model(x, ref_strategy="saddle_balanced")
        fov = foveated_forward(model, views, ref_strategy="saddle_balanced")
        compare("equal resolution, released reference selection (saddle_balanced)", ref, fov,
                same_ref=fov["ref_index"] == int(ref["ref_index"][0]))
        w2c, K = DA3.cameras_from_head(ref["pose_enc"], tuple(x.shape[-2:]))
        ref_p = model(x, w2c=to_homogeneous(w2c), K=K)
        fov_p = foveated_forward(model, views, w2c=to_homogeneous(w2c[0]), K=K[0])
        compare("equal resolution, posed through E_c", ref_p, fov_p)
        fov_b = foveated_forward(model, views, balance=True)
        compare("equal resolution, balance=True is a no-op", model(x), fov_b)
        S = len(views)
        fov_g = foveated_forward(model, views, group_ids=[s % 2 for s in range(S)])
        compare("equal resolution split into two groups (round-off only)", model(x), fov_g, need_exact=False, tol=1e-4)
        # mixed resolution: permuting the non-reference views only permutes the outputs
        rgb_c, _ = load_views(files, args.context_res)
        ctx = list(to_model_input(rgb_c).to(device)[0])
        mixed = [views[s] if s % 2 == 0 else ctx[s] for s in range(S)]
        perm = [0] + list(range(S - 1, 0, -1))
        a = foveated_forward(model, mixed)
        b = foveated_forward(model, [mixed[i] for i in perm])
        d = max(float((a["depth"][i] - b["depth"][j]).abs().max() / a["depth"][i].abs().max()) for j, i in enumerate(perm))
        dp = float((a["pose_enc"][perm] - b["pose_enc"]).abs().max())
        checks["mixed resolution, view-order equivariance"] = {
            "bit_identical": False, "max_rel_diff": d, "max_abs_diff": {"pose_enc": dp}, "pass": bool(d <= 1e-3),
            "criterion": "permuting non-reference views permutes the outputs (max rel. diff <= 1e-3)"}
        print(f"[parity] mixed resolution equivariance: max_rel={d:.2e}")
        # informational: one mixed pass vs all views at full resolution
        t = {}
        for name, vs in ((f"all-{args.res}", views), (f"mixed {args.res}/{args.context_res}", mixed)):
            r = run_pass(model, vs, device)
            t[name] = {"time_s": r.get("time_s"), "peak_mem_gb": r.get("peak_mem_gb")}
    all_pass = all(c["pass"] for c in checks.values())
    jdump({"weights": args.weights, "device": str(device), "views": S, "image_hw": list(x.shape[-2:]),
           "checks": checks, "timing": t, "all_pass": all_pass}, args.out)
    print("[parity] ALL PASS" if all_pass else "[parity] SOME CHECKS FAILED")


# ============================================================================= probe (GPU): gates 1 + 3
def subset_indices(n_all: int, n_views: int, k: int) -> list:
    """k interleaved, evenly spaced sets of n_views frames (distinct when the sequence is long
    enough), so that a scene contributes several probes; the scene stays the unit of analysis."""
    k = max(1, int(k))
    if n_all <= n_views:
        return [np.arange(n_all)]
    grid = np.round(np.linspace(0, n_all - 1, n_views * k)).astype(int)
    subs = []
    for j in range(k):
        s = np.unique(grid[j::k])
        if len(s) == n_views and not any(np.array_equal(s, t) for t in subs):
            subs.append(s)
    return subs or [evenly(n_all, n_views)]


def probe_configs(args):
    """(name, spec, uniform_res, special) for every probe pass. special: None, "aa" (A/A null: full
    resolution with the reference view moved), "focus_only" (the focus views alone)."""
    from da3.enhancements.foveated import FoveationSpec

    hi = max(args.uniform)
    specs = [replace(FoveationSpec.parse(s), fraction=args.fraction) for s in args.foveated]
    cfgs = [(f"u{r}", None, int(r), None) for r in args.uniform]
    cfgs.append((f"u{hi}_aa", None, hi, "aa"))
    if specs:
        cfgs.append(("focus_only", specs[0], specs[0].focus_res, "focus_only"))
    cfgs += [(s.name, s, None, None) for s in specs]
    if specs:
        for seed in args.focus_seeds:
            s = replace(specs[0], strategy="random", seed=int(seed))
            cfgs.append((s.name, s, None, None))
        if getattr(args, "diagnostics", False):   # exploratory (added after the limited run, not pre-registered)
            cfgs.append((f"diag_up{specs[0].context_res}", specs[0], specs[0].focus_res, "upsampled"))
    return cfgs


def two_pass_row(lo: dict, hi: dict, focus: np.ndarray, gts, gt_w2c, dims, hws_lo, hws_hi) -> dict:
    """Two-pass control (exploratory): pass A = every view at the context resolution, pass B = the
    focus views alone at the focus resolution. B is mapped into A's frame with the similarity fitted on
    the focus cameras (seen in both passes); focus views take B's depth (times that scale) and pose,
    context views keep A's. `lo` / `hi`: {"preds": [n], "w2c": (n, 3, 4), "time_s", "peak_mem_gb"}."""
    from da3.enhancements.foveated.metrics import (align_cameras, pairwise_pose_errors, pairwise_rot_trans_errors,
                                                   transform_w2c)
    from da3.enhancements.foveated.plan import backbone_cost

    fi = np.flatnonzero(focus)
    s, R, t = align_cameras(hi["w2c"][fi], lo["w2c"][fi])
    w2c = np.array(lo["w2c"], np.float64)
    w2c[fi] = transform_w2c(hi["w2c"][fi], s, R, t)
    preds = [hi["preds"][i] * s if focus[i] else lo["preds"][i] for i in range(len(focus))]
    Er, Et = pairwise_rot_trans_errors(w2c, gt_w2c)
    cost_a, cost_b = backbone_cost(hws_lo, **dims), backbone_cost(hws_hi, **dims)
    return {"tokens": cost_a["tokens"] + cost_b["tokens"], "gflops": cost_a["gflops"] + cost_b["gflops"],
            "time_s": lo["time_s"] + hi["time_s"], "peak_mem_gb": max(lo["peak_mem_gb"], hi["peak_mem_gb"]),
            "pair_err": pairwise_pose_errors(w2c, gt_w2c), "pair_rot": Er, "pair_trans": Et, "align_scale": s,
            **depth_rows(preds, gts, groups=focus)}


def cmd_probe(args):
    from da3.enhancements.foveated.analysis import load_plan, probe_gates
    from da3.enhancements.foveated.metrics import pairwise_pose_errors, pairwise_rot_trans_errors
    from da3.enhancements.foveated.plan import backbone_cost, model_dims, select_focus, view_resolutions
    from da3.eval.gt_depth import gt_depth_index, read_gt_depth
    from da3.io_utils import read_rgb

    absolutise(args, "out", "bench_root", "work_root", "official_src", "plan")
    plan = load_plan(args.plan)
    MV, _ = open_harness(args)
    device = torch.device(args.device)
    model = load_model(args.weights, device)
    dims = model_dims(model)
    configs = probe_configs(args)
    # two-pass control (exploratory): needs the uniform pass at the primary context resolution
    configs_ctx = next((s.context_res for (_, s, _, sp) in configs if sp == "focus_only"), None)
    if configs_ctx not in args.uniform:
        configs_ctx = None
    warmup(model, device)
    rows = []
    for data in args.datasets:
        ds = MV.get(data)()
        scenes = ds.SCENES[: args.max_scenes] if args.max_scenes else ds.SCENES
        for scene in scenes:
            sd = ds.get_data(scene)
            index = gt_depth_index(data, sd)
            subsets = subset_indices(len(sd.image_files), args.n_views, args.subsets)
            for sub, idx in enumerate(subsets):
                files = [str(sd.image_files[i]) for i in idx]
                images = [read_rgb(f) for f in files]
                gts = [read_gt_depth(data, index[f][0], index[f][1], shape=index[f][2]) for f in files]
                gt_w2c = np.asarray(sd.extrinsics)[idx]
                n = len(idx)
                base = select_focus(n, args.fraction)          # evenly spaced focus views (primary)
                keep_tp = {}                                   # outputs reused by the two-pass control
                for name, spec, res, special in configs:
                    own = spec is not None and special is None
                    focus = select_focus(n, spec.fraction, strategy=spec.strategy, seed=spec.seed) if own else base
                    members = base.copy() if special == "focus_only" else np.ones(n, bool)
                    order = np.flatnonzero(members)
                    if special == "aa":   # same views and resolution; only the reference (first) view changes
                        f_idx = np.flatnonzero(base)
                        k = int(f_idx[len(f_idx) // 2]) if len(f_idx) > 1 else min(1, n - 1)
                        order = np.array([k] + [i for i in range(n) if i != k])
                    res_list = view_resolutions(focus, spec.focus_res, spec.context_res) if own else [res] * n
                    imgs = images
                    if special == "upsampled":   # context views carry context-res information on the focus grid
                        imgs = [im if focus[i] else degrade_to(im, spec.context_res) for i, im in enumerate(images)]
                    r = run_pass(model, prep_views([imgs[i] for i in order], [res_list[i] for i in order]), device,
                                 ref_strategy=args.ref_strategy, balance=bool(own and spec.balance))
                    row = {"data": data, "scene": scene, "subset": sub, "config": name,
                           "kind": special or ("uniform" if spec is None else "foveated"), "n_views": n,
                           "frames": [int(i) for i in idx], "focus": focus.tolist(), "members": members.tolist(),
                           "oom": r["oom"]}
                    if not r["oom"]:
                        preds, w2c = [None] * n, np.full((n, 3, 4), np.nan)
                        for j, i in enumerate(order):
                            preds[i] = r["out"]["depth"][j].float().numpy()
                            w2c[i] = r["w2c"][j]
                        m_idx = np.flatnonzero(members)
                        E, Er, Et = (np.full((n, n), np.nan) for _ in range(3))
                        E[np.ix_(m_idx, m_idx)] = pairwise_pose_errors(w2c[m_idx], gt_w2c[m_idx])
                        er, et = pairwise_rot_trans_errors(w2c[m_idx], gt_w2c[m_idx])
                        Er[np.ix_(m_idx, m_idx)], Et[np.ix_(m_idx, m_idx)] = er, et
                        cost = backbone_cost(r["hw"], **dims)
                        row.update({"tokens": cost["tokens"], "gflops": cost["gflops"], "time_s": r["time_s"],
                                    "peak_mem_gb": r["peak_mem_gb"], "pair_err": E, "pair_rot": Er, "pair_trans": Et,
                                    **depth_rows(preds, [g if mm else None for g, mm in zip(gts, members)],
                                                 groups=focus)})
                        if args.diagnostics and configs_ctx and name in (f"u{configs_ctx}", "focus_only"):
                            keep_tp[name] = {"preds": preds, "w2c": w2c, "time_s": r["time_s"],
                                             "peak_mem_gb": r["peak_mem_gb"], "hw": r["hw"]}
                    rows.append(row)
                    del r
                if len(keep_tp) == 2:   # two-pass control from the all-context-res pass + the focus-only pass
                    lo, hi = keep_tp[f"u{configs_ctx}"], keep_tp["focus_only"]
                    row = {"data": data, "scene": scene, "subset": sub, "config": f"twopass_{configs_ctx}+focus",
                           "kind": "two_pass", "n_views": n, "frames": [int(i) for i in idx], "focus": base.tolist(),
                           "members": np.ones(n, bool).tolist(), "oom": False}
                    row.update(two_pass_row(lo, hi, base, gts, gt_w2c, dims, lo["hw"], hi["hw"]))
                    rows.append(row)
            print(f"[probe] {data}/{scene}: {len(subsets)} view subsets x {len(configs)} configurations")
    analysis = probe_gates(rows, plan)
    jdump({"weights": args.weights, "datasets": args.datasets, "n_views": args.n_views, "fraction": args.fraction,
           "subsets": args.subsets, "configs": [c[0] for c in configs], "max_scenes": args.max_scenes,
           "diagnostics": bool(args.diagnostics),
           "plan": plan["path"], "plan_sha256": plan["sha256"], "rows": rows, "analysis": analysis}, args.out)
    print_gates(analysis)


def print_gates(analysis: dict, prefix: str = ""):
    m, nf = analysis.get("margins", {}), analysis.get("noise_floor", {})
    if m:
        print(f"{prefix}margins: depth {100 * m['depth']:.1f}%, pose {100 * m['pose']:.1f}% (A/A noise floor: depth "
              f"{100 * nf['depth']['floor']:.1f}%, pose {100 * nf['pose']['floor']:.1f}%)")
    for k, g in analysis.get("gates", analysis).items():
        if isinstance(g, dict) and "result" in g:
            print(f"{prefix}{g['result']:12s} [{g.get('role', '')}] {k}: {g['value']}")


# ============================================================================= scaling (GPU)
def depth_summary(dr: dict, fm: np.ndarray) -> dict:
    """Scaling-row depth fields from depth_rows output on the depth views (fm = their focus mask):
    AbsRel focus / context, scene-scale AbsRel, and the diagnostics (group-scale AbsRel, scale ratio
    context / focus: median per-view alignment scale of each group)."""
    a = {k: np.asarray(dr[k], np.float64) for k in ("abs_rel", "abs_rel_scene", "abs_rel_group", "scale")}
    has_ctx = bool((~fm).any())
    ratio = (float(np.nanmedian(a["scale"][~fm]) / np.nanmedian(a["scale"][fm]))
             if has_ctx and np.isfinite(a["scale"][fm]).any() and np.isfinite(a["scale"][~fm]).any() else float("nan"))
    return {"abs_rel_focus": nanmean(a["abs_rel"][fm]),
            "abs_rel_context": nanmean(a["abs_rel"][~fm]) if has_ctx else float("nan"),
            "abs_rel_scene_focus": nanmean(a["abs_rel_scene"][fm]),
            "abs_rel_scene_all": nanmean(a["abs_rel_scene"]),
            "abs_rel_group_all": nanmean(a["abs_rel_group"]),
            "scale_ratio_ctx_focus": ratio}


def scaling_two_pass_row(lo: dict, hi: dict, focus, depth_views, gt_w2c, gt_cache, scene, n, spec) -> dict:
    """Large-view two-pass control (exploratory): lo = every view at the context resolution, hi = the
    focus views alone at the focus resolution; hi is mapped into lo's frame with the similarity fitted
    on the focus cameras; focus views take hi's pose and depth (times that scale)."""
    from da3.enhancements.foveated.metrics import (align_cameras, pairwise_pose_errors, transform_w2c,
                                                   view_subset_aucs)

    fi = np.flatnonzero(focus)
    w2c_lo = np.asarray(lo["w2c"], np.float64)            # all n views, in order
    w2c_hi = np.asarray(hi["w2c"], np.float64)            # the focus views, in order
    s, R, t = align_cameras(w2c_hi, w2c_lo[fi])
    w2c = w2c_lo.copy()
    w2c[fi] = transform_w2c(w2c_hi, s, R, t)
    aucs = view_subset_aucs(pairwise_pose_errors(w2c, gt_w2c), focus)
    gi = [i for i in range(n) if depth_views[i]]
    preds = [hi["preds"][i] * s if focus[i] else lo["preds"][i] for i in gi]
    fm = np.array([bool(focus[i]) for i in gi])
    dr = depth_rows(preds, [gt_cache[i] for i in gi], groups=fm)
    return {"scene": scene, "n_views": n, "kind": "two_pass", "config": f"twopass_{spec.context_res}+focus",
            "views_used": n, "oom": False, "tokens": lo["tokens"] + hi["tokens"], "gflops": lo["gflops"] + hi["gflops"],
            "time_s": lo["time_s"] + hi["time_s"], "time_runs": [lo["time_s"] + hi["time_s"]],
            "timed": bool(lo["timed"] and hi["timed"]),
            "peak_mem_gb": max(lo["peak_mem_gb"], hi["peak_mem_gb"]), "auc": aucs, "align_scale": s,
            **depth_summary(dr, fm)}


def cmd_scaling(args):
    from da3.enhancements.foveated import FoveationSpec
    from da3.enhancements.foveated.metrics import pairwise_pose_errors, view_subset_aucs
    from da3.enhancements.foveated.plan import (backbone_cost, equal_budget_res, model_dims, patch_tokens,
                                                processed_hw, select_focus, subset_with_focus, view_resolutions)
    from da3.eval.gt_depth import gt_depth_index, read_gt_depth
    from da3.io_utils import read_rgb

    absolutise(args, "out", "bench_root", "work_root", "official_src")
    MV, _ = open_harness(args)
    device = torch.device(args.device)
    model = load_model(args.weights, device)
    dims = model_dims(model)
    spec = FoveationSpec(fraction=args.fraction, focus_res=args.focus_res, context_res=args.context_res)
    warmup(model, device)
    ds = MV.get(args.dataset)()
    rows, failed = [], {}   # failed: kind -> smallest view count that ran out of memory
    for si, scene in enumerate(args.scenes):
        sd = ds.get_data(scene)
        n_all = len(sd.image_files)
        index = gt_depth_index(args.dataset, sd)
        cache = {}
        repeats = max(1, args.timing_repeats) if si < args.timing_scenes else 1
        for N in sorted(set(args.n_views)):
            if N > n_all:
                print(f"[scaling] {scene}: {N} views requested, the sequence has {n_all}; skipped")
                continue
            idx = evenly(n_all, N)
            n = len(idx)
            for i in idx:
                if i not in cache:
                    cache[i] = read_rgb(str(sd.image_files[i]))
            images = [cache[i] for i in idx]
            gt_w2c = np.asarray(sd.extrinsics)[idx]
            orig_hw = images[0].shape[:2]
            focus = select_focus(n, spec.fraction, strategy=spec.strategy, seed=spec.seed)
            fov_plan = view_resolutions(focus, spec.focus_res, spec.context_res)
            fov_tokens = sum(patch_tokens(processed_hw(orig_hw, r)) for r in fov_plan)
            eq_res = equal_budget_res(orig_hw, n, fov_tokens)
            n_sub = int(fov_tokens // patch_tokens(processed_hw(orig_hw, spec.focus_res)))
            sub = subset_with_focus(n, focus, n_sub)
            ctx_every = max(1, args.depth_every)
            depth_views = focus | (np.arange(n) % ctx_every == 0)
            configs = [("all_focus_res", f"u{spec.focus_res}", np.ones(n, bool), [spec.focus_res] * n, False),
                       ("foveated", spec.name, np.ones(n, bool), fov_plan, False)]
            if args.balance:
                configs.append(("foveated_bal", spec.name + "_bal", np.ones(n, bool), fov_plan, True))
            configs += [("uniform_equal_budget", f"u{eq_res}", np.ones(n, bool), [eq_res] * n, False),
                        ("all_context_res", f"u{spec.context_res}", np.ones(n, bool), [spec.context_res] * n, False),
                        ("fewer_views_equal_budget", f"sub{int(sub.sum())}_{spec.focus_res}", sub,
                         [spec.focus_res] * int(sub.sum()), False)]
            if args.two_pass:   # exploratory: pass B of the two-pass control (pass A = all_context_res)
                configs.append(("focus_only", f"focus{int(focus.sum())}_{spec.focus_res}", focus.copy(),
                                [spec.focus_res] * int(focus.sum()), False))
            keep_tp, gt_cache = {}, {}
            for kind, name, members, plan, bal in configs:
                row = {"scene": scene, "n_views": n, "kind": kind, "config": name, "views_used": int(members.sum())}
                if kind in failed and n >= failed[kind]:   # a larger pass of the same kind cannot fit either
                    row.update({"oom": True, "skipped": True})
                    rows.append(row)
                    continue
                mi = np.flatnonzero(members)
                views = prep_views([images[i] for i in mi], plan)
                r = run_pass(model, views, device, balance=bal, ref_strategy="first")
                if r["oom"]:
                    failed[kind] = min(n, failed.get(kind, n))
                    row["oom"] = True
                    rows.append(row)
                    print(f"[scaling] {scene} N={n} {name}: out of memory")
                    continue
                times = [r["time_s"]]
                for _ in range(repeats - 1):   # timing only: the outputs are deterministic
                    rr = run_pass(model, views, device, balance=bal, ref_strategy="first")   # same work as run 1
                    if not rr["oom"]:
                        times.append(rr["time_s"])
                    del rr
                r["time_s"] = float(np.median(times))
                del views
                cost = backbone_cost(r["hw"], **dims)
                f_loc = focus[mi]
                aucs = view_subset_aucs(pairwise_pose_errors(r["w2c"], gt_w2c[mi]), f_loc)
                dv = [j for j, i in enumerate(mi) if depth_views[i]]
                preds = [r["out"]["depth"][j].float().numpy() for j in dv]
                gts = []
                for j in dv:
                    gi = int(mi[j])
                    if gi not in gt_cache:
                        f = str(sd.image_files[idx[gi]])
                        gt_cache[gi] = read_gt_depth(args.dataset, index[f][0], index[f][1], shape=index[f][2])
                    gts.append(gt_cache[gi])
                fm = np.array([bool(f_loc[j]) for j in dv])
                dr = depth_rows(preds, gts, groups=fm)
                row.update({"oom": False, "tokens": cost["tokens"], "gflops": cost["gflops"], "time_s": r["time_s"],
                            "time_runs": times, "timed": repeats > 1, "peak_mem_gb": r["peak_mem_gb"], "auc": aucs,
                            **depth_summary(dr, fm)})
                rows.append(row)
                if kind in ("all_context_res", "focus_only"):
                    keep_tp[kind] = {"w2c": r["w2c"], "mi": mi, "time_s": r["time_s"], "peak_mem_gb": r["peak_mem_gb"],
                                     "timed": repeats > 1,
                                     "tokens": cost["tokens"], "gflops": cost["gflops"],
                                     "preds": {int(mi[j]): p for j, p in zip(dv, preds)}}
                print(f"[scaling] {scene} N={n} {name}: {r['time_s']:.1f} s, {r['peak_mem_gb']:.2f} GB, "
                      f"tokens {cost['tokens']}, focus AUC@30 {100 * aucs.get('focus', {}).get('auc30', float('nan')):.1f}")
                del r
            if len(keep_tp) == 2:   # two-pass control: A = all views at the context res, B = focus views alone
                rows.append(scaling_two_pass_row(keep_tp["all_context_res"], keep_tp["focus_only"], focus, depth_views,
                                                 gt_w2c, gt_cache, scene, n, spec))
            free_cuda()
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    jdump({"weights": args.weights, "dataset": args.dataset, "scenes": args.scenes, "n_views": args.n_views,
           "spec": spec.to_dict(), "gpu": gpu, "rows": rows}, args.out)


# ============================================================================= sec6views (CPU)
def cmd_sec6views(args):
    from da3.enhancements.foveated.adapter import META_FILE
    from da3.enhancements.foveated.metrics import pairwise_pose_errors, view_subset_aucs
    from da3.enhancements.foveated.plan import select_focus
    from da3.eval.gt_depth import GT_DEPTH_DATASETS, gt_depth_index, read_gt_depth
    from da3.eval.harness import evaluation_dir, load_scene_exports, unposed_scenes

    absolutise(args, "out", "bench_root", "work_root", "official_src")
    MV, work_root = open_harness(args)
    work_dir = evaluation_dir(work_root, args.tag)
    rows = []
    for data in args.datasets:
        ds = MV.get(data)()
        for scene, exports in unposed_scenes(work_dir, data).items():
            sc = load_scene_exports(exports)
            meta = dict(np.load(exports / META_FILE)) if (exports / META_FILE).exists() else {}
            S = len(sc["results"]["extrinsics"])
            own = np.asarray(meta["focus"], bool) if "focus" in meta else np.ones(S, bool)
            masks = {"own": own}
            for f in args.fractions:
                masks[f"f{f:g}"] = select_focus(S, f)
            E = pairwise_pose_errors(sc["results"]["extrinsics"], sc["gt_extrinsics"])
            row = {"data": data, "scene": scene, "n_views": S, "masks": {k: v.tolist() for k, v in masks.items()},
                   "auc": {k: view_subset_aucs(E, v) for k, v in masks.items()},
                   **{k: (float(meta[k]) if k in meta else float("nan")) for k in ("tokens", "gflops", "time_s", "peak_mem_gb")},
                   "config": str(meta["config"]) if "config" in meta else args.config}
            if data in GT_DEPTH_DATASETS:
                index = gt_depth_index(data, ds.get_data(scene))
                gts = []
                for f in sc["image_files"]:
                    e = index.get(str(f))
                    gts.append(read_gt_depth(data, e[0], e[1], shape=e[2]) if e else None)
                depth = sc["results"]["depth"]
                row.update(depth_rows([depth[i] for i in range(S)], gts, groups=own))
            rows.append(row)
        print(f"[sec6views] {args.tag} {data}: {sum(r['data'] == data for r in rows)} scenes")
    jdump({"tag": args.tag, "config": args.config, "weights": args.weights, "datasets": args.datasets,
           "fractions": args.fractions, "rows": rows}, args.out)


# ============================================================================= zoom (GPU): gate 2
class Eth3dZip:
    """Selective extraction of ETH3D scenes from DA3-BENCH's eth3d.zip (the full archive is 15 GB;
    full-resolution GT depth maps are ~100 MB each, so only the source views' maps are extracted)."""

    def __init__(self, zip_path: str, bench_root: str):
        self.zf = zipfile.ZipFile(zip_path)
        self.root = Path(bench_root) / "eth3d"
        self.index = defaultdict(dict)   # scene -> {path inside the scene: ZipInfo}
        for info in self.zf.infolist():
            if info.is_dir():
                continue
            parts = info.filename.replace("\\", "/").split("/")
            if "eth3d" in parts:
                parts = parts[parts.index("eth3d") + 1:]
            if len(parts) >= 2:
                self.index[parts[0]]["/".join(parts[1:])] = info

    def extract(self, scene: str, want) -> int:
        n = 0
        for rel, info in self.index.get(scene, {}).items():
            if not want(rel):
                continue
            dst = self.root / scene / rel
            if dst.exists() and dst.stat().st_size == info.file_size:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with self.zf.open(info) as src, open(dst, "wb") as f:
                shutil.copyfileobj(src, f, 1 << 22)
            n += 1
        return n

    def has(self, scene: str, rel: str) -> bool:
        return rel in self.index.get(scene, {})


def render_depth_from_mesh(mesh_path: str, K: np.ndarray, w2c: np.ndarray, hw) -> np.ndarray:
    """Fallback GT: z-depth of the GT mesh seen by a pinhole camera (Open3D ray casting)."""
    import open3d as o3d

    mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(mesh_path))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    H, W = hw
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d_cam = np.stack([uu, vv, np.ones_like(uu)], -1).reshape(-1, 3) @ np.linalg.inv(K).T   # unit z
    R, t = w2c[:3, :3], w2c[:3, 3]
    origin = -R.T @ t
    rays = np.concatenate([np.broadcast_to(origin, d_cam.shape), d_cam @ R], 1).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(H, W)
    return np.where(np.isfinite(hit), hit, 0.0).astype(np.float32)


def cmd_zoom(args):
    from da3.enhancements.foveated import metrics as FM
    from da3.enhancements.foveated import zoom as Z
    from da3.eval.gt_depth import eth3d_gt_paths, read_gt_depth
    from da3.io_utils import preprocess_like_official, read_rgb, scale_intrinsics

    from da3.enhancements.foveated.analysis import load_plan, zoom_gates

    absolutise(args, "out", "bench_root", "work_root", "official_src", "eth3d_zip", "fig_dir", "plan")
    plan = load_plan(args.plan)
    zf = Eth3dZip(args.eth3d_zip, args.bench_root) if args.eth3d_zip else None
    if zf is not None:
        for scene in args.scenes:
            if scene not in zf.index:
                print(f"[zoom] WARNING: scene {scene!r} not found in {args.eth3d_zip} (expected members "
                      f"'eth3d/<scene>/...'); scenes seen in the zip: {sorted(zf.index)[:20]}")
                continue
            n = zf.extract(scene, lambda rel: rel.startswith(("images/", "dslr_calibration_jpg/")))
            print(f"[zoom] {scene}: extracted {n} image / calibration files")
    MV, _ = open_harness(args)
    device = torch.device(args.device)
    model = load_model(args.weights, device)
    warmup(model, device)
    ds = MV.get("eth3d")()
    fig_dir = Path(args.fig_dir) if args.fig_dir else None
    rows, examples = [], []
    for scene in args.scenes:
        sd = ds.get_data(scene)
        if len(sd.image_files) < 3:
            print(f"[zoom] {scene}: too few images, skipped")
            continue
        sel = harness_sample(len(sd.image_files), args.max_frames)
        files = [str(sd.image_files[i]) for i in sel]
        gt_w2c = np.asarray(sd.extrinsics, np.float64)[sel]
        gt_K = np.asarray(sd.intrinsics, np.float64)[sel]
        n = len(files)
        src = [int(i) for i in np.unique(np.round(np.linspace(0, n - 1, args.views_per_scene + 2))[1:-1].astype(int))]
        if zf is not None:   # GT depth + masks of the source views only (~100 MB per map)
            stems = {os.path.splitext(os.path.basename(files[v]))[0] for v in src}
            zf.extract(scene, lambda rel: rel.startswith(("ground_truth_depth/", "masks_for_images/"))
                       and os.path.splitext(os.path.basename(rel))[0] in stems)
        # one read of every full-resolution photo: keep the 504 / 896 versions, the full image only for sources
        img504, img896, full, orig = [], [], {}, []
        for v, f in enumerate(files):
            im = read_rgb(f)
            orig.append(im.shape[:2])
            img504.append(preprocess_like_official(im, args.res))
            img896.append(preprocess_like_official(im, args.res_hi))
            if v in src:
                full[v] = im
            del im
        hw504 = img504[0].shape[:2]

        def views_of(imgs):
            return prep_views(imgs, [args.res] * len(imgs))

        keep = ("depth", "ray")
        base = views_of(img504)
        rA = run_pass(model, base, device, keep=keep)
        if rA["oom"]:
            print(f"[zoom] {scene}: out of memory at {args.res} px, skipped")
            continue
        rB = run_pass(model, prep_views(img896, [args.res_hi] * n), device, keep=("depth",))
        print(f"[zoom] {scene}: {n} views; global {args.res} px {rA['time_s']:.1f} s, "
              f"{args.res_hi} px " + ("OOM" if rB["oom"] else f"{rB['time_s']:.1f} s"))
        for v in src:
            H0, W0 = orig[v]
            K0 = gt_K[v]
            dpath, mpath = eth3d_gt_paths(files[v])
            gt = read_gt_depth("eth3d", dpath, mpath, shape=(H0, W0)) if os.path.exists(dpath) else None
            gt_src = "ETH3D GT depth map"
            if gt is None or not (gt > 0).any():
                mesh = Path(files[v].replace("\\", "/").split("/images/")[0]) / "combined_mesh.ply"
                if not mesh.exists() and zf is not None and zf.has(scene, "combined_mesh.ply"):
                    zf.extract(scene, lambda rel: rel == "combined_mesh.ply")
                if not mesh.exists():
                    print(f"[zoom] {scene} view {v}: no GT depth and no mesh, skipped")
                    continue
                gt = render_depth_from_mesh(str(mesh), K0, gt_w2c[v], (H0, W0))
                gt_src = "rendered from the GT mesh"
            picks = {}   # box -> selectors that chose it (each distinct box is evaluated once)
            for sel in args.selectors:
                seed = zlib.crc32(f"{scene}:{v}".encode())        # deterministic "random" selector
                for b in Z.select_regions(gt, zoom=args.zoom, n=args.regions_per_view, mode=sel, image=full[v], seed=seed):
                    picks.setdefault(tuple(int(t) for t in b), []).append(sel)
            for box, selectors in picks.items():
                zv = Z.zoom_view(K0, box, hw504)
                zimg = Z.render_zoom(full[v], K0, zv)
                cimg = Z.crop_view(full[v], box, hw504)
                pts = Z.box_points(box, hw504)
                gt_pts = Z.nearest(gt, pts)
                rays = Z.source_rays(K0, pts)
                preds, scene_scale, rv = {}, {}, {}

                def src_scale(depth_src):
                    """Scale of the source view's whole depth map vs GT (same pass), GT at the cell centres."""
                    h, w = depth_src.shape
                    rows_ = ((np.arange(h) + 0.5) * H0 / h).astype(int)
                    cols_ = ((np.arange(w) + 0.5) * W0 / w).astype(int)
                    return FM.pooled_scale([depth_src], [gt[rows_[:, None], cols_[None]]])

                def rv_check(r, zoom_index):
                    Rs, Rz = r["w2c"][v][:3, :3], r["w2c"][zoom_index][:3, :3]
                    out = {"rv_err_head_deg": FM.rot_angle_deg(Rs @ Rz.T, zv.R_v),
                           "zoom_focal_err": float(abs(np.log(r["K"][zoom_index][0, 0] / zv.K[0, 0])))}
                    if "ray" in r["out"]:
                        from da3.enhancements.foveated import cameras_from_rays

                        e, _ = cameras_from_rays([r["out"]["ray"][v], r["out"]["ray"][zoom_index]], [hw504, hw504])
                        e = e.numpy()
                        out["rv_err_ray_deg"] = FM.rot_angle_deg(e[0][:3, :3] @ e[1][:3, :3].T, zv.R_v)
                    return out

                d_src = rA["out"]["depth"][v].float().numpy()
                preds[f"global_{args.res}"] = Z.depth_from_global(d_src, (H0, W0), pts)
                scene_scale[f"global_{args.res}"] = src_scale(d_src)
                if not rB["oom"]:
                    d_hi = rB["out"]["depth"][v].float().numpy()
                    preds[f"global_{args.res_hi}"] = Z.depth_from_global(d_hi, (H0, W0), pts)
                    scene_scale[f"global_{args.res_hi}"] = src_scale(d_hi)
                extra_c, extra_z = views_of([cimg])[0], views_of([zimg])[0]
                r = run_pass(model, base + [extra_c], device, keep=keep)
                if not r["oom"]:
                    preds["crop_unposed"] = Z.depth_from_crop(r["out"]["depth"][-1].float().numpy(), box, pts)
                    scene_scale["crop_unposed"] = src_scale(r["out"]["depth"][v].float().numpy())
                r = run_pass(model, [extra_c], device)
                if not r["oom"]:
                    preds["crop_mono"] = Z.depth_from_crop(r["out"]["depth"][0].float().numpy(), box, pts)
                r = run_pass(model, base + [extra_z], device, keep=keep)
                if not r["oom"]:
                    preds["zoom_unposed"] = Z.depth_from_zoom(r["out"]["depth"][-1].float().numpy(), zv, K0, pts)
                    scene_scale["zoom_unposed"] = src_scale(r["out"]["depth"][v].float().numpy())
                    rv["zoom_unposed"] = rv_check(r, n)
                r = run_pass(model, [extra_z], device)
                if not r["oom"]:
                    preds["zoom_mono"] = Z.depth_from_zoom(r["out"]["depth"][0].float().numpy(), zv, K0, pts)
                # posed: DA3's own cameras (pass 1) for the scene views, the exact zoom camera derived from them
                Kp = rA["K"]
                f_ratio = zv.K[0, 0] / K0[0, 0]
                f_src_full = Kp[v][0, 0] * W0 / hw504[1]
                Kz = np.array([[f_src_full * f_ratio, 0, hw504[1] / 2], [0, f_src_full * f_ratio, hw504[0] / 2], [0, 0, 1]])
                w2c_p = np.concatenate([np.stack([np.vstack([e, [0, 0, 0, 1]]) for e in rA["w2c"]]),
                                        Z.zoom_w2c(rA["w2c"][v], zv.R_v)[None]])
                r = run_pass(model, base + [extra_z], device, keep=keep, w2c=torch.from_numpy(w2c_p).float(),
                             K=torch.from_numpy(np.concatenate([Kp, Kz[None]])).float())
                if not r["oom"]:
                    preds["zoom_posed_da3cams"] = Z.depth_from_zoom(r["out"]["depth"][-1].float().numpy(), zv, K0, pts)
                    scene_scale["zoom_posed_da3cams"] = src_scale(r["out"]["depth"][v].float().numpy())
                    rv["zoom_posed_da3cams"] = rv_check(r, n)
                # posed with the GT cameras (oracle cameras)
                K_gt = np.stack([scale_intrinsics(K, o, hw504) for K, o in zip(gt_K, orig)])
                w2c_g = np.concatenate([gt_w2c, Z.zoom_w2c(gt_w2c[v], zv.R_v)[None]])
                r = run_pass(model, base + [extra_z], device, keep=keep, w2c=torch.from_numpy(w2c_g).float(),
                             K=torch.from_numpy(np.concatenate([K_gt, zv.K[None]])).float())
                if not r["oom"]:
                    preds["zoom_posed_gtcams"] = Z.depth_from_zoom(r["out"]["depth"][-1].float().numpy(), zv, K0, pts)
                    scene_scale["zoom_posed_gtcams"] = src_scale(r["out"]["depth"][v].float().numpy())
                    rv["zoom_posed_gtcams"] = rv_check(r, n)
                del r
                # metrics on the points valid for every method
                common = (gt_pts > 0) & np.isfinite(gt_pts)
                for z in preds.values():
                    common &= np.isfinite(z) & (z > 0)
                row = {"scene": scene, "view": v, "image": os.path.basename(files[v]), "box": list(box),
                       "gt_source": gt_src, "valid_points": int(common.sum()),
                       "rv_true_deg": float(np.degrees(np.arccos(np.clip(zv.R_v[2, 2], -1, 1)))),
                       "zoom_factor": float(zv.K[0, 0] / (K0[0, 0] * hw504[1] / W0)), "methods": {}}
                if common.sum() < 1000:
                    row["skipped"] = "fewer than 1000 common valid points"
                    rows += [{**row, "selector": s} for s in selectors]
                    continue
                g = gt_pts[common]
                for mname, z in preds.items():
                    zc = z[common]
                    e = FM.depth_errors(zc, g)
                    m = {"abs_rel": e["abs_rel"], "delta1": e["delta1"], "delta105": e["delta105"]}
                    m.update(FM.region_point_metrics(zc * e["scale"], g, rays[common]))
                    if mname in scene_scale and np.isfinite(scene_scale[mname]):
                        m["abs_rel_scene"] = FM.depth_errors(zc, g, scale=scene_scale[mname])["abs_rel"]
                    m.update(rv.get(mname, {}))
                    row["methods"][mname] = m
                rows += [{**row, "selector": s} for s in selectors]
                print(f"[zoom] {scene} view {v} box {box} ({', '.join(selectors)}): " + ", ".join(
                    f"{k} {m['abs_rel']:.4f}" for k, m in row["methods"].items()))
                if fig_dir is not None and len(examples) < args.n_examples:
                    examples.append({"title": f"{scene} · {os.path.basename(files[v])}", "full": full[v][::8, ::8].copy(),
                                     "box": box, "zoom": zimg, "gt": np.where(common, gt_pts, np.nan).reshape(hw504),
                                     "pred": {k: np.where(common, preds[k] * FM.depth_errors(preds[k][common], g)["scale"],
                                                          np.nan).reshape(hw504)
                                              for k in (f"global_{args.res}", f"global_{args.res_hi}", "zoom_posed_da3cams")
                                              if k in preds}})
        del rA, rB, full, img504, img896
        free_cuda()
    if fig_dir is not None and examples:
        zoom_figure(examples, fig_dir / "zoom_examples.png")
    jdump({"weights": args.weights, "scenes": args.scenes, "zoom": args.zoom, "res": args.res, "res_hi": args.res_hi,
           "selectors": args.selectors, "plan": plan["path"], "plan_sha256": plan["sha256"], "rows": rows,
           "analysis": {"plan_sha256": plan["sha256"], "gates": zoom_gates(rows, plan)}}, args.out)
    print_gates(zoom_gates(rows, plan))


def zoom_methods_mean(rows) -> dict:
    """Per method: mean of the region metrics, median of the camera checks (R_v, zoom focal)."""
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m, vals in r.get("methods", {}).items():
            for k, v in vals.items():
                agg[m][k].append(v)
    out = {}
    for m, d in agg.items():
        s = {}
        for k, v in d.items():
            v = np.asarray(v, np.float64)
            if k.startswith(("rv_", "zoom_focal")):
                s[k] = float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")
            else:
                s[k] = nanmean(v)
        s["regions"] = len(d.get("abs_rel", []))
        out[m] = s
    return out


def zoom_figure(examples, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 3 + max(len(e["pred"]) for e in examples)
    fig, axes = plt.subplots(len(examples), cols, figsize=(3.2 * cols, 2.4 * len(examples)), squeeze=False)
    for ax_row, e in zip(axes, examples):
        x0, y0, w, h = e["box"]
        ax_row[0].imshow(e["full"])
        ax_row[0].add_patch(plt.Rectangle((x0 / 8, y0 / 8), w / 8, h / 8, fill=False, ec="r", lw=1.5))
        ax_row[0].set_title(e["title"], fontsize=8)
        ax_row[1].imshow(e["zoom"])
        ax_row[1].set_title("virtual zoom view", fontsize=8)
        lo, hi = np.nanpercentile(e["gt"], [2, 98])
        ax_row[2].imshow(e["gt"], cmap="turbo", vmin=lo, vmax=hi)
        ax_row[2].set_title("GT depth (region)", fontsize=8)
        for ax, (k, p) in zip(ax_row[3:], e["pred"].items()):
            ax.imshow(p, cmap="turbo", vmin=lo, vmax=hi)
            ax.set_title(k, fontsize=8)
        for ax in ax_row:
            ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)


# ============================================================================= analyse (CPU)
def md_table(head, rows) -> str:
    def fmt(v):
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            return "–" if not np.isfinite(v) else f"{v:.4g}"
        return str(v)

    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


PROBE_COLS = [("AbsRel focus", "abs_rel", "focus"), ("AbsRel context", "abs_rel", "context"),
              ("AbsRel all", "abs_rel", "all"), ("AbsRel scene-scale", "abs_rel_scene", "all"),
              ("δ1 focus", "delta1", "focus"), ("AUC@3 all", "auc03", "all"), ("AUC@30 all", "auc30", "all"),
              ("AUC@30 focus", "auc30", "focus"), ("AUC@30 context", "auc30", "context"),
              # exploratory diagnostics (*): not pre-registered, added after the limited run
              ("AUC@30 cross*", "auc30", "cross"), ("AUC@30 cross rot*", "auc30_rot", "cross"),
              ("AUC@30 cross trans*", "auc30_trans", "cross"), ("AbsRel group-scale*", "abs_rel_group", "all"),
              ("scale ctx/focus*", "scale_ratio", "all"),
              ("tokens", "tokens", "all"),
              ("GFLOPs", "gflops", "all"), ("time (s)", "time_s", "all"), ("peak mem (GB)", "peak_mem_gb", "all")]


def config_order(cfg: str):
    """Uniform configurations from high to low resolution, then the foveated ones."""
    return (0, -int(cfg[1:]), "") if cfg.startswith("u") and cfg[1:].isdigit() else (1, 0, cfg)


def probe_agg(rows) -> dict:
    """{(dataset | 'all', config): {column: mean over scenes}} of the probe rows (AUC in %); each
    configuration on its own focus mask (the uniform rows on the evenly spaced one)."""
    from da3.enhancements.foveated.analysis import index_rows, scene_values

    index = index_rows(rows)
    configs = sorted({c for (_, _, _, c) in index}, key=config_order)
    datas = sorted({d for (d, _, _, _) in index})
    out = {}
    for cfg in configs:
        per = {lab: scene_values(index, cfg, metric, part) for lab, metric, part in PROBE_COLS}
        for d in ["all"] + datas:
            m = {lab: (100.0 if metric.startswith("auc") else 1.0)
                 * nanmean([x for (dd, _), x in per[lab].items() if d == "all" or dd == d])
                 for lab, metric, part in PROBE_COLS}
            m["scenes"] = sum(1 for (dd, _) in per["AbsRel all"] if d == "all" or dd == d)
            out[(d, cfg)] = m
    return out


def gates_flat(gates: dict) -> dict:
    """Scalar fields only, for the result records' tables (the full gates are in gates.json)."""
    return {k: {f: g.get(f, "") for f in ("result", "role", "value", "criterion")} for k, g in gates.items()}


STATUS_COLORS = {"PASS": "#0ca30c", "FAIL": "#d03b3b", "INCONCLUSIVE": "#fab219", "n.a.": "#c3c2b7"}   # status palette


def forest_entries(gates: dict) -> list:
    """Every ratio criterion of the gates: label, estimate, CI, decision threshold, result."""
    out = []
    for name, g in gates.items():
        for key in ("G1a", "G1b", "C1", "C2"):
            c = g.get(key)
            if not isinstance(c, dict) or "threshold" not in c or c.get("unit") == "deg":
                continue
            role = g.get("role", "")
            out.append({"label": f"{name.split(' (')[0]} · {c['label']}" + (f" [{role}]" if role != "primary" else ""),
                        "estimate": c["estimate"], "lo": c["ci_lo"], "hi": c["ci_hi"], "threshold": c["threshold"],
                        "result": c["result"]})
    return out


def forest_figure(entries, path, title):
    """One row per criterion: CI bar and estimate coloured by the decision (status palette, with the
    decision written next to it), dashed tick = decision threshold, grey line = no effect (1.0)."""
    import matplotlib.pyplot as plt

    if not entries:
        return None
    ink, axis_c, grid_c = "#0b0b0b", "#c3c2b7", "#e1e0d9"
    n = len(entries)
    fig, ax = plt.subplots(figsize=(11, 0.45 * n + 1.5))
    for i, e in enumerate(entries):
        y = n - 1 - i
        col = STATUS_COLORS.get(e["result"], axis_c)
        if np.isfinite(e["lo"]) and np.isfinite(e["hi"]):
            ax.plot([e["lo"], e["hi"]], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
        if np.isfinite(e["estimate"]):
            ax.plot(e["estimate"], y, "o", ms=8, color=col, mec="#fcfcfb", mew=1.5, zorder=3)
        ax.plot([e["threshold"]] * 2, [y - 0.32, y + 0.32], color=ink, lw=1.2, ls=(0, (2, 2)), zorder=2)
        ax.text(1.01, y, e["result"], transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=ink)
    ax.axvline(1.0, color=axis_c, lw=1, zorder=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([e["label"] for e in entries][::-1], fontsize=8)
    ax.set_xlabel("ratio test / reference · dot = estimate, bar = CI, dashed tick = decision threshold", fontsize=9)
    ax.grid(axis="x", color=grid_c, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(axis_c)
    ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def margins_table(an: dict) -> tuple[str, dict]:
    nf, m = an["noise_floor"], an["margins"]
    rows, extra = [], {}
    for fam, lab in (("depth", "depth (AbsRel)"), ("pose", "pose (AUC@30)")):
        r = nf[fam]
        note = ("noise floor above the cap: the protocol cannot resolve the pre-registered tolerance"
                if r.get("above_cap") else "ok")
        rows.append([lab, f"{r['estimate']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]", 100 * r["floor"], 100 * m[fam],
                     r["scenes"], note])
        extra[lab] = {"A/A ratio": r["estimate"], "A/A CI lo": r["ci_lo"], "A/A CI hi": r["ci_hi"],
                      "noise floor (%)": 100 * r["floor"], "margin used (%)": 100 * m[fam], "status": note}
    return md_table(["metric family", "A/A ratio (reference view moved) [CI]", "noise floor (%)", "margin used (%)",
                     "scenes", "status"], rows), extra


def _ci_cell(est, lo, hi, ratio=True):
    if not np.isfinite(est):
        return "–"
    return f"{est:.3f} [{lo:.3f}, {hi:.3f}]" if ratio else f"{est:+.1f} [{lo:+.1f}, {hi:+.1f}]"


def sec6_ci_table(sv_files, group, plan) -> tuple[str, dict]:
    """Paired scene-bootstrap CIs of every foveated Sec. 6 run vs the original DA3 (504 px) and vs the
    uniform-resolution curve at the same tokens (interpolated per scene in log tokens). Descriptive:
    the report defines no Sec. 6 gate."""
    from da3.enhancements.foveated import stats as S

    bs, level = plan["bootstrap"], plan["ci_level"]
    per, tags = {}, {}
    for f in sv_files:
        js = jload(f)
        tags[js["config"]] = js["tag"]
        d = {}
        for r in js["rows"]:
            v = {"tokens": r.get("tokens", np.nan)}
            if "abs_rel" in r:
                ar = np.asarray(r["abs_rel"], np.float64)
                for mk, mask in r["masks"].items():
                    v[f"abs_rel_focus@{mk}"] = nanmean(ar[np.asarray(mask, bool)])
            d[(r["data"], r["scene"])] = v
        per[js["config"]] = d
    rec = {r["tag"]: r for r in group}

    def harness(cfg, mode, key):
        r, out = rec.get(tags.get(cfg)), {}
        if r is None:
            return out
        for dk, scenes in r.get("sec6_per_scene", {}).items():
            if dk.endswith("_" + mode) and isinstance(scenes, dict):
                for s, m in scenes.items():
                    if s != "mean" and isinstance(m, dict) and m.get(key) is not None:
                        out[(dk[: -len(mode) - 1], s)] = 100.0 * float(m[key])
        return out

    uni = sorted((c for c in per if c.startswith("u") and c[1:].isdigit()), key=lambda c: int(c[1:]))
    ref = "u504" if "u504" in per else (uni[-1] if uni else None)
    rows_md, extra = [], {}
    for cfg in sorted(c for c in per if c not in uni):
        fr = config_fraction(cfg)
        mk = f"f{fr:g}" if fr is not None else "own"
        metrics = [("AbsRel focus (ratio)", lambda c: {k: v.get(f"abs_rel_focus@{mk}", np.nan) for k, v in per[c].items()}, True),
                   ("AUC@3 (pts)", lambda c: harness(c, "pose", "auc03"), False),
                   ("AUC@30 (pts)", lambda c: harness(c, "pose", "auc30"), False),
                   ("F1 w/o poses (pts)", lambda c: harness(c, "recon_unposed", "fscore"), False)]
        for lab, fn, is_ratio in metrics:
            t = fn(cfg)
            cells, n_sc = [cfg, lab], 0
            for comp in ("original", "equal"):
                if comp == "original":
                    b = fn(ref) if ref else {}
                    ks = [k for k in sorted(set(t) & set(b)) if np.isfinite(t[k]) and np.isfinite(b[k])]
                    a_, b_ = [t[k] for k in ks], [b[k] for k in ks]
                else:
                    curves = {c: fn(c) for c in uni}
                    a_, b_ = [], []
                    for k, val in sorted(t.items()):
                        pts = [(per[c].get(k, {}).get("tokens", np.nan), curves[c].get(k, np.nan)) for c in uni]
                        u = interp_equal_budget(pts, per[cfg].get(k, {}).get("tokens", np.nan))
                        if np.isfinite(val) and np.isfinite(u):
                            a_.append(val)
                            b_.append(u)
                est, lo, hi = (S.ratio_of_means if is_ratio else S.mean_difference)(
                    a_, b_, n=bs["n"], seed=bs["seed"], level=level)
                cells.append(_ci_cell(est, lo, hi, is_ratio))
                n_sc = max(n_sc, len(a_))
                extra.setdefault(f"{cfg} · {lab}", {})[f"vs {'original' if comp == 'original' else 'equal tokens'}"] = \
                    _ci_cell(est, lo, hi, is_ratio)
            cells.append(n_sc)
            rows_md.append(cells)
    head = ["config", "metric", f"vs original DA3 ({ref}) [{int(100 * level)}% CI]",
            f"vs uniform at equal tokens [{int(100 * level)}% CI]", "scenes"]
    return md_table(head, rows_md), extra


def scaling_ci_table(sc, plan) -> tuple[str, dict]:
    """Foveated vs every alternative at each view count, paired by scene: Δ AUC@30 of the focus
    pairs, AbsRel ratio of the focus views (scene-bootstrap CIs), and the time ratio on the scenes
    with repeated timings (median and range)."""
    from da3.enhancements.foveated import stats as S

    bs, level = plan["bootstrap"], plan["ci_level"]
    by = defaultdict(dict)
    for r in sc["rows"]:
        if not r.get("oom"):
            by[(r["n_views"], r["kind"])][r["scene"]] = r
    rows_md, extra = [], {}
    for n in sorted({k[0] for k in by}):
        fov = by.get((n, "foveated"), {})
        for kind in SCALING_KINDS:
            if kind == "foveated" or (n, kind) not in by or not fov:
                continue
            other = by[(n, kind)]
            ks = sorted(set(fov) & set(other))
            a = [100 * fov[k]["auc"].get("focus", {}).get("auc30", np.nan) for k in ks]
            b = [100 * other[k]["auc"].get("focus", {}).get("auc30", np.nan) for k in ks]
            d = S.mean_difference(a, b, n=bs["n"], seed=bs["seed"], level=level)
            r_ = S.ratio_of_means([fov[k]["abs_rel_focus"] for k in ks], [other[k]["abs_rel_focus"] for k in ks],
                                  n=bs["n"], seed=bs["seed"], level=level)
            sp = [other[k]["time_s"] / fov[k]["time_s"] for k in ks if fov[k].get("timed") and other[k].get("timed")]
            sp_txt = f"{np.median(sp):.2f} ({min(sp):.2f}–{max(sp):.2f})" if sp else "–"
            rows_md.append([n, kind, _ci_cell(*d, ratio=False), _ci_cell(*r_), sp_txt, len(ks)])
            extra[f"N={n} · foveated vs {kind}"] = {"Δ AUC@30 focus (pts)": d[0], "CI lo": d[1], "CI hi": d[2],
                                                    "AbsRel focus ratio": r_[0], "time ratio (median)": float(np.median(sp)) if sp else np.nan}
    head = ["views", "foveated vs", f"Δ AUC@30 focus pairs, pts [{int(100 * level)}% CI]", "AbsRel focus ratio [CI]",
            "time other / foveated (timed scenes)", "scenes"]
    return md_table(head, rows_md), extra


def zoom_selector_table(rows) -> tuple[str, dict]:
    cols = ["abs_rel", "abs_rel_scene", "delta105", "F1@0.5%", "F1@1%", "F1@2%", "chamfer_%",
            "rv_err_head_deg", "rv_err_ray_deg", "zoom_focal_err", "regions"]
    out_rows, extra = [], {}
    for sel in sorted({r.get("selector", "oracle") for r in rows}):
        agg = zoom_methods_mean([r for r in rows if r.get("selector", "oracle") == sel])
        for m, v in agg.items():
            out_rows.append([sel, m] + [v.get(c, np.nan) for c in cols])
            extra[f"{sel} · {m}"] = {c: v.get(c, np.nan) for c in cols}
    return md_table(["regions", "method"] + cols, out_rows), extra


def nanmax(x):
    x = np.asarray(x, np.float64)
    return float(np.nanmax(x)) if np.isfinite(x).any() else float("nan")


def readable_log_axis(ax, axis="x", ticks=None):
    """Log axis with plain-number labels at 1-2-5 steps (or at the given ticks)."""
    from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter, ScalarFormatter

    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    a.set_major_locator(FixedLocator(ticks) if ticks is not None else LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    a.set_major_formatter(fmt)
    a.set_minor_formatter(NullFormatter())


SCALING_KINDS = ["all_focus_res", "foveated", "foveated_bal", "uniform_equal_budget", "fewer_views_equal_budget",
                 "all_context_res", "focus_only", "two_pass"]   # the last two: exploratory (--two-pass)


def interp_equal_budget(points, tokens):
    """Value of a uniform-resolution curve [(tokens, value)] at `tokens` (linear in log tokens)."""
    pts = sorted((t, v) for t, v in points if np.isfinite(t) and np.isfinite(v))
    if len(pts) < 2 or not (pts[0][0] <= tokens <= pts[-1][0]):
        return float("nan")
    return float(np.interp(np.log(tokens), np.log([p[0] for p in pts]), [p[1] for p in pts]))


def config_fraction(cfg: str):
    """'fov25_504-252_bal' -> 0.25; uniform configurations -> None."""
    import re

    m = re.match(r"^fov(\d+)_", cfg)
    return int(m.group(1)) / 100.0 if m else None


def sec6views_aggregate(files) -> dict:
    """{config: {"tag", "masks", "data": {data | 'all': metrics}}} from sec6views JSONs (one per run).
    Metric keys: '<metric>_<focus|context|all>@<mask>' with mask 'own' (the run's focus views) or
    'f<fraction>' (the stride focus set of that fraction, comparable across runs)."""
    out = {}
    for f in files:
        js = jload(f)
        rows = js["rows"]
        mask_keys = sorted({k for r in rows for k in r["masks"]})
        res = {}
        for d in sorted({r["data"] for r in rows}) + ["all"]:
            rs = [r for r in rows if d == "all" or r["data"] == d]
            m = {"scenes": len(rs)}
            for k in ("tokens", "gflops", "time_s", "peak_mem_gb"):
                m[k] = nanmean([r[k] for r in rs])
            for mk in mask_keys:
                for part in ("all", "focus", "context"):
                    for thr in ("auc03", "auc30"):
                        m[f"{thr}_{part}@{mk}"] = nanmean([r["auc"][mk][part][thr] for r in rs if part in r["auc"][mk]])
                vals = defaultdict(list)
                for r in rs:
                    if "abs_rel" not in r:
                        continue
                    sel = np.asarray(r["masks"][mk], bool)
                    vals["focus"] += list(np.asarray(r["abs_rel"], np.float64)[sel])
                    vals["context"] += list(np.asarray(r["abs_rel"], np.float64)[~sel])
                    vals["all"] += list(np.asarray(r["abs_rel"], np.float64))
                for part, v in vals.items():
                    m[f"abs_rel_{part}@{mk}"] = nanmean(v)
            res[d] = m
        out[js["config"]] = {"tag": js["tag"], "masks": mask_keys, "data": res}
    return out


def cmd_analyse(args):
    import matplotlib

    matplotlib.use("Agg")

    from da3.compare import load_records, make_record, protocol, save_record
    from da3.compare.tables import sec6_markdown
    from da3.enhancements.foveated.analysis import load_plan, probe_exploratory, probe_gates, zoom_gates
    from da3.train_utils import environment_info, git_commit

    out_dir = Path(args.out_dir).resolve()
    results_root = Path(args.results_root).resolve()
    plan = load_plan(Path(args.plan).resolve())
    level = int(round(100 * plan["ci_level"]))
    commit, env = git_commit(REPO), environment_info()
    records = load_records(results_root)
    md = ["# Foveated DA3 (enhancement #2) · results\n",
          "Setting **[official]**: released weights, network unchanged; one forward pass with focus views at full "
          "resolution and context views at lower resolution (docs/da3_enhancement_report.md §4.3). Probe, large-view "
          "and zoom numbers come from new protocols and are never mixed with Sec. 6.\n",
          f"Pre-registered analysis plan `{Path(plan['path']).name}` v{plan['version']}, SHA-256 `{plan['sha256'][:12]}` "
          f"(docs/FOVEATED_ANALYSIS_PLAN.md). Unit of analysis = scene; {level}% scene-bootstrap CIs; every gate is "
          "PASS / FAIL / INCONCLUSIVE from its CI, with the margins at which the decision would change.\n"]
    gates_all, files, warnings = {}, [], []
    meta = {"plan_sha256": plan["sha256"], "plan_version": plan["version"]}

    for model in args.models:
        weights = f"official:{model}"
        md.append(f"\n## {model}\n")
        # ---- parity
        p = out_dir / f"parity_{model}.json"
        if p.exists():
            par = jload(p)
            md.append("### Routing check (equal resolutions = original DA3)\n")
            md.append(md_table(["check", "bit-identical", "max rel. diff", "result"],
                               [[k, c["bit_identical"], c["max_rel_diff"], "PASS" if c["pass"] else "FAIL"]
                                for k, c in par["checks"].items()]))
            gates_all[f"{model} · G0 routing parity"] = {
                "result": "PASS" if par["all_pass"] else "FAIL", "role": "primary",
                "value": f"{sum(c['pass'] for c in par['checks'].values())}/{len(par['checks'])} checks",
                "criterion": "equal resolutions bit-identical to DA3.forward; mixed-resolution equivariance"}
        # ---- probe: gates 1 and 3
        p = out_dir / f"probe_{model}.json"
        if p.exists():
            pr = jload(p)
            if pr.get("plan_sha256") and pr["plan_sha256"] != plan["sha256"]:
                warnings.append(f"{model}: the probe ran under plan {pr['plan_sha256'][:12]}, this analysis uses "
                                f"{plan['sha256'][:12]}: the plan was changed after the run.")
            an = probe_gates(pr["rows"], plan)
            agg = probe_agg(pr["rows"])
            n_sc = len({(r["data"], r["scene"]) for r in pr["rows"]})
            md.append(f"### Probe: {pr['n_views']} views per subset, {pr.get('subsets', 1)} subsets per scene, "
                      f"{n_sc} scenes ({', '.join(NAMES.get(d, d) for d in pr['datasets'])})\n")
            mt, m_extra = margins_table(an)
            md.append("**Margins** (pre-registered 3%, raised to the noise floor of the treatment-blind A/A run if "
                      "larger):\n")
            md.append(mt)
            md.append("**Decisions:**\n")
            md.append(md_table(["gate", "role", "result", f"estimate [{level}% CI] → decision (margins that change it)"],
                               [[k, g.get("role", ""), g["result"], g["value"]] for k, g in an["gates"].items()]))
            md.append("Means over scenes (AbsRel median-aligned per view; AUC in %; each configuration on its own focus "
                      "views, the uniform rows on the evenly spaced ones):\n")
            md.append(md_table(["dataset · config"] + [c[0] for c in PROBE_COLS] + ["scenes"],
                               [[f"{NAMES.get(d, d)} · {c}"] + list(v.values()) for (d, c), v in agg.items()]))
            md.append("Columns marked * are exploratory diagnostics (not pre-registered): `cross` = focus–context pairs, "
                      "`rot` / `trans` = rotation-only / translation-only errors; group-scale = one depth scale per "
                      "resolution group instead of one per scene; scale ctx/focus = median per-view depth scale of the "
                      "context views over that of the focus views (1 = consistent).\n")
            expl = probe_exploratory(pr["rows"], plan)
            if expl:
                md.append(f"**Exploratory diagnostics** (added after the limited run of 2026-11-23; descriptive, "
                          f"{level}% scene-bootstrap CIs, no decision; see the plan's deviations log):\n")
                md.append(md_table(["comparison", f"estimate [{level}% CI]", "scenes"],
                                   [[k, _ci_cell(e["estimate"], e["ci_lo"], e["ci_hi"], ratio=e.get("unit") != "pts"),
                                     e["scenes"]] for k, e in expl.items()]))
            for k, g in an["gates"].items():
                gates_all[f"{model} · {k}"] = g
            extra = {"probe": {f"{NAMES.get(d, d)} · {c}": v for (d, c), v in agg.items()},
                     "gates": gates_flat(an["gates"]), "margins": m_extra,
                     "exploratory": {k: {f: e[f] for f in ("estimate", "ci_lo", "ci_hi", "scenes")}
                                     for k, e in expl.items()}}
            save_record(make_record(
                enhancement="foveated", variant="probe", tag=f"fov_{model}__probe{pr['n_views']}",
                setting="official", weights=weights,
                protocol=protocol(pr["datasets"], "all" if not pr.get("max_scenes") else
                                  [f"first {pr['max_scenes']} per dataset"], pr["n_views"], ["probe"], "first"),
                extra=extra, commit=commit, env={**env, **meta},
                notes=f"{pr['n_views']} views x {pr.get('subsets', 1)} subsets per scene; plan {plan['sha256'][:12]}"),
                results_root)
            files.append(probe_figure(agg, pr["datasets"], out_dir / f"probe_{model}.png"))
            files.append(forest_figure(forest_entries(an["gates"]), out_dir / f"gates_{model}.png",
                                       f"{model}: probe gates ({level}% scene-bootstrap CIs)"))
        # ---- Sec. 6 (records from eval_bench.py) + per-view analysis of the same runs
        sv = sorted(out_dir.glob(f"sec6views_{args.tag_prefix}_{model}_*.json"))
        if sv:
            sagg = sec6views_aggregate(sv)
            tags = {c["tag"] for c in sagg.values()}
            group = [r for r in records if r["tag"] in tags and r.get("sec6")]
            md.append("### Sec. 6 protocol (released harness, ≤ 100 frames, seed 42)\n")
            md.append(sec6_markdown(group) if group else "_no Sec. 6 records found_\n")
            masks = sorted({mk for c in sagg.values() for mk in c["masks"] if mk.startswith("f")},
                           key=lambda s: -float(s[1:]))
            md.append("Per-view analysis of the same runs. AbsRel median-aligned per view; `f=x` = the evenly spaced focus "
                      "set of fraction x (the focus views of the foveated runs), applied to every row so that rows are "
                      "comparable; tokens / time / memory are per scene (mean; memory: max).\n")
            head = ["config", "tokens", "GFLOPs", "time (s)", "peak mem (GB)"]
            for mk in masks:
                head += [f"AbsRel focus ({mk})", f"AbsRel context ({mk})", f"AUC@30 focus ({mk})"]
            rows_md, extra = [], {}
            for cfg in uniform_configs(sagg) + sorted(c for c in sagg if c not in uniform_configs(sagg)):
                m = sagg[cfg]["data"]["all"]
                row = [cfg, m["tokens"], m["gflops"], m["time_s"], m["peak_mem_gb"]]
                for mk in masks:
                    row += [m.get(f"abs_rel_focus@{mk}", np.nan), m.get(f"abs_rel_context@{mk}", np.nan),
                            100 * m.get(f"auc30_focus@{mk}", np.nan)]
                rows_md.append(row)
                extra[f"All · {cfg}"] = dict(zip(head[1:], row[1:]))
            md.append(md_table(head, rows_md))
            ci_md, ci_extra = sec6_ci_table(sv, group, plan)
            md.append(f"Paired {level}% scene-bootstrap CIs (descriptive; ratios = foveated / reference, differences in "
                      "points; equal tokens = the uniform curve interpolated per scene in log tokens):\n")
            md.append(ci_md)
            ref_tag = sagg["u504"]["tag"] if "u504" in sagg else None
            save_record(make_record(
                enhancement="foveated", variant="sec6_views", tag=f"{args.tag_prefix}_{model}__sec6views",
                setting="official", weights=weights, protocol=group[0]["protocol"] if group else protocol(["unknown"]),
                extra={"focus_context": extra, "paired_ci": ci_extra}, baseline_tag=ref_tag, commit=commit,
                env={**env, **meta}, notes="per-view analysis of the Sec. 6 runs (exports + GT depth)"), results_root)
            fov_masks = [f"f{config_fraction(c):g}" for c in sagg if config_fraction(c) is not None]
            files.append(sec6_figure(sagg, group, fov_masks[0] if fov_masks else None, out_dir / f"sec6_{model}.png"))
        # ---- large-view protocol
        p = out_dir / f"scaling_{model}.json"
        if p.exists():
            sc = jload(p)
            tab, extra = scaling_tables(sc)
            md.append(f"### Large-view protocol ({NAMES.get(sc['dataset'], sc['dataset'])} full sequences, {sc['gpu']})\n")
            md.append(tab)
            ci_md, ci_extra = scaling_ci_table(sc, plan)
            md.append(f"Foveated vs each alternative, paired by scene ({level}% CIs; times from the scenes with repeated "
                      "timings, median of the repeats):\n")
            md.append(ci_md)
            save_record(make_record(
                enhancement="foveated", variant="scaling", tag=f"fov_{model}__scaling", setting="official", weights=weights,
                protocol=protocol([sc["dataset"]], sorted(sc["scenes"]), max(sc["n_views"]), ["scaling"], "first"),
                extra={"scaling": extra, "paired_ci": ci_extra}, commit=commit, env={**env, **meta},
                notes=f"spec {sc['spec']}; times = forward pass + camera decoding"), results_root)
            files.append(scaling_figure(sc, out_dir / f"scaling_{model}.png"))
        # ---- region zoom: gate 2
        p = out_dir / f"zoom_{model}.json"
        if p.exists():
            zj = jload(p)
            if zj.get("plan_sha256") and zj["plan_sha256"] != plan["sha256"]:
                warnings.append(f"{model}: the zoom ran under plan {zj['plan_sha256'][:12]}, this analysis uses "
                                f"{plan['sha256'][:12]}.")
            zg = zoom_gates(zj["rows"], plan)
            tab, extra = zoom_selector_table(zj["rows"])
            n_reg = len({(r["scene"], r["view"], tuple(r["box"])) for r in zj["rows"] if r.get("methods")})
            md.append(f"### Region zoom (ETH3D, {n_reg} distinct regions, 1/{zj['zoom']:g} of the image side)\n")
            md.append(md_table(["gate", "role", "result", "estimate [CI] → decision"],
                               [[k, g.get("role", ""), g["result"], g["value"]] for k, g in zg.items()]))
            md.append("Means per region selector (AbsRel / δ / F1 median-aligned per region; `abs_rel_scene` with the "
                      "source view's scale from the same pass; R_v and focal errors are medians):\n")
            md.append(tab)
            for k, g in zg.items():
                gates_all[f"{model} · {k}"] = g
            save_record(make_record(
                enhancement="foveated", variant="zoom_region", tag=f"fov_{model}__zoom", setting="official", weights=weights,
                protocol=protocol(["eth3d"], sorted(zj["scenes"]), 100, ["zoom"], "first"),
                extra={"zoom": extra, "gates": gates_flat(zg)}, commit=commit, env={**env, **meta},
                notes="DA3 cameras = head poses of the unposed pass; oracle regions use GT, random / texture do not"),
                results_root)
            files.append(forest_figure(forest_entries(zg), out_dir / f"gates_zoom_{model}.png",
                                       f"{model}: region-zoom gate ({level}% scene-bootstrap CIs)"))
            if (out_dir / f"zoom_{model}" / "zoom_examples.png").exists():
                files.append(out_dir / f"zoom_{model}" / "zoom_examples.png")

    head = ["## Go / no-go gates (report §4.3; pre-registered plan, three-way decisions)\n",
            md_table(["gate", "role", "result", "value", "criterion"],
                     [[k, g.get("role", ""), g["result"], g["value"], g.get("criterion", "")] for k, g in gates_all.items()])]
    if warnings:
        head = ["**Warnings:**\n" + "\n".join(f"- {w}" for w in warnings) + "\n"] + head
    md[3:3] = head
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    jdump({"plan_sha256": plan["sha256"], "gates": gates_all, "warnings": warnings}, out_dir / "gates.json")
    for w in warnings:
        print("WARNING:", w)
    print_gates({"gates": gates_all})
    print("figures:", [str(f) for f in files if f])


def uniform_configs(sagg) -> list[str]:
    return sorted((c for c in sagg if c.startswith("u") and c[1:].isdigit()), key=lambda c: -int(c[1:]))


def probe_figure(agg, datasets, path):
    import matplotlib.pyplot as plt

    datas = [d for d in datasets if any(k[0] == d for k in agg)] + ["all"]
    fig, axes = plt.subplots(2, len(datas), figsize=(4.4 * len(datas), 7), squeeze=False)
    for j, d in enumerate(datas):
        uni = sorted([(m["tokens"], m, c) for (dd, c), m in agg.items() if dd == d and c.startswith("u") and c[1:].isdigit()],
                     key=lambda t: t[0])
        fov = sorted([(m["tokens"], m, c) for (dd, c), m in agg.items()
                      if dd == d and c.startswith(("fov", "twopass", "diag_")) and "_random" not in c],
                     key=lambda t: t[2])
        for ax, key, lab in ((axes[0][j], "AbsRel focus", "AbsRel of focus views ↓"),
                             (axes[1][j], "AUC@30 context", "AUC@30 (%) of context-view pairs ↑")):
            ax.plot([t for t, _, _ in uni], [m[key] for _, m, _ in uni], "o-", color="k", label="uniform resolution")
            for t, m, c in uni:
                ax.annotate(c, (t, m[key]), fontsize=7, xytext=(3, 3), textcoords="offset points")
            for i, (t, m, c) in enumerate(fov):
                ax.plot(t, m[key], "*", ms=12, color=f"C{i}", label=c)
            readable_log_axis(ax)
            ax.set_xlabel("patch tokens per view subset")
            ax.set_title(f"{NAMES.get(d, d)}: {lab}", fontsize=9)
            ax.grid(alpha=0.3)
        axes[0][j].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def sec6_figure(sagg, group, mk, path):
    import matplotlib.pyplot as plt

    sec6 = {r["tag"]: r["sec6"] for r in group}

    def tok(c):
        return sagg[c]["data"]["all"]["tokens"]

    panels = [("AUC@3 in % (mean over datasets) ↑", lambda c: _sec6_mean(sec6.get(sagg[c]["tag"], {}), "pose", "auc03")),
              ("F1 w/o poses in % (mean) ↑", lambda c: _sec6_mean(sec6.get(sagg[c]["tag"], {}), "recon_unposed", "fscore"))]
    if mk:
        panels.append((f"AbsRel of the focus views ({mk}) ↓", lambda c: sagg[c]["data"]["all"].get(f"abs_rel_focus@{mk}", np.nan)))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.8), squeeze=False)
    uni = sorted(uniform_configs(sagg), key=tok)
    for ax, (lab, fn) in zip(axes[0], panels):
        ax.plot([tok(c) for c in uni], [fn(c) for c in uni], "o-", color="k", label="uniform resolution")
        for c in uni:
            ax.annotate(c, (tok(c), fn(c)), fontsize=7, xytext=(3, 3), textcoords="offset points")
        for c in sorted(sagg):
            if c not in uni:
                ax.plot(tok(c), fn(c), "*", ms=12, label=c)
        readable_log_axis(ax)
        ax.set_xlabel("patch tokens per scene (mean)")
        ax.set_title(lab, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _sec6_mean(s, mode, key):
    v = [s[k].get(key) for k in s if k.endswith("_" + mode) and isinstance(s[k], dict) and s[k].get(key) is not None]
    return 100 * float(np.mean(v)) if v else np.nan


def scaling_tables(sc):
    groups = defaultdict(list)
    for r in sc["rows"]:
        groups[(r["n_views"], r["kind"])].append(r)
    head = ["views", "configuration", "kind", "views used", "tokens", "GFLOPs", "time (s)", "peak mem (GB)",
            "speed-up vs all at focus res.", "AUC@3 focus", "AUC@30 focus", "AUC@30 all", "AUC@30 cross*",
            "AbsRel focus", "AbsRel context", "AbsRel scene-scale*", "AbsRel group-scale*", "scale ctx/focus*"]
    out_rows, extra = [], {}
    order = sorted(groups, key=lambda k: (k[0], SCALING_KINDS.index(k[1]) if k[1] in SCALING_KINDS else 99))
    for n, kind in order:
        rs = groups[(n, kind)]
        ok = [r for r in rs if not r.get("oom")]
        ref = [r for r in groups.get((n, "all_focus_res"), []) if not r.get("oom")]
        if not ok:
            out_rows.append([n, rs[0]["config"], kind, rs[0]["views_used"]] + ["OOM"] + [float("nan")] * (len(head) - 5))
            extra[f"N={n} · {kind}"] = {"OOM": True}
            continue
        t = nanmean([r["time_s"] for r in ok])
        speed = nanmean([r["time_s"] for r in ref]) / t if ref else float("nan")
        vals = [nanmean([r["tokens"] for r in ok]), nanmean([r["gflops"] for r in ok]), t,
                nanmax([r["peak_mem_gb"] for r in ok]), speed,
                100 * nanmean([r["auc"].get("focus", {}).get("auc03", np.nan) for r in ok]),
                100 * nanmean([r["auc"].get("focus", {}).get("auc30", np.nan) for r in ok]),
                100 * nanmean([r["auc"]["all"]["auc30"] for r in ok]),
                100 * nanmean([r["auc"].get("cross", {}).get("auc30", np.nan) for r in ok]),
                nanmean([r["abs_rel_focus"] for r in ok]), nanmean([r["abs_rel_context"] for r in ok]),
                nanmean([r.get("abs_rel_scene_all", np.nan) for r in ok]),
                nanmean([r.get("abs_rel_group_all", np.nan) for r in ok]),
                nanmean([r.get("scale_ratio_ctx_focus", np.nan) for r in ok])]
        out_rows.append([n, ok[0]["config"], kind, ok[0]["views_used"]] + vals)
        extra[f"N={n} · {kind} ({ok[0]['config']})"] = dict(zip(head[4:], vals))
    return md_table(head, out_rows), extra


def scaling_figure(sc, path):
    import matplotlib.pyplot as plt

    present = {r["kind"] for r in sc["rows"]}
    kinds = [k for k in SCALING_KINDS if k in present] + sorted(present - set(SCALING_KINDS))
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for i, kind in enumerate(kinds):
        pts = defaultdict(list)
        for r in sc["rows"]:
            if r["kind"] == kind and not r.get("oom"):
                pts[r["n_views"]].append(r)
        ns = sorted(pts)
        if not ns:
            continue
        st = dict(marker="o", color=f"C{i}", label=kind)
        axes[0].plot(ns, [nanmean([r["time_s"] for r in pts[n]]) for n in ns], **st)
        axes[1].plot(ns, [nanmax([r["peak_mem_gb"] for r in pts[n]]) for n in ns], **st)
        axes[2].plot(ns, [100 * nanmean([r["auc"].get("focus", {}).get("auc30", np.nan) for r in pts[n]]) for n in ns], **st)
        oom = sorted({r["n_views"] for r in sc["rows"] if r["kind"] == kind and r.get("oom")})
        if oom:   # mark the first view count that did not fit
            axes[1].axvline(oom[0], color=f"C{i}", ls=":", lw=1)
    all_n = sorted({r["n_views"] for r in sc["rows"]})
    for ax, lab in zip(axes, ("forward time (s)", "peak GPU memory (GB); dotted = first OOM",
                              "AUC@30 (%) of focus-view pairs")):
        ax.set_xlabel("views in one pass")
        ax.set_title(lab, fontsize=9)
        readable_log_axis(ax, ticks=all_n)
        ax.grid(alpha=0.3)
    readable_log_axis(axes[0], "y")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ============================================================================= CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, harness=True):
        p.add_argument("--weights", default="official:da3-small")
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        if harness:
            p.add_argument("--official-src", required=True)
            p.add_argument("--bench-root", required=True)
            p.add_argument("--work-root", required=True)

    p = sub.add_parser("parity")
    common(p, harness=False)
    p.add_argument("--images", required=True)
    p.add_argument("--views", type=int, default=8)
    p.add_argument("--res", type=int, default=504)
    p.add_argument("--context-res", type=int, default=252)
    p.add_argument("--out", required=True)

    p = sub.add_parser("probe")
    common(p)
    p.add_argument("--datasets", nargs="+", default=["hiroom", "7scenes"])
    p.add_argument("--max-scenes", type=int, default=0)
    p.add_argument("--n-views", type=int, default=16)
    p.add_argument("--fraction", type=float, default=0.5)
    p.add_argument("--subsets", type=int, default=3, help="interleaved view subsets per scene")
    p.add_argument("--uniform", type=int, nargs="+", default=[504, 378, 336, 280, 252])
    p.add_argument("--foveated", nargs="*", default=["context_res=252", "context_res=280", "context_res=252,balance=1"],
                   help="the first one is the primary configuration (it must match the plan)")
    p.add_argument("--focus-seeds", type=int, nargs="*", default=[1, 2, 3],
                   help="robustness: the primary configuration with random focus views")
    p.add_argument("--ref-strategy", default="first")
    p.add_argument("--diagnostics", action="store_true",
                   help="exploratory, not pre-registered: context views at context-res information on the focus grid "
                        "(diag_up<res>) and the two-pass control (all views at the context res + focus views alone)")
    p.add_argument("--plan", default=str(PLAN))
    p.add_argument("--out", required=True)

    p = sub.add_parser("scaling")
    common(p)
    p.add_argument("--dataset", default="7scenes")
    p.add_argument("--scenes", nargs="+", default=["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"])
    p.add_argument("--n-views", type=int, nargs="+", default=[100, 250, 500, 1000])
    p.add_argument("--fraction", type=float, default=0.1)
    p.add_argument("--focus-res", type=int, default=504)
    p.add_argument("--context-res", type=int, default=252)
    p.add_argument("--balance", action="store_true", help="also run the attention-mass balanced variant")
    p.add_argument("--depth-every", type=int, default=10, help="depth metrics on focus views + every k-th view")
    p.add_argument("--timing-repeats", type=int, default=3, help="passes per configuration on the timed scenes (median)")
    p.add_argument("--timing-scenes", type=int, default=2, help="the first k scenes get repeated timings")
    p.add_argument("--two-pass", action="store_true",
                   help="exploratory: also the focus views alone at the focus res, and the two-pass control "
                        "(all views at the context res + focus views alone, aligned on the focus cameras)")
    p.add_argument("--out", required=True)

    p = sub.add_parser("sec6views")
    common(p)
    p.add_argument("--tag", required=True)
    p.add_argument("--config", default="")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--fractions", type=float, nargs="*", default=[0.25, 0.1])
    p.add_argument("--out", required=True)

    p = sub.add_parser("zoom")
    common(p)
    p.add_argument("--eth3d-zip", default="", help="DA3-BENCH eth3d.zip: extract only what is needed")
    p.add_argument("--scenes", nargs="+", default=["courtyard", "electro", "kicker", "pipes", "relief", "delivery_area",
                                                     "facade", "office", "playground", "relief_2", "terrains"])
    p.add_argument("--selectors", nargs="+", default=["oracle", "random", "texture"],
                   help="region choice: oracle (GT detail, primary), random, texture (image only)")
    p.add_argument("--views-per-scene", type=int, default=2)
    p.add_argument("--regions-per-view", type=int, default=1)
    p.add_argument("--zoom", type=float, default=4.0, help="region side = image side / zoom")
    p.add_argument("--res", type=int, default=504)
    p.add_argument("--res-hi", type=int, default=896)
    p.add_argument("--max-frames", type=int, default=100)
    p.add_argument("--n-examples", type=int, default=4)
    p.add_argument("--fig-dir", default="")
    p.add_argument("--plan", default=str(PLAN))
    p.add_argument("--out", required=True)

    p = sub.add_parser("analyse")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--results-root", required=True)
    p.add_argument("--models", nargs="+", default=["da3-small", "da3-base"])
    p.add_argument("--tag-prefix", default="fov", help="Sec. 6 run tags are <prefix>_<model>__<config>")
    p.add_argument("--plan", default=str(PLAN))

    args = ap.parse_args()
    {"parity": cmd_parity, "probe": cmd_probe, "scaling": cmd_scaling, "sec6views": cmd_sec6views,
     "zoom": cmd_zoom, "analyse": cmd_analyse}[args.cmd](args)


if __name__ == "__main__":
    main()
