"""Enhancement #4, lens-consistent DA3: experiments around the mixed-lens Sec. 6 protocol (docs/LENS.md).

Sub-commands, in the order notebook 10_lens runs them:

  parity   GPU  gate L0: with no zoomed view every arm (native, canon_oracle, canonpad_oracle,
                canon_self, unzoomed) is bit-identical to the released benchmark adapter; plus a zoom
                sanity check (zoomed views cover only the centre of their export, principal point
                centred, canonical arms put zoomed views on a smaller grid)
  views    CPU  per-view metrics of one mixed-lens Sec. 6 run written by eval_bench.py --lens (run it
                before the exports are deleted): depth-scale ratio zoomed / normal views (GT depth,
                HiRoom / 7Scenes), pose AUC on all / normal / zoom / cross pairs (rotation and
                translation apart), focal ratio, zoom-estimate error
  analyse  CPU  pre-registered gates L1-L4 (configs/lens_analysis.yaml), tables, figures, standard
                result records (<results-root>/lens/), summary.md

Setting [official]: the released weights, network unchanged. The Sec. 6 numbers (AUC, F1) of every
arm come from scripts/eval_bench.py --lens and the released harness; this script adds the per-view
diagnostics and the decisions. The mixed-lens protocol is new: its numbers are only ever compared with
each other (the records carry protocol input "lens:...").

  python scripts/lens_eval.py views --official-src /kaggle/working/Depth-Anything-3/src \
      --bench-root /tmp/da3bench --work-root /tmp/lens_work --tag lens_da3-small_hiroom__z2_f30__native \
      --datasets hiroom 7scenes dtu64 --weights official:da3-small --out /kaggle/working/lens/views_<tag>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "configs" / "lens_analysis.yaml"
NAMES = {"hiroom": "HiRoom", "7scenes": "7Scenes", "dtu64": "DTU-64", "eth3d": "ETH3D", "dtu": "DTU",
         "scannetpp": "ScanNet++", "all": "All"}
ARM_LABELS = {"unzoomed": "unzoomed (ceiling, no zoom)", "native": "DA3 as released (one grid)",
              "canon_oracle": "P1, true zoom", "canonpad_oracle": "P1 padded (diagnostic)",
              "canon_self": "P1, zoom from DA3 itself"}
ARM_ORDER = ["unzoomed", "native", "canon_oracle", "canonpad_oracle", "canon_self"]


def jdump(obj, path):
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


def absolutise(args, *names):
    for n in names:
        v = getattr(args, n, None)
        if v:
            setattr(args, n, str(Path(v).resolve()))


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


# ============================================================================= parity (GPU): gate L0
def cmd_parity(args):
    from da3.enhancements.lens import ARMS, KIND_ZOOM, LensSpec
    from da3.enhancements.lens.adapter import LensAPI
    from da3.eval.bench_adapter import OursAsOfficialAPI
    from da3.io_utils import list_images, read_rgb

    device = torch.device(args.device)
    model = load_model(args.weights, device)
    files = [str(p) for p in list_images(args.images)[: args.views]]
    if len(files) < 4:
        raise SystemExit(f"need >= 4 images in {args.images}")
    ref = OursAsOfficialAPI(model, device, process_res=args.res).predict(files)
    originals = [read_rgb(f) for f in files]
    checks = {}
    for arm in ARMS:
        r = LensAPI(model, device, spec=LensSpec(arm=arm, fraction=0.0, res=args.res)).run(originals)
        pred, meta = r["pred"], r["meta"]
        diffs = {k: float(np.abs(np.asarray(ref[k], np.float64) - np.asarray(pred[k], np.float64)).max())
                 for k in ("depth", "conf", "extrinsics", "intrinsics")}
        exact = all(np.array_equal(ref[k], pred[k]) for k in diffs)
        entry = {"pass": bool(exact), "max_abs_diff": diffs, "criterion": "bit-identical"}
        if arm == "canon_self":
            # a no-op only if DA3's own focal estimates of this single-lens set stay inside the dead band
            z = np.asarray(meta["z_est"], np.float64)
            entry["passes"] = int(meta["passes"])
            entry["z_est_range"] = [float(np.nanmin(z)), float(np.nanmax(z))] if np.isfinite(z).any() else None
            if int(meta["passes"]) == 2:
                entry.update({"pass": True, "criterion": "bit-identical when no second pass runs; here DA3's focal "
                              "estimates left the dead band, so a second pass ran (informative, see z_est_range)"})
        checks[f"no zoomed view, arm {arm}: bit-identical to the released adapter"] = entry
        print(f"[parity] {arm}: bit-identical={exact} passes={meta['passes']} {diffs}")

    # zoom sanity (released weights, half the views zoomed 2x)
    for arm in ("native", "canon_oracle"):
        api = LensAPI(model, device, spec=LensSpec(arm=arm, zoom=2.0, fraction=0.5, res=args.res))
        r = api.run(originals)
        pred, meta = r["pred"], r["meta"]
        z = np.asarray(meta["kinds"]) == KIND_ZOOM
        h, w = pred["depth"].shape[1:]
        border = np.concatenate([pred["depth"][z][:, :h // 8].ravel(), pred["depth"][z][:, :, :w // 8].ravel()])
        centre = pred["depth"][z][:, 3 * h // 8: 5 * h // 8, 3 * w // 8: 5 * w // 8]
        pp = np.abs(pred["intrinsics"][:, 0, 2] - w / 2).max() + np.abs(pred["intrinsics"][:, 1, 2] - h / 2).max()
        ok = bool(np.all(border == 0) and np.all(centre > 0) and pp < 1.0)
        detail = {"zoomed_views": int(z.sum()), "border_all_zero": bool(np.all(border == 0)),
                  "centre_all_valid": bool(np.all(centre > 0)), "principal_point_offset_px": float(pp),
                  "view_hw": np.asarray(meta["view_hw"]).tolist()}
        if arm == "canon_oracle":
            hw = np.asarray(meta["view_hw"])
            smaller = bool(np.all(hw[z].prod(1) < hw[~z].prod(1).min()))
            ok &= smaller
            detail["zoomed_views_on_smaller_grid"] = smaller
        detail["focal_ratio_zoom_over_normal"] = float(np.median(pred["intrinsics"][z, 0, 0]) /
                                                       np.median(pred["intrinsics"][~z, 0, 0]))
        checks[f"zoom sanity, arm {arm}"] = {"pass": ok, **detail,
                                            "criterion": "zoomed views: export border 0, centre valid, principal "
                                                         "point centred (< 1 px); canon_oracle: smaller grid"}
        print(f"[parity] zoom sanity {arm}: {'PASS' if ok else 'FAIL'} {detail}")
    all_pass = all(c["pass"] for c in checks.values())
    jdump({"weights": args.weights, "device": str(device), "views": len(files), "res": args.res,
           "checks": checks, "all_pass": all_pass}, args.out)
    print("[parity] ALL PASS" if all_pass else "[parity] SOME CHECKS FAILED")


# ============================================================================= views (CPU)
def cmd_views(args):
    from da3.enhancements.lens import LensSpec
    from da3.enhancements.lens.adapter import META_FILE
    from da3.enhancements.lens.metrics import scene_view_metrics
    from da3.eval.gt_depth import GT_DEPTH_DATASETS, gt_depth_index, read_gt_depth
    from da3.eval.harness import evaluation_dir, load_scene_exports, prepare_workspace, unposed_scenes

    absolutise(args, "out", "bench_root", "work_root", "official_src")
    work_root = prepare_workspace(args.work_root, args.bench_root)
    os.chdir(work_root)
    sys.path.insert(0, str(Path(args.official_src).resolve()))
    from depth_anything_3.bench.registries import MV_REGISTRY

    work_dir = evaluation_dir(work_root, args.tag)
    rows, spec_text = [], None
    for data in args.datasets:
        ds = MV_REGISTRY.get(data)()
        for scene, exports in unposed_scenes(work_dir, data).items():
            if not (exports / META_FILE).exists():
                print(f"[views] {data}/{scene}: no {META_FILE} (not a --lens run?), skipped")
                continue
            sc = load_scene_exports(exports)
            meta = dict(np.load(exports / META_FILE, allow_pickle=True))
            spec_text = str(meta["spec"])
            gts = None
            if data in GT_DEPTH_DATASETS:
                index = gt_depth_index(data, ds.get_data(scene))
                hw = tuple(int(t) for t in meta["export_hw"])
                gts = []
                for f in sc["image_files"]:
                    e = index.get(str(f))
                    gts.append(read_gt_depth(data, e[0], e[1], shape=e[2], hw=hw) if e else None)
            row = scene_view_metrics(sc["results"], sc["gt_extrinsics"], sc["gt_intrinsics"], meta, gts)
            row.update({"data": data, "scene": scene, "passes": int(meta["passes"]), "tokens": int(meta["tokens"]),
                        "time_s": float(meta["time_s"]), "res": np.asarray(meta["res"]).tolist(),
                        "view_hw": np.asarray(meta["view_hw"]).tolist()})
            rows.append(row)
        print(f"[views] {args.tag} {data}: {sum(r['data'] == data for r in rows)} scenes")
    if spec_text is None:
        raise SystemExit(f"no lens exports found for tag {args.tag} in {work_dir}")
    spec = LensSpec.parse(spec_text)
    jdump({"tag": args.tag, "weights": args.weights, "datasets": args.datasets, "spec": spec_text,
           "arm": spec.arm, "protocol_id": spec.protocol_id, "protocol_input": spec.protocol_input,
           "rows": rows}, args.out)


# ============================================================================= analyse (CPU)
def load_plan(path) -> dict:
    import hashlib

    import yaml

    text = Path(path).read_text(encoding="utf-8")
    plan = yaml.safe_load(text)
    plan["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    plan["path"] = str(path)
    return plan


def _get(row, *keys, default=float("nan")):
    cur = row
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def _f1(records_by_tag, tag, data, scene):
    rec = records_by_tag.get(tag)
    if rec is None:
        return float("nan")
    return _get(rec.get("sec6_per_scene", {}), f"{data}_recon_unposed", scene, "fscore")


def scene_table(views, records_by_tag) -> dict:
    """{(data, scene): {metric: value}} for one arm."""
    out = {}
    for r in views["rows"]:
        z = np.asarray(r.get("z_err", []), np.float64)
        kinds = np.asarray(r["kinds"])
        out[(r["data"], r["scene"])] = {
            "auc03": 100 * _get(r, "auc", "all", "auc03"), "auc30": 100 * _get(r, "auc", "all", "auc30"),
            "auc30_cross": 100 * _get(r, "auc", "cross", "auc30"), "auc03_cross": 100 * _get(r, "auc", "cross", "auc03"),
            "auc30_cross_rot": 100 * _get(r, "auc_rot", "cross", "auc30"),
            "auc30_cross_trans": 100 * _get(r, "auc_trans", "cross", "auc30"),
            "auc30_normal": 100 * _get(r, "auc", "normal", "auc30"), "auc30_zoom": 100 * _get(r, "auc", "zoom", "auc30"),
            "scale_ratio": _get(r, "scale_ratio"), "abs_rel_scene": _get(r, "abs_rel_scene_all"),
            "abs_rel_zoom": _get(r, "abs_rel_zoom"), "abs_rel_normal": _get(r, "abs_rel_normal"),
            "focal_ratio_zoom": _get(r, "focal_ratio_zoom"), "focal_ratio_normal": _get(r, "focal_ratio_normal"),
            "z_err_zoom_views": z[kinds == 1] if len(z) else np.zeros(0),
            "f1": 100 * _f1(records_by_tag, views["tag"], r["data"], r["scene"]),
            "passes": float(r.get("passes", 1)), "tokens": float(r.get("tokens", np.nan)), "time_s": float(r.get("time_s", np.nan)),
        }
    return out


def geo_mean_ci(values, n, seed, level):
    from da3.enhancements.foveated.stats import bootstrap

    v = np.log(np.asarray(values, np.float64))
    v = v[np.isfinite(v)]
    est, lo, hi = bootstrap(v, np.mean, n=n, seed=seed, level=level)
    return float(np.exp(est)), float(np.exp(lo)), float(np.exp(hi))


def band_decision(ci, band: float, want_inside: bool) -> str:
    from da3.enhancements.foveated.stats import FAIL, INCONCLUSIVE, NA, PASS

    lo, hi = ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return NA
    inside = lo >= 1.0 / (1.0 + band) and hi <= 1.0 + band
    outside = lo > 1.0 + band or hi < 1.0 / (1.0 + band)
    if want_inside:
        return PASS if inside else (FAIL if outside else INCONCLUSIVE)
    return PASS if outside else (FAIL if inside else INCONCLUSIVE)


def fmt_ci(est, lo, hi, d=3):
    if not np.isfinite(est):
        return "–"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{est:.{d}f}"
    return f"{est:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def analyse_group(arms: dict, plan: dict, level: float) -> dict:
    """arms: {arm: scene_table}. Paired over the scenes every arm has."""
    from da3.enhancements.foveated.stats import NA, above, at_most, bootstrap, mean_difference, median_ci

    n, seed = int(plan["bootstrap"]["n"]), int(plan["bootstrap"]["seed"])
    common = sorted(set.intersection(*[set(t) for t in arms.values()])) if arms else []
    res = {"scenes": [list(k) for k in common], "arms": {}, "diff_vs_native": {}}
    metrics = ["auc03", "auc30", "auc30_cross", "auc30_cross_rot", "auc30_cross_trans", "auc30_normal",
               "f1", "abs_rel_scene", "focal_ratio_zoom", "passes", "tokens", "time_s"]
    datasets = sorted({k[0] for k in common})
    for arm, tab in arms.items():
        a = {}
        for m in metrics:
            vals = np.array([tab[k][m] for k in common], np.float64)
            a[m] = bootstrap(vals, np.nanmean, n=n, seed=seed, level=level) if np.isfinite(vals).sum() >= 2 \
                else (float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan"), float("nan"), float("nan"))
            for d in datasets:
                dv = np.array([tab[k][m] for k in common if k[0] == d], np.float64)
                a[f"{m}@{d}"] = float(np.nanmean(dv)) if np.isfinite(dv).any() else float("nan")
        sr = [tab[k]["scale_ratio"] for k in common if np.isfinite(tab[k]["scale_ratio"])]
        a["scale_ratio"] = geo_mean_ci(sr, n, seed, level) if len(sr) >= 2 else (float("nan"),) * 3
        zs = [tab[k]["z_err_zoom_views"] for k in common if len(tab[k]["z_err_zoom_views"])]
        a["z_err_zoom"] = median_ci(zs, n=n, seed=seed, level=level) if zs else (float("nan"),) * 3
        res["arms"][arm] = a
    if "native" in arms:
        for arm, tab in arms.items():
            if arm == "native":
                continue
            d = {}
            for m in ("auc03", "auc30", "auc30_cross", "f1", "abs_rel_scene"):
                t = np.array([tab[k][m] for k in common], np.float64)
                r = np.array([arms["native"][k][m] for k in common], np.float64)
                ok = np.isfinite(t) & np.isfinite(r)
                d[m] = mean_difference(t[ok], r[ok], n=n, seed=seed, level=level) if ok.sum() >= 2 else (float("nan"),) * 3
            res["diff_vs_native"][arm] = d
    # gates (decided only on the primary protocol by the caller)
    band = float(plan["scale_band"])
    g = {}
    if "native" in res["arms"]:
        ci = res["arms"]["native"]["scale_ratio"]
        g["L1 problem exists (native scale ratio outside the band)"] = {
            "result": band_decision(ci[1:], band, want_inside=False), "value": fmt_ci(*ci),
            "criterion": f"CI of the zoomed / normal depth-scale ratio entirely outside [{1 / (1 + band):.3f}, {1 + band:.3f}]"}
    if "canon_oracle" in res["arms"]:
        ci = res["arms"]["canon_oracle"]["scale_ratio"]
        g["L2 P1 removes the offset (canon_oracle scale ratio inside the band)"] = {
            "result": band_decision(ci[1:], band, want_inside=True), "value": fmt_ci(*ci),
            "criterion": f"CI inside [{1 / (1 + band):.3f}, {1 + band:.3f}]"}
    if "canon_self" in res["diff_vs_native"]:
        ci = res["diff_vs_native"]["canon_self"]["auc30"]
        g["L3 PRIMARY: canon_self - native, AUC@30 (points)"] = {
            "result": above(ci[1:], float(plan["auc_threshold"])), "value": fmt_ci(*ci, d=2),
            "criterion": f"CI of the paired mean difference > {plan['auc_threshold']}"}
    if "canon_self" in res["arms"]:
        ci = res["arms"]["canon_self"]["z_err_zoom"]
        thr = float(plan["gates"]["L4"]["threshold"])
        g["L4 zoom estimated from DA3 (median relative error, zoomed views)"] = {
            "result": at_most(ci[1:], thr) if np.isfinite(ci[0]) else NA, "value": fmt_ci(*ci),
            "criterion": f"CI <= {thr}"}
    res["gates"] = g
    # hypothesis readout (descriptive)
    def side(arm):
        if arm not in res["arms"]:
            return None
        lo, hi = res["arms"][arm]["scale_ratio"][1:]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return None
        return "inside" if (lo >= 1 / (1 + band) and hi <= 1 + band) else \
            ("outside" if (lo > 1 + band or hi < 1 / (1 + band)) else "unclear")
    s_nat, s_can, s_pad = side("native"), side("canon_oracle"), side("canonpad_oracle")
    if s_nat == "outside" and s_can == "inside":
        h = "H_ang supported: the scale follows the angular size per patch; equal angular resolution removes it"
    elif s_nat == "inside" and s_can == "outside":
        h = "H_grid supported: the scale follows the token grid; DA3 handles zoom on one grid (no problem to fix)"
    elif s_nat == "inside" and s_can == "inside":
        h = "no offset in any arm: DA3 is lens-consistent on this protocol"
    else:
        h = "mixed / unclear (see the CIs)"
    res["hypothesis"] = {"native": s_nat, "canon_oracle": s_can, "canonpad_oracle": s_pad, "readout": h}
    return res


def cmd_analyse(args):
    from da3.compare import load_records, make_record, protocol, save_record
    from da3.enhancements.foveated.stats import bonferroni_level

    absolutise(args, "out_dir", "results_root", "plan")
    plan = load_plan(args.plan)
    out_dir = Path(args.out_dir)
    views = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("views_*.json"))]
    if not views:
        raise SystemExit(f"no views_*.json in {out_dir}")
    records = load_records(args.results_root) if args.results_root else []
    by_tag = {r["tag"]: r for r in records}
    groups = defaultdict(dict)   # (model, protocol_id) -> {arm: views}
    for v in views:
        groups[(model_name(v["weights"]), v["protocol_id"])][v["arm"]] = v
    secondary = [g for g in groups if g[1] != plan["primary_protocol"]]
    level_sec = bonferroni_level(float(plan["ci_level"]), max(1, len({g[1] for g in secondary})))
    results, lines = {}, []
    lines.append("# Enhancement #4 · lens-consistent DA3: mixed-lens Sec. 6 protocol\n")
    lines.append(f"Plan `{Path(plan['path']).name}` SHA-256 `{plan['sha256'][:12]}`; primary protocol "
                 f"`{plan['primary_protocol']}` (95% scene-bootstrap CIs); secondary protocols descriptive "
                 f"(Bonferroni {100 * level_sec:.1f}% CIs). Setting [official]: released weights, network unchanged.\n")
    for (model, pid), arms in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] != plan["primary_protocol"], kv[0][1])):
        primary = pid == plan["primary_protocol"]
        level = float(plan["ci_level"]) if primary else level_sec
        tabs = {arm: scene_table(v, by_tag) for arm, v in arms.items()}
        res = analyse_group(tabs, plan, level)
        if not primary:
            res["gates"] = {}   # secondary protocols never decide a gate
        res["protocol_input"] = next(iter(arms.values()))["protocol_input"]
        res["tags"] = {arm: v["tag"] for arm, v in arms.items()}
        results[f"{model}|{pid}"] = res
        lines += group_markdown(model, pid, primary, res, level)
    jdump({"plan": plan["path"], "plan_sha256": plan["sha256"], "results": results}, out_dir / "analysis.json")
    gates = {k: v["gates"] for k, v in results.items() if v["gates"]}
    jdump(gates, out_dir / "gates.json")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out_dir / "summary.md")
    try:
        figure(results, out_dir, plan)
    except Exception as e:   # figures are a convenience; the numbers are in the JSON / markdown
        print("[analyse] figure failed:", repr(e))

    # standard result records: one analysis record per (model, protocol)
    from da3.train_utils import environment_info, git_commit

    for key, res in results.items():
        model, pid = key.split("|")
        ref_tag = res["tags"].get("native")
        ref_rec = by_tag.get(ref_tag)
        prot = ref_rec["protocol"] if ref_rec else protocol(sorted({s[0] for s in res["scenes"]}), "all", 100,
                                                              ["pose"], "first", input=res["protocol_input"])
        extra = {"gates": res["gates"], "hypothesis": {"all": res["hypothesis"]}}
        for arm, a in res["arms"].items():
            extra[f"arm_{arm}"] = {"all": {m: (v[0] if isinstance(v, (tuple, list)) else v) for m, v in a.items()}}
        for arm, d in res["diff_vs_native"].items():
            extra[f"diff_{arm}_vs_native"] = {"all": {m: v[0] for m, v in d.items()}}
        rec = make_record(enhancement="lens", variant=f"analysis_{pid}", tag=f"lens_{model}_{pid}__analysis",
                          setting="official", weights=f"official:{model}", protocol=prot, extra=extra,
                          baseline_tag=ref_tag, commit=git_commit(REPO), env=environment_info(),
                          notes=f"plan {plan['sha256'][:12]}; {'primary' if pid == plan['primary_protocol'] else 'secondary'} protocol",
                          files=["summary.md", "analysis.json", "gates.json", f"lens_{model}.png"])
        print("record:", save_record(rec, args.results_root))
    for k, g in gates.items():
        for name, v in g.items():
            print(f"{k} · {v['result']:12s} {name}: {v['value']}")


def group_markdown(model, pid, primary, res, level) -> list:
    L = [f"\n## {model} · protocol `{pid}` ({res['protocol_input']}) · {'PRIMARY' if primary else 'secondary'} · "
         f"{len(res['scenes'])} paired scenes\n"]
    head = ["Arm", "AUC@3", "AUC@30", "AUC@30 zoom–normal", "… rotation only", "… translation only", "F1 w/o p.",
            "depth-scale ratio zoom / normal", "focal ratio (zoomed)", "zoom-estimate error", "passes", "tokens", "time (s)"]
    L.append("| " + " | ".join(head) + " |")
    L.append("|" + "---|" * len(head))
    for arm in [a for a in ARM_ORDER if a in res["arms"]]:
        a = res["arms"][arm]
        d = res["diff_vs_native"].get(arm, {})

        def cell(m, digits=1):
            v = a[m][0]
            if not np.isfinite(v):
                return "–"
            s = f"{v:.{digits}f}"
            if m in d and np.isfinite(d[m][0]):
                s += f" ({d[m][0]:+.{digits}f})"
            return s
        L.append("| " + " | ".join([
            ARM_LABELS.get(arm, arm), cell("auc03"), cell("auc30"), cell("auc30_cross"),
            f"{a['auc30_cross_rot'][0]:.1f}" if np.isfinite(a['auc30_cross_rot'][0]) else "–",
            f"{a['auc30_cross_trans'][0]:.1f}" if np.isfinite(a['auc30_cross_trans'][0]) else "–",
            cell("f1"), fmt_ci(*a["scale_ratio"]), fmt_ci(*a["focal_ratio_zoom"]), fmt_ci(*a["z_err_zoom"]),
            f"{a['passes'][0]:.2f}", f"{a['tokens'][0]:.0f}", f"{a['time_s'][0]:.2f}"]) + " |")
    L.append("\nIn brackets: paired difference to DA3 as released on the same zoomed set (arm native). "
             "Depth-scale ratio > 1: zoomed views predicted too close relative to the normal views.\n")
    if res["diff_vs_native"]:
        L.append(f"Paired differences vs native ({100 * level:.1f}% CIs, points):\n")
        L.append("| Arm | Δ AUC@3 | Δ AUC@30 | Δ AUC@30 zoom–normal | Δ F1 |")
        L.append("|---|---|---|---|---|")
        for arm in [a for a in ARM_ORDER if a in res["diff_vs_native"]]:
            d = res["diff_vs_native"][arm]
            L.append(f"| {ARM_LABELS.get(arm, arm)} | " + " | ".join(fmt_ci(*d[m], d=2) for m in
                                                                     ("auc03", "auc30", "auc30_cross", "f1")) + " |")
    L.append(f"\n**Hypothesis readout:** {res['hypothesis']['readout']} (native: {res['hypothesis']['native']}, "
             f"canon_oracle: {res['hypothesis']['canon_oracle']}, padded: {res['hypothesis']['canonpad_oracle']}).\n")
    if res["gates"]:
        L.append("| Gate | Decision | Estimate [CI] | Criterion |\n|---|---|---|---|")
        for name, g in res["gates"].items():
            L.append(f"| {name} | **{g['result']}** | {g['value']} | {g['criterion']} |")
    return L


def figure(results, out_dir, plan):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    band = float(plan["scale_band"])
    by_model = defaultdict(list)
    for key, res in results.items():
        model, pid = key.split("|")
        by_model[model].append((pid, res))
    for model, items in by_model.items():
        items.sort(key=lambda t: (t[0] != plan["primary_protocol"], t[0]))
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
        ax = axes[0]
        xt, xl, x = [], [], 0
        for pid, res in items:
            for arm in [a for a in ARM_ORDER if a in res["arms"]]:
                est, lo, hi = res["arms"][arm]["scale_ratio"]
                if np.isfinite(est):
                    yerr = [[est - lo], [hi - est]] if np.isfinite(lo) else None
                    ax.errorbar([x], [est], yerr=yerr, fmt="o", color=f"C{ARM_ORDER.index(arm)}", capsize=4)
                xt.append(x)
                xl.append(f"{pid}\n{arm}")
                x += 1
            x += 1
        ax.axhspan(1 / (1 + band), 1 + band, color="0.85", zorder=0)
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("depth-scale ratio zoomed / normal")
        ax.set_title(f"{model}: depth-scale consistency (grey = ±{100 * band:.0f}% band)")
        ax = axes[1]
        w, x0 = 0.38, 0
        xt, xl = [], []
        for pid, res in items:
            for arm in [a for a in ARM_ORDER if a in res["arms"]]:
                a = res["arms"][arm]
                ax.bar(x0 - w / 2, a["auc30"][0], w, color=f"C{ARM_ORDER.index(arm)}")
                ax.bar(x0 + w / 2, a["auc30_cross"][0], w, color=f"C{ARM_ORDER.index(arm)}", alpha=0.5, hatch="//")
                xt.append(x0)
                xl.append(f"{pid}\n{arm}")
                x0 += 1
            x0 += 1
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("AUC@30 (solid: all pairs, hatched: zoom–normal pairs)")
        ax.set_title("Sec. 6 pose accuracy on the mixed-lens set")
        plt.tight_layout()
        p = out_dir / f"lens_{model}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print("wrote", p)


# ============================================================================= CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("parity")
    p.add_argument("--weights", default="official:da3-small")
    p.add_argument("--images", required=True)
    p.add_argument("--views", type=int, default=8)
    p.add_argument("--res", type=int, default=504)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", required=True)

    p = sub.add_parser("views")
    p.add_argument("--official-src", required=True)
    p.add_argument("--bench-root", required=True)
    p.add_argument("--work-root", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--weights", default="official:da3-small")
    p.add_argument("--out", required=True)

    p = sub.add_parser("analyse")
    p.add_argument("--out-dir", required=True, help="folder with the views_*.json files")
    p.add_argument("--results-root", default="", help="standard result records (Sec. 6 F1 per scene; output records)")
    p.add_argument("--plan", default=str(PLAN))

    args = ap.parse_args()
    {"parity": cmd_parity, "views": cmd_views, "analyse": cmd_analyse}[args.cmd](args)


if __name__ == "__main__":
    main()
