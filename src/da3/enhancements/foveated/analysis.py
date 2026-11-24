"""Pre-registered analysis of enhancement #2 (Foveated DA3): per-scene values -> three-way gates.

Plan constants: configs/foveated_analysis.yaml; reasons: docs/FOVEATED_ANALYSIS_PLAN.md. No rule
here looks at a foveated result before it is applied:
* margins come from the plan, raised only by the noise floor of a treatment-blind A/A comparison
  (the original DA3 with only the reference view changed);
* primary configurations and criteria are fixed by the plan; secondary configurations get
  simultaneous (Bonferroni) CIs; the only selection step (G3 rescue) chooses on the dev half of the
  scenes and decides on the held-out test half.

Probe rows (scripts/foveated_eval.py probe) hold, per (dataset, scene, view subset, configuration):
per-view depth errors (NaN for views not in the pass), the pairwise pose-error matrix, the focus
mask that defines "focus" / "context" for that configuration, and the views that took part.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import stats as S
from .metrics import subset_auc

DEPTH_METRICS = ("abs_rel", "delta1", "delta105", "abs_rel_scene", "abs_rel_group")
# pose-error matrix of each AUC flavour: max(RRA, RTA) (the benchmark's), rotation only, translation only
PAIR_KEYS = {"": "pair_err", "rot": "pair_rot", "trans": "pair_trans"}


def load_plan(path) -> dict:
    import yaml

    text = Path(path).read_text(encoding="utf-8")
    plan = yaml.safe_load(text)
    plan["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    plan["path"] = str(path)
    return plan


def _nanmean(x) -> float:
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def config_resolutions(cfg: str):
    """'fov50_504-252_bal' -> (504, 252); None for non-foveated names."""
    m = re.match(r"^fov\d+_(\d+)-(\d+)", cfg)
    return (int(m.group(1)), int(m.group(2))) if m else None


# ----------------------------------------------------------------------------- per-row metrics
def row_metric(row: dict, metric: str, part: str, mask) -> float:
    """`metric` of one row on a part of its views: 'focus' (mask), 'context' (~mask), 'all', or
    'cross' (pose pairs focus-context). Depth metrics average the per-view errors; pose metrics are
    AUCs over the pairs of the part."""
    mask = np.asarray(mask, bool)
    members = np.asarray(row.get("members", np.ones(len(mask), bool)), bool)
    sel = {"focus": mask, "context": ~mask, "all": np.ones_like(mask), "cross": mask}[part] & members
    if metric in DEPTH_METRICS:
        if metric not in row:   # rows written before the diagnostics existed
            return float("nan")
        v = np.asarray(row[metric], np.float64)[sel]
        return _nanmean(v)
    if metric == "scale_ratio":   # diagnostic: median per-view depth scale, context / focus views (1 = consistent)
        if "scale" not in row:
            return float("nan")
        sc = np.asarray(row["scale"], np.float64)
        f, c = sc[mask & members], sc[~mask & members]
        f, c = f[np.isfinite(f)], c[np.isfinite(c)]
        return float(np.median(c) / np.median(f)) if len(f) and len(c) else float("nan")
    if metric in ("tokens", "gflops", "time_s", "peak_mem_gb"):
        return float(row.get(metric, float("nan")))
    base, _, flavour = metric.partition("_")   # "auc30", "auc30_rot", "auc30_trans"
    thr = int(base[3:])
    key = PAIR_KEYS[flavour]
    if key not in row:
        return float("nan")
    E = np.asarray(row[key], np.float64)
    res = (subset_auc(E, mask & members, ~mask & members, thresholds=(thr,)) if part == "cross"
           else subset_auc(E, sel, thresholds=(thr,)))
    return res[f"auc{thr:02d}"] if res["pairs"] else float("nan")


def index_rows(rows) -> dict:
    return {(r["data"], r["scene"], r.get("subset", 0), r["config"]): r for r in rows if not r.get("oom")}


def scene_values(index: dict, config: str, metric: str, part: str, mask_config: str | None = None) -> dict:
    """{(data, scene): mean over the scene's view subsets} for `config`, with focus / context defined
    by the focus mask of `mask_config` (default: the configuration's own) on the same subset."""
    out = defaultdict(list)
    for (d, s, sub, c), r in index.items():
        if c != config:
            continue
        mrow = index.get((d, s, sub, mask_config or config))
        if mrow is None:
            continue
        out[(d, s)].append(row_metric(r, metric, part, mrow["focus"]))
    return {k: _nanmean(v) for k, v in out.items()}


def paired(index, test, ref, metric, part, mask_config, keys=None):
    a = scene_values(index, test, metric, part, mask_config)
    b = scene_values(index, ref, metric, part, mask_config)
    ks = sorted(set(a) & set(b) & (set(keys) if keys is not None else set(a)))
    ks = [k for k in ks if np.isfinite(a[k]) and np.isfinite(b[k]) and b[k] != 0]
    return ks, np.array([a[k] for k in ks]), np.array([b[k] for k in ks])


def _ratio(index, plan, test, ref, metric, part, mask_config, keys=None, level=None) -> dict:
    bs = plan["bootstrap"]
    ks, a, b = paired(index, test, ref, metric, part, mask_config, keys)
    est, lo, hi = S.ratio_of_means(a, b, n=bs["n"], seed=bs["seed"], level=level or plan["ci_level"])
    return {"test": test, "ref": ref, "metric": f"{metric} ({part})", "estimate": est, "ci_lo": lo, "ci_hi": hi,
            "scenes": len(ks), "wilcoxon_p": S.wilcoxon_p(a, b), "level": level or plan["ci_level"]}


# ----------------------------------------------------------------------------- margins
def noise_floor(index, plan) -> dict:
    """Treatment-blind A/A: full resolution with only the reference view changed. The noise floor of
    a metric family is the largest relative effect the A/A CI allows (max |CI bound - 1|)."""
    p = plan["probe"]
    out = {}
    for fam, (metric, part) in {"depth": ("abs_rel", "all"), "pose": ("auc30", "all")}.items():
        r = _ratio(index, plan, p["aa"], p["hi"], metric, part, p["hi"])
        lo, hi = r["ci_lo"], r["ci_hi"]
        r["floor"] = max(abs(lo - 1.0), abs(hi - 1.0)) if np.isfinite(lo) and np.isfinite(hi) else float("nan")
        out[fam] = r
    return out


def margins(plan, nf) -> dict:
    """Margin per metric family: the pre-registered one, raised to the A/A noise floor when that is
    larger (a difference the method shows under an irrelevant change is not meaningful), but never
    above cap_factor x prereg: a noisier measurement must give wider CIs, not a looser criterion.
    `nf[fam]["above_cap"]` flags a protocol that cannot resolve the pre-registered tolerance."""
    mc = plan["margin"]
    m0 = float(mc["prereg"])
    cap = m0 * float(mc.get("cap_factor", 2.0))
    out = {}
    for fam in ("depth", "pose"):
        f = nf[fam]["floor"]
        nf[fam]["above_cap"] = bool(np.isfinite(f) and f > cap)
        if mc.get("rule", "max_prereg_noise") != "max_prereg_noise" or not np.isfinite(f):
            out[fam] = m0
        else:
            out[fam] = min(max(m0, f), cap)
    return out


# ----------------------------------------------------------------------------- gates
def _p(x) -> str:
    return f"{100 * x:.1f}%"


def _crit(c: dict) -> str:
    """The decision at every margin m, in words (from the two critical margins of the criterion)."""
    if not all(np.isfinite(v) for v in c.values()):
        return "no CI"
    if "pass_from" in c:          # non-inferiority: PASS iff m >= pass_from, FAIL iff m < fail_below
        p, f = c["pass_from"], c["fail_below"]
        txt = "PASS at any margin" if p <= 0 else f"PASS if margin >= {_p(p)}"
        return txt + (f", FAIL if margin < {_p(f)}" if f > 0 else ", never FAIL")
    if "pass_below" in c:         # headroom: PASS iff m < pass_below, FAIL iff m >= fail_from
        p, f = c["pass_below"], c["fail_from"]
        txt = "never PASS" if p <= 0 else f"PASS if margin < {_p(p)}"
        return txt + (", FAIL at any margin" if f <= 0 else f", FAIL if margin >= {_p(f)}")
    p, f = c["pass_up_to"], c["fail_above"]   # improvement: PASS iff m <= pass_up_to, FAIL iff m > fail_above
    txt = "never PASS" if p < 0 else f"PASS if margin <= {_p(p)}"
    return txt + (", FAIL at any margin" if f < 0 else f", FAIL if margin > {_p(f)}")


def gate_g1(index, plan, m, keys=None, level=None) -> dict:
    """G1a headroom: AbsRel (all views) at the low resolution vs full resolution is worse beyond the
    margin. G1b context views are informative: focus-view AbsRel with the context views vs the focus
    views alone improves beyond the margin. G1 = G1a AND G1b."""
    p = plan["probe"]
    a = _ratio(index, plan, p["lo"], p["hi"], "abs_rel", "all", p["hi"], keys, level)
    a["result"] = S.above((a["ci_lo"], a["ci_hi"]), 1 + m["depth"])
    a.update(threshold=1 + m["depth"], rule="above", label=f"G1a headroom: AbsRel {p['lo']} / {p['hi']}")
    a["critical"] = {"pass_below": a["ci_lo"] - 1, "fail_from": a["ci_hi"] - 1}
    b = _ratio(index, plan, p["primary"], p["focus_only"], "abs_rel", "focus", p["primary"], keys, level)
    b["result"] = S.at_most((b["ci_lo"], b["ci_hi"]), 1 - m["depth"])
    b.update(threshold=1 - m["depth"], rule="at_most", label="G1b context informative: focus AbsRel with / without context")
    b["critical"] = {"pass_up_to": 1 - b["ci_hi"], "fail_above": 1 - b["ci_lo"]}
    pose = _ratio(index, plan, p["lo"], p["hi"], "auc30", "all", p["hi"], keys, level)          # secondary, reported
    ctx_pose = _ratio(index, plan, p["primary"], p["focus_only"], "auc30", "focus", p["primary"], keys, level)
    res = S.combine_and(a["result"], b["result"])
    value = (f"G1a {p['lo']}/{p['hi']} AbsRel ratio {S.fmt_ci(a['estimate'], a['ci_lo'], a['ci_hi'])} -> {a['result']} "
             f"({_crit(a['critical'])}); G1b with/without context AbsRel ratio "
             f"{S.fmt_ci(b['estimate'], b['ci_lo'], b['ci_hi'])} -> {b['result']} ({_crit(b['critical'])}); "
             f"secondary: AUC@30 {p['lo']}/{p['hi']} {S.fmt_ci(pose['estimate'], pose['ci_lo'], pose['ci_hi'])}, "
             f"focus-pair AUC@30 with/without context {S.fmt_ci(ctx_pose['estimate'], ctx_pose['ci_lo'], ctx_pose['ci_hi'])}; "
             f"{a['scenes']} scenes")
    return {"result": res, "role": "primary", "value": value,
            "criterion": f"G1a: AbsRel ratio {p['lo']}/{p['hi']} > 1 + m (headroom); G1b: focus AbsRel with/without "
                         f"context views <= 1 - m (context informative); m = {100 * m['depth']:.1f}%",
            "margin": m["depth"], "G1a": a, "G1b": b, "secondary": {"auc30_lo_hi": pose, "auc30_context": ctx_pose}}


def gate_g3(index, plan, cfg, m, keys=None, level=None, role="primary") -> dict:
    """C1 focus-view AbsRel non-inferior to all views at the focus resolution; C2 context-pair AUC@30
    non-inferior to all views at the context resolution. G3 = C1 AND C2."""
    res = config_resolutions(cfg)
    if res is None:
        return {"result": S.NA, "role": role, "value": f"unknown configuration {cfg}", "criterion": ""}
    hi_cfg, lo_cfg = f"u{res[0]}", f"u{res[1]}"
    c1 = _ratio(index, plan, cfg, hi_cfg, "abs_rel", "focus", cfg, keys, level)
    c1["result"] = S.at_most((c1["ci_lo"], c1["ci_hi"]), 1 + m["depth"])
    c1.update(threshold=1 + m["depth"], rule="at_most", label=f"C1 focus AbsRel {cfg} / {hi_cfg}")
    c1["critical"] = S.critical_ratio_at_most((c1["ci_lo"], c1["ci_hi"]))
    c2 = _ratio(index, plan, cfg, lo_cfg, "auc30", "context", cfg, keys, level)
    c2["result"] = S.at_least((c2["ci_lo"], c2["ci_hi"]), 1 - m["pose"])
    c2.update(threshold=1 - m["pose"], rule="at_least", label=f"C2 context AUC@30 {cfg} / {lo_cfg}")
    c2["critical"] = S.critical_ratio_at_least((c2["ci_lo"], c2["ci_hi"]))
    out = S.combine_and(c1["result"], c2["result"])
    if c1["scenes"] == 0 or c2["scenes"] == 0:
        out = S.NA
    lvl = int(round(100 * (level or plan["ci_level"])))
    value = (f"C1 focus AbsRel {cfg}/{hi_cfg} {S.fmt_ci(c1['estimate'], c1['ci_lo'], c1['ci_hi'])} -> {c1['result']} "
             f"({_crit(c1['critical'])}); C2 context AUC@30 {cfg}/{lo_cfg} "
             f"{S.fmt_ci(c2['estimate'], c2['ci_lo'], c2['ci_hi'])} -> {c2['result']} ({_crit(c2['critical'])}); "
             f"{lvl}% CIs, {c1['scenes']} scenes")
    return {"result": out, "role": role, "value": value,
            "criterion": f"C1 ratio <= 1 + {100 * m['depth']:.1f}% AND C2 ratio >= 1 - {100 * m['pose']:.1f}%",
            "margin_depth": m["depth"], "margin_pose": m["pose"], "C1": c1, "C2": c2}


def _mean_tokens(index, cfg) -> float:
    return _nanmean([r.get("tokens", np.nan) for (d, s, sub, c), r in index.items() if c == cfg])


def probe_gates(rows, plan) -> dict:
    """All probe decisions: noise floor and margins, G1, G3 (primary), G3 secondary (mitigations,
    Bonferroni CIs), G3 robustness (random focus views), and the pre-declared G3 rescue."""
    p = plan["probe"]
    level = plan["ci_level"]
    index = index_rows(rows)
    nf = noise_floor(index, plan)
    m = margins(plan, nf)
    gates = {"G1 uniform-resolution curve": gate_g1(index, plan, m, level=level),
             "G3 cross-scale probe": gate_g3(index, plan, p["primary"], m, level=level)}
    present = {c for (_, _, _, c) in index}
    mit = [c for c in p.get("mitigations", []) if c in present]
    lvl2 = S.bonferroni_level(level, len(mit))
    for cfg in mit:
        gates[f"G3 secondary ({cfg})"] = gate_g3(index, plan, cfg, m, level=lvl2, role="secondary")
    for cfg in [c for c in p.get("focus_seeds", []) if c in present]:
        gates[f"G3 robustness ({cfg})"] = gate_g3(index, plan, cfg, m, level=level, role="robustness")
    if gates["G3 cross-scale probe"]["result"] != S.PASS and mit:
        keys = sorted({(d, s) for (d, s, _, _) in index})
        dev, test = S.stratified_split(keys, [k[0] for k in keys], plan["split"]["dev_fraction"], plan["split"]["seed"])
        cands = []
        for cfg in mit:   # selection on dev, point estimates only (pre-declared rule)
            g = gate_g3(index, plan, cfg, m, keys=dev, level=level)
            if "C1" in g and np.isfinite(g["C2"]["estimate"]) and g["C2"]["estimate"] >= 1 - m["pose"]:
                cands.append((g["C1"]["estimate"], _mean_tokens(index, cfg), cfg))
        if cands:
            sel = sorted(cands)[0][2]
            g = gate_g3(index, plan, sel, m, keys=test, level=level, role="rescue")
            g["value"] = f"selected on dev ({len(dev)} scenes): {sel}; test ({len(test)} scenes): " + g["value"]
            gates["G3 rescue (dev selection, test confirmation)"] = g
        else:
            gates["G3 rescue (dev selection, test confirmation)"] = {
                "result": S.FAIL, "role": "rescue", "criterion": "",
                "value": f"no mitigation keeps context AUC@30 within the margin on the dev split ({len(dev)} scenes)"}
    return {"plan_sha256": plan.get("sha256"), "margins": m, "noise_floor": nf, "gates": gates}


# ----------------------------------------------------------------------------- exploratory diagnostics
def _diff(index, plan, test, ref, metric, part, mask_config) -> dict:
    """Paired difference of scene means (test − ref) in AUC points, scene-bootstrap CI."""
    bs = plan["bootstrap"]
    ks, a, b = paired(index, test, ref, metric, part, mask_config)
    est, lo, hi = S.mean_difference(100 * a, 100 * b, n=bs["n"], seed=bs["seed"], level=plan["ci_level"])
    return {"test": test, "ref": ref, "metric": f"{metric} ({part})", "estimate": est, "ci_lo": lo, "ci_hi": hi,
            "scenes": len(ks), "unit": "pts"}


def probe_exploratory(rows, plan) -> dict:
    """Descriptive comparisons added AFTER the limited run of 2026-11-23 (docs/FOVEATED_ANALYSIS_PLAN.md,
    deviations log). They make no decision and never change a gate:
    * full-resolution context ceiling for G1b (all views at the focus res vs the focus views alone);
    * cross-pair (focus–context) pose AUC, with rotation-only and translation-only errors;
    * one scale per resolution group vs one per scene (are the groups only at different scales?);
    * context information on the focus grid (diag_up<res>): mismatch of patch scale vs loss of detail;
    * context-resolution sweep (any foveated configuration beyond the plan's);
    * the two-pass control (all views at the context res + focus views alone, aligned)."""
    p = plan["probe"]
    index = index_rows(rows)
    present = {c for (_, _, _, c) in index}
    hi, lo, prim, fo = p["hi"], p["lo"], p["primary"], p["focus_only"]
    out = {}

    def ratio(label, test, ref, metric, part, mask_cfg):
        if test in present and ref in present:
            out[label] = _ratio(index, plan, test, ref, metric, part, mask_cfg)

    def diff(label, test, ref, metric, part, mask_cfg):
        if test in present and ref in present:
            out[label] = _diff(index, plan, test, ref, metric, part, mask_cfg)

    ratio(f"G1b ceiling: focus AbsRel, {hi} / focus views alone", hi, fo, "abs_rel", "focus", hi)
    diff(f"G1b ceiling: focus-pair AUC@30, {hi} − focus views alone", hi, fo, "auc30", "focus", hi)
    mixed = sorted(c for c in present if c.startswith("fov") and "_random" not in c)
    for cfg in mixed:
        diff(f"cross-pair AUC@30, {cfg} − {lo}", cfg, lo, "auc30", "cross", cfg)
        diff(f"cross-pair AUC@30 rotation only, {cfg} − {lo}", cfg, lo, "auc30_rot", "cross", cfg)
        diff(f"cross-pair AUC@30 translation only, {cfg} − {lo}", cfg, lo, "auc30_trans", "cross", cfg)
        ratio(f"AbsRel all views, one scale per group: {cfg} / {lo}", cfg, lo, "abs_rel_group", "all", cfg)
    for cfg in sorted(c for c in present if c.startswith("diag_up")):
        diff(f"upsampled context: cross-pair AUC@30, {cfg} − {prim}", cfg, prim, "auc30", "cross", cfg)
        diff(f"upsampled context: context-pair AUC@30, {cfg} − {prim}", cfg, prim, "auc30", "context", cfg)
        ratio(f"upsampled context: focus AbsRel, {cfg} / {hi}", cfg, hi, "abs_rel", "focus", cfg)
    for cfg in sorted(c for c in present if c.startswith("twopass")):
        ratio(f"two-pass: focus AbsRel, {cfg} / {prim}", cfg, prim, "abs_rel", "focus", cfg)
        ratio(f"two-pass: focus AbsRel, {cfg} / {hi}", cfg, hi, "abs_rel", "focus", cfg)
        diff(f"two-pass: cross-pair AUC@30, {cfg} − {prim}", cfg, prim, "auc30", "cross", cfg)
        diff(f"two-pass: all-pair AUC@30, {cfg} − {prim}", cfg, prim, "auc30", "all", cfg)
        ratio(f"two-pass: scene-scale AbsRel, {cfg} / {prim}", cfg, prim, "abs_rel_scene", "all", cfg)
    return out


# ----------------------------------------------------------------------------- zoom (G2)
def zoom_gates(rows, plan) -> dict:
    """G2 per region selector: C1 region AbsRel ratio (zoom / global high-res) <= max_ratio; C2
    median R_v error (camera head) <= rv_max_deg. Primary selector at the plan's CI level, the
    secondary selectors with simultaneous (Bonferroni) CIs. Cluster = scene."""
    z = plan["zoom"]
    bs = plan["bootstrap"]
    level = plan["ci_level"]
    out = {}
    sels = [z["primary_selector"]] + list(z.get("secondary_selectors", []))
    lvl2 = S.bonferroni_level(level, len(sels) - 1)
    for sel in sels:
        lvl = level if sel == z["primary_selector"] else lvl2
        regions = [r for r in rows if r.get("selector", "oracle") == sel and r.get("methods")
                   and z["primary_method"] in r["methods"] and z["reference"] in r["methods"]]
        if not regions:
            out[f"G2 region zoom ({sel})"] = {"result": S.NA, "role": "primary" if sel == z["primary_selector"] else "secondary",
                                              "value": "no region with both methods", "criterion": ""}
            continue
        by_scene = defaultdict(lambda: {"a": [], "b": [], "rv": []})
        for r in regions:
            g = by_scene[r["scene"]]
            g["a"].append(r["methods"][z["primary_method"]]["abs_rel"])
            g["b"].append(r["methods"][z["reference"]]["abs_rel"])
            g["rv"].append(r["methods"][z["primary_method"]].get("rv_err_head_deg", np.nan))
        scenes = sorted(by_scene)
        a = np.array([_nanmean(by_scene[s]["a"]) for s in scenes])
        b = np.array([_nanmean(by_scene[s]["b"]) for s in scenes])
        est, lo, hi = S.ratio_of_means(a, b, n=bs["n"], seed=bs["seed"], level=lvl)
        c1 = S.at_most((lo, hi), z["max_ratio"])
        rv_est, rv_lo, rv_hi = S.median_ci([by_scene[s]["rv"] for s in scenes], n=bs["n"], seed=bs["seed"], level=lvl)
        c2 = S.at_most((rv_lo, rv_hi), z["rv_max_deg"])
        out[f"G2 region zoom ({sel})"] = {
            "result": S.combine_and(c1, c2), "role": "primary" if sel == z["primary_selector"] else "secondary",
            "value": (f"C1 region AbsRel {z['primary_method']}/{z['reference']} {S.fmt_ci(est, lo, hi)} -> {c1} "
                      + (f"(PASS for any threshold >= {hi:.3f}, FAIL below {lo:.3f}); " if np.isfinite(hi)
                         else "(no CI: needs >= 2 scenes); ")
                      + f"C2 median R_v error {S.fmt_ci(rv_est, rv_lo, rv_hi, 2)}° -> {c2}; "
                      f"{len(regions)} regions, {len(scenes)} scenes, {int(round(100 * lvl))}% CIs"),
            "criterion": f"C1 ratio <= {z['max_ratio']} AND C2 R_v error <= {z['rv_max_deg']}°",
            "C1": {"estimate": est, "ci_lo": lo, "ci_hi": hi, "result": c1, "wilcoxon_p": S.wilcoxon_p(a, b),
                   "threshold": z["max_ratio"], "rule": "at_most", "scenes": len(scenes), "level": lvl,
                   "label": f"C1 region AbsRel {z['primary_method']} / {z['reference']} ({sel})"},
            "C2": {"estimate": rv_est, "ci_lo": rv_lo, "ci_hi": rv_hi, "result": c2, "threshold": z["rv_max_deg"],
                   "rule": "at_most", "unit": "deg", "label": f"C2 median R_v error ({sel})"},
            "scenes": len(scenes)}
    return out
