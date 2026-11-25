"""Pre-registered analysis of the LoRA arms vs the released DA3 (docs/LORA_ANALYSIS_PLAN.md; constants in
configs/lora_analysis.yaml, whose SHA-256 is stored in the output).

Inputs:
  --kitti   kitti_eval.py JSON on the TEST drives with arms: released, <arm>_s<seed> (e.g. e3_s0, e3_s1,
            pose_only_s0, ..., own_output_s1)
  --sec6-released / --sec6-e3   da3-result/1 records (eval_bench.py) of the released model and of the E3
            adapters (one per seed), same protocol
Outputs (in --out): analysis.json, summary.md, and da3-result/1 records of the KITTI test evaluation
(released + every adapter) under <results-root>/{baseline,lora}/ for notebook 09.

  python scripts/lora_analysis.py --kitti test_da3-small.json --sec6-released r.json --sec6-e3 e3s0.json e3s1.json \
      --out /kaggle/working/lora/analysis --results-root /kaggle/working/lora/results
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from da3.lora_stats import (
    NOT_EVALUATED,
    decide_superiority,
    holm,
    non_inferiority,
    superiority_ratio,
)

REPO = Path(__file__).resolve().parents[1]
ENDPOINT_METRIC = {"kitti_absrel_clipscale": "absrel_clip", "kitti_ate": "ate", "kitti_rpe_trans": "rpe_trans",
                   "kitti_rpe_rot": "rpe_rot"}


def seed_arms(per_drive: dict, arm: str) -> list[str]:
    return sorted(a for a in per_drive if a.rsplit("_s", 1)[0] == arm and a.rsplit("_s", 1)[-1].isdigit())


def drive_values(per_drive: dict, arm: str, metric: str, drives: list[str]) -> tuple[np.ndarray, dict]:
    """Per-drive value of an arm, averaged over its seeds; also the per-seed arrays."""
    if arm == "released":
        v = np.array([per_drive["released"][d][metric] for d in drives])
        return v, {"released": v}
    names = seed_arms(per_drive, arm)
    if not names:
        return np.full(len(drives), np.nan), {}
    per_seed = {n: np.array([per_drive[n][d][metric] for d in drives]) for n in names}
    return np.mean(list(per_seed.values()), axis=0), per_seed


def kitti_hypothesis(h: dict, per_drive: dict, drives: list[str], alpha: float, eq: float) -> dict:
    metric = ENDPOINT_METRIC[h["endpoint"]]
    ref_arm = h.get("vs", "released")
    A, _ = drive_values(per_drive, ref_arm, metric, drives)
    B, seeds = drive_values(per_drive, h["arm"], metric, drives)
    r = superiority_ratio(A, B, alpha=alpha, eq_margin=eq)
    r.update(id=h["id"], arm=h["arm"], vs=ref_arm, endpoint=h["endpoint"], n_seeds=len(seeds))
    r["per_seed"] = {n: float(np.nanmean(v) / np.nanmean(A)) for n, v in seeds.items()}
    if seeds:
        signs = {np.sign(v - 1) for v in r["per_seed"].values()}
        r["seed_robust"] = len(signs) == 1 and np.sign(r["estimate"] - 1) in signs
    r["decision"] = decide_superiority(r, alpha) if len(seeds) >= 1 else NOT_EVALUATED
    return r


def sec6_unit_values(records: list[dict], dataset: str, metric: str, mode: str) -> dict:
    """{scene: mean over records} of one Sec. 6 metric (AUC / F1 in points, Chamfer raw)."""
    key = f"{dataset}_{mode}"
    vals = {}
    for rec in records:
        for scene, m in rec.get("sec6_per_scene", {}).get(key, {}).items():
            if scene == "mean" or not isinstance(m, dict) or metric not in m:
                continue
            v = float(m[metric]) * (100.0 if metric in ("auc03", "auc30", "fscore") else 1.0)
            vals.setdefault(scene, []).append(v)
    return {s: float(np.mean(v)) for s, v in vals.items() if len(v) == len(records)}


def sec6_component(c: dict, released: dict | None, e3: list[dict], alpha: float) -> dict:
    if released is None or not e3:
        return {"id": c["id"], "decision": NOT_EVALUATED, "reason": "Sec. 6 records missing"}
    metric = {"auc03": "auc03", "auc30": "auc30", "fscore": "fscore", "chamfer": "overall"}[c["metric"]]
    mode = c.get("mode", "pose")
    A, B, groups, missing = [], [], [], []
    for d in c["datasets"]:
        a, b = sec6_unit_values([released], d, metric, mode), sec6_unit_values(e3, d, metric, mode)
        common = sorted(set(a) & set(b))
        if not common:
            missing.append(d)
        A += [a[s] for s in common]; B += [b[s] for s in common]; groups += [d] * len(common)
    r = non_inferiority(A, B, groups, margin_abs=c.get("margin_abs"), margin_rel=c.get("margin_rel"), alpha=alpha)
    r.update(id=c["id"], metric=c["metric"], mode=mode, planned_datasets=c["datasets"], missing_datasets=missing)
    if missing and r["decision"] != NOT_EVALUATED:
        r["note"] = f"not evaluated on {missing}: the claim is restricted to {r.get('datasets')}"
    return r


def fmt(r: dict) -> str:
    if r.get("decision") == NOT_EVALUATED:
        return f"| {r['id']} | – | – | – | **{NOT_EVALUATED}** {r.get('reason', '')} |"
    return (f"| {r['id']} | {r['estimate']:.4f} | [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | {r['p']:.4f} "
            f"| **{r['decision']}** |")


def kitti_records(res: dict, plan_sha: str, preset: str, results_root: Path) -> list[str]:
    from da3.compare import make_record, protocol, save_record

    per_drive, drives = res["summary"]["per_drive"], res["drives"]
    proto = protocol(["kitti_test"], drives, res["args"]["views"], ["clip"], "first")
    paths = []
    for arm in per_drive:
        if arm == "posed":
            continue
        means = {m: float(np.mean([per_drive[arm][d][m] for d in drives])) for m in next(iter(per_drive[arm].values()))}
        ad = res.get("adapters", {}).get(arm)
        tag = f"kitti_test_{preset}_{arm}"
        rec = make_record(enhancement="lora" if ad else "baseline", variant=arm if ad else "head", tag=tag,
                          setting="official+LoRA" if ad else "official", weights=f"official:{preset}", protocol=proto,
                          extra={"kitti": {"all": means, **{d: per_drive[arm][d] for d in drives}}},
                          baseline_tag=None if not ad else f"kitti_test_{preset}_released", commit=res.get("commit", "unknown"),
                          env=res.get("env", {}), notes=f"KITTI held-out drives, plan sha256 {plan_sha[:12]}",
                          adapter=ad)
        paths.append(str(save_record(rec, results_root)))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti", required=True)
    ap.add_argument("--sec6-released", default=None)
    ap.add_argument("--sec6-e3", nargs="*", default=[])
    ap.add_argument("--plan", default=str(REPO / "configs" / "lora_analysis.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--results-root", default=None)
    args = ap.parse_args()

    plan_text = Path(args.plan).read_text(encoding="utf-8")
    plan, plan_sha = yaml.safe_load(plan_text), hashlib.sha256(plan_text.encode()).hexdigest()
    alpha = float(plan["alpha_one_sided"])
    res = json.loads(Path(args.kitti).read_text())
    per_drive, drives = res["summary"]["per_drive"], res["drives"]
    preset = res["args"]["preset"]
    seeds_used = sorted({a.rsplit("_s", 1)[-1] for a in per_drive if a not in ("released", "posed")})
    warnings = []
    for a in res.get("adapters", {}).values():
        if a.get("plan_sha256") and a["plan_sha256"] != plan_sha:
            warnings.append(f"adapter {a['path']} was trained under plan sha {a['plan_sha256'][:12]}, analysis uses {plan_sha[:12]}")
    if len(seeds_used) < int(plan["min_seeds"]):
        warnings.append(f"only {len(seeds_used)} seed(s): fewer than the plan's minimum {plan['min_seeds']} -> no confirmatory claim")

    h1 = kitti_hypothesis(plan["primary"], per_drive, drives, alpha, float(plan["primary"]["equivalence_margin"]))
    sec = [kitti_hypothesis(h, per_drive, drives, alpha, float(plan["primary"]["equivalence_margin"]))
           for h in plan["secondary_holm"]]
    adj = holm({r["id"]: r["p"] for r in sec if r["decision"] != NOT_EVALUATED}, alpha)
    gated = h1["decision"] == "PASS"
    for r in sec:
        if r["id"] in adj:
            r["p_holm"], rejected = adj[r["id"]]
            if not gated:
                r["label"] = "exploratory (H1 did not pass)"
            elif not rejected and r["decision"] == "PASS":
                r["decision"] = "INCONCLUSIVE"   # p < alpha alone, but not after Holm
    released = json.loads(Path(args.sec6_released).read_text()) if args.sec6_released else None
    e3 = [json.loads(Path(p).read_text()) for p in args.sec6_e3]
    ni = []
    for c in plan["non_inferiority"]["components"]:
        if c["benchmark"] == "mono":
            ni.append({"id": c["id"], "decision": NOT_EVALUATED, "reason": "plan v2: monocular check not evaluated"})
        else:
            ni.append(sec6_component(c, released, e3, alpha))
    evaluated = [r for r in ni if r["decision"] != NOT_EVALUATED]
    no_regression = "PASS" if evaluated and all(r["decision"] == "PASS" for r in evaluated) else \
        "FAIL" if any(r["decision"] == "FAIL" for r in evaluated) else "INCONCLUSIVE" if evaluated else NOT_EVALUATED
    if len(seeds_used) < int(plan["min_seeds"]):
        h1["decision_note"] = "fewer seeds than the plan's minimum: descriptive only"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    analysis = {"plan_version": plan["version"], "plan_sha256": plan_sha, "model": preset, "drives": drives,
                "seeds": seeds_used, "H1": h1, "secondary": sec, "non_inferiority": ni,
                "no_regression_claim": no_regression, "warnings": warnings}
    if args.results_root:
        analysis["records"] = kitti_records(res, plan_sha, preset, Path(args.results_root))
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, default=float))

    md = [f"# LoRA vs released DA3 ({preset}): pre-registered analysis (plan v{plan['version']}, sha {plan_sha[:12]})\n",
          f"{len(drives)} held-out KITTI test drives; seeds {seeds_used}. Ratios = LoRA / reference (lower is better). "
          "Decisions follow docs/LORA_ANALYSIS_PLAN.md §3-4; nothing here was chosen after seeing the results.\n"]
    md += [f"> **Warning:** {w}" for w in warnings]
    md += ["\n## Primary (H1) and secondary (Holm, gated by H1)\n",
           "| ID | ratio | 95% CI | p (one-sided) | decision |", "|---|---|---|---|---|", fmt(h1)]
    md += [fmt(r) + (f" Holm p {r['p_holm']:.4f}" if "p_holm" in r else "") + (f" _{r['label']}_" if "label" in r else "")
           for r in sec]
    md += ["\n| ID | per-seed ratios | seed-robust | units improved | d_z |", "|---|---|---|---|---|"]
    md += [f"| {r['id']} | " + ", ".join(f"{k} {v:.3f}" for k, v in r.get("per_seed", {}).items())
           + f" | {r.get('seed_robust', '–')} | {r.get('frac_improved', float('nan')):.2f} | {r.get('d_z', float('nan')):.2f} |"
           for r in [h1, *sec] if r["decision"] != NOT_EVALUATED]
    md += ["\n## Non-inferiority on the paper's benchmarks (intersection-union test)\n",
           "| ID | estimate (points or ratio) | 95% CI | p | decision |", "|---|---|---|---|---|"]
    md += [fmt(r) + (f" {r['note']}" if r.get("note") else "") for r in ni]
    md.append(f"\n**No-regression claim: {no_regression}** (only over the components that were evaluated).")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
