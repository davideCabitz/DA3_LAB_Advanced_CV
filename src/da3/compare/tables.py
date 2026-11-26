"""Comparison tables: original DA3 vs enhancement variants, from standard result records.

* `sec6_table`: paper-protocol metrics (pose AUC@3/30, F1 without / with poses; DTU Chamfer) for
  every record, grouped by protocol (weights + datasets + scenes + frame sampling). Each row shows
  the value and the difference to the reference row of its group: the original DA3 with its API
  default (enhancement "baseline", variant "head").
* `extra_table`: enhancement-specific metric groups (e.g. Self-Audit failure detection), one row
  per (record, dataset).

Tables are returned as lists of dicts and rendered to Markdown without pandas; notebooks can wrap
them in a DataFrame (`pandas.DataFrame(rows)`) if they want.
"""

from __future__ import annotations

import math
from collections import defaultdict

from .schema import protocol_key

REFERENCE = ("baseline", "head")
DATA_NAMES = {"hiroom": "HiRoom", "eth3d": "ETH3D", "dtu": "DTU", "dtu64": "DTU-64", "7scenes": "7Scenes",
              "scannetpp": "ScanNet++"}
# (mode, key, label, scale, higher_is_better)
SEC6_METRICS = [
    ("pose", "auc03", "AUC@3", 100.0, True),
    ("pose", "auc30", "AUC@30", 100.0, True),
    ("recon_unposed", "fscore", "F1 w/o p.", 100.0, True),
    ("recon_posed", "fscore", "F1 w/ p.", 100.0, True),
]
DTU_RECON = {"recon_unposed": ("overall", "CD w/o p. (mm)", 1.0, False),
             "recon_posed": ("overall", "CD w/ p. (mm)", 1.0, False)}


def _num(v):
    return v if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) else None


def _fmt(v, digits=1):
    return "–" if v is None else f"{v:.{digits}f}"


def _fmt_delta(d, higher_is_better, digits=1):
    if d is None:
        return ""
    sign = "+" if d >= 0 else "−"
    better = (d > 0) == higher_is_better if abs(d) > 10 ** (-digits) / 2 else None
    mark = "" if better is None else (" ▲" if better else " ▼")
    return f" ({sign}{abs(d):.{digits}f}{mark})"


def sec6_columns(datasets) -> list[tuple]:
    """[(data, mode, key, label, scale, higher_is_better)] for the given datasets."""
    cols = []
    for d in datasets:
        for mode, key, label, scale, hib in SEC6_METRICS:
            if d == "dtu64" and mode != "pose":
                continue  # the harness evaluates DTU-64 for pose only
            if d == "dtu" and mode == "pose":
                continue  # DTU pose is reported on the dtu64 split
            if d == "dtu" and mode in DTU_RECON:
                key, label, scale, hib = DTU_RECON[mode]
            cols.append((d, mode, key, label, scale, hib))
    return cols


def model_name(weights: str) -> str:
    return weights.split(":", 1)[1] if weights.startswith("official:") else weights


def sec6_table(records: list[dict]) -> list[dict]:
    """One table per protocol group: {"group": ..., "columns": [...], "rows": [...]} ."""
    groups = defaultdict(list)
    for r in records:
        if r.get("sec6"):
            groups[protocol_key(r)].append(r)
    tables = []
    for key, recs in groups.items():
        datasets = list(key[1])
        cols = [c for c in sec6_columns(datasets)
                if any(_num(r["sec6"].get(f"{c[0]}_{c[1]}", {}).get(c[2])) is not None for r in recs)]
        ref = next((r for r in recs if (r["enhancement"], r["variant"]) == REFERENCE), None)
        rows = []
        order = sorted(recs, key=lambda r: (r["enhancement"] != "baseline", r["enhancement"], r["variant"]))
        for r in order:
            row = {"model": model_name(r["weights"]), "enhancement": r["enhancement"], "variant": r["variant"],
                   "tag": r["tag"], "setting": r["setting"], "values": {}, "deltas": {}}
            for d, mode, k, label, scale, hib in cols:
                v = _num(r["sec6"].get(f"{d}_{mode}", {}).get(k))
                v = None if v is None else v * scale
                row["values"][(d, label)] = v
                if ref is not None and r is not ref:
                    rv = _num(ref["sec6"].get(f"{d}_{mode}", {}).get(k))
                    row["deltas"][(d, label)] = None if (v is None or rv is None) else v - rv * scale
            rows.append(row)
        scenes = key[2]
        tables.append({"weights": key[0], "datasets": datasets,
                       "scenes": "all" if scenes == "all" else f"{len(scenes)} (limited)",
                       "max_frames": key[3], "input": key[5] if len(key) > 5 else None,
                       "reference": None if ref is None else ref["tag"],
                       "columns": cols, "rows": rows})
    return tables


def sec6_markdown(records: list[dict]) -> str:
    out = []
    for t in sec6_table(records):
        out.append(f"#### {model_name(t['weights'])} · datasets {', '.join(t['datasets'])} · scenes {t['scenes']} "
                   f"· ≤ {t['max_frames']} frames" + (f" · input {t['input']}" if t.get("input") else ""))
        out.append("Differences in brackets are relative to the original DA3 (API default: camera-head poses)"
                   + ("" if t["reference"] else "; **no reference run in this group**") + ". ▲ better, ▼ worse.\n")
        head = ["Model", "Enhancement", "Variant"] + [f"{DATA_NAMES.get(d, d)} {label}" for d, _, _, label, _, _ in t["columns"]]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for row in t["rows"]:
            cells = [row["model"], row["enhancement"], row["variant"]]
            for d, _, _, label, _, hib in t["columns"]:
                cells.append(_fmt(row["values"][(d, label)]) + _fmt_delta(row["deltas"].get((d, label)), hib))
            lines.append("| " + " | ".join(cells) + " |")
        out.append("\n".join(lines) + "\n")
    return "\n".join(out) if out else "_No Sec. 6 results found._\n"


def extra_table(records: list[dict], enhancement: str, group: str, metrics: list[str] | None = None,
                digits: int = 3) -> tuple[list[str], list[list[str]]]:
    """Rows (model, variant, dataset, metric values...) for one extra-metric group."""
    recs = [r for r in records if r["enhancement"] == enhancement and group in r.get("extra", {})]
    if not recs:
        return [], []
    if metrics is None:
        seen = []
        for r in recs:
            for per in r["extra"][group].values():
                if isinstance(per, dict):
                    for m in per:
                        if m not in seen and not isinstance(per[m], (dict, list)):
                            seen.append(m)
        metrics = seen
    head = ["Model", "Variant", "Dataset"] + metrics
    rows = []
    for r in sorted(recs, key=lambda r: (r["weights"], r["variant"])):
        for ds, per in r["extra"][group].items():
            if not isinstance(per, dict):
                continue
            cells = [model_name(r["weights"]), r["variant"], DATA_NAMES.get(ds, ds)]
            for m in metrics:
                v = per.get(m)
                if isinstance(v, bool):
                    cells.append("PASS" if v else "FAIL")
                elif _num(v) is not None:
                    cells.append(f"{v:.{digits}f}")
                elif isinstance(v, str):
                    cells.append(v)
                else:
                    cells.append("–")
            rows.append(cells)
    return head, rows


def extra_markdown(records: list[dict], enhancement: str, group: str, metrics=None, digits=3) -> str:
    head, rows = extra_table(records, enhancement, group, metrics, digits)
    if not head:
        return f"_No `{group}` results for `{enhancement}`._\n"
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def summary_markdown(records: list[dict]) -> str:
    """Everything: Sec. 6 tables for all groups, then every extra group of every enhancement."""
    parts = ["## Paper protocol (Sec. 6): original DA3 vs enhancements\n", sec6_markdown(records)]
    groups = defaultdict(set)
    for r in records:
        for g in r.get("extra", {}):
            groups[r["enhancement"]].add(g)
    for enh in sorted(groups):
        parts.append(f"## Enhancement-specific metrics: {enh}\n")
        for g in sorted(groups[enh]):
            parts.append(f"### {g}\n")
            parts.append(extra_markdown(records, enh, g))
    return "\n".join(parts)
