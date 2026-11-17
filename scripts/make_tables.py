"""Build the Phase 4 results tables from experiments/eval/*.json.

Rows are grouped by setting: (a) [paper] numbers copied from Tab. 2 / Tab. 3, (b) [official]
released weights evaluated by us, (c) [reduced-scale] our training. Pose: AUC@3 / AUC@30 (x100).
Reconstruction: F1 (x100), DTU: Chamfer distance "overall" (mm). DTU pose uses the dtu64 split
(as in the released benchmark).

  python scripts/make_tables.py --out reports/results_tables.md
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COLS = ["hiroom", "eth3d", "dtu", "7scenes", "scannetpp"]
NAMES = {"hiroom": "HiRoom", "eth3d": "ETH3D", "dtu": "DTU", "7scenes": "7Scenes", "scannetpp": "ScanNet++"}

# Tab. 2 (Auc3, Auc30) and Tab. 3 (w/o pose, w/ pose; F1, DTU = CD mm) of the paper.
PAPER_POSE = {
    "DA3-Giant": [(80.3, 95.9), (48.4, 91.2), (94.1, 99.4), (28.5, 86.8), (85.0, 98.1)],
    "DA3-Large": [(58.7, 94.2), (32.2, 86.9), (70.2, 96.7), (29.2, 86.6), (60.2, 94.7)],
    "DA3-Base": [(19.0, 83.2), (15.1, 74.6), (60.1, 95.9), (20.1, 82.9), (25.1, 83.4)],
    "DA3-Small": [(9.49, 75.2), (8.59, 62.1), (30.6, 91.2), (14.0, 78.7), (10.9, 71.9)],
}
PAPER_RECON = {
    "DA3-Giant": [(85.1, 95.6), (79.0, 87.1), (1.85, 1.85), (53.5, 56.5), (77.0, 79.3)],
    "DA3-Large": [(69.5, 87.1), (65.8, 75.2), (2.08, 1.23), (56.3, 49.2), (67.9, 75.7)],
    "DA3-Base": [(25.9, 71.4), (49.5, 66.7), (2.87, 2.36), (49.9, 50.6), (47.2, 67.8)],
    "DA3-Small": [(18.3, 52.2), (41.6, 63.4), (5.83, 2.49), (41.0, 46.8), (32.3, 53.8)],
}


def fmt(v, scale=100.0):
    return "–" if v is None else f"{v * scale:.1f}"


def ours_row(summary):
    m = summary["means"]
    pose, recon = [], []
    for c in COLS:
        pk = f"{'dtu64' if c == 'dtu' else c}_pose"
        p = m.get(pk) or {}
        pose += [fmt(p.get("auc03")), fmt(p.get("auc30"))]
        for mode in ("recon_unposed", "recon_posed"):
            r = m.get(f"{c}_{mode}") or {}
            recon.append(fmt(r.get("overall"), 1.0) if c == "dtu" and r else fmt(r.get("fscore")))
    return pose, recon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=str(REPO / "experiments" / "eval"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = [json.loads(p.read_text()) for p in sorted(Path(args.eval_dir).glob("*.json"))]

    h_pose = "| Setting | Model | " + " | ".join(f"{NAMES[c]} Auc3 | {NAMES[c]} Auc30" for c in COLS) + " | scenes |"
    h_rec = "| Setting | Model | " + " | ".join(
        f"{NAMES[c]} {'CD↓' if c == 'dtu' else 'F1'} w/o p. | w/ p." for c in COLS) + " | scenes |"
    sep = lambda h: "|" + "---|" * (h.count("|") - 1)  # noqa: E731
    pose_rows, rec_rows = [h_pose, sep(h_pose)], [h_rec, sep(h_rec)]
    for name in PAPER_POSE:
        pose_rows.append(f"| (a) paper | {name} | " + " | ".join(f"{a} | {b}" for a, b in PAPER_POSE[name]) + " | all |")
        rec_rows.append(f"| (a) paper | {name} | " + " | ".join(f"{a} | {b}" for a, b in PAPER_RECON[name]) + " | all |")
    for s in runs:
        tag = "(b) official, run by us" if s["setting"].startswith("official") else "(c) reduced-scale"
        p, r = ours_row(s)
        scenes = "all" if s["scenes"] == "all" else f"{len(s['scenes'])} (limited)"
        pose_rows.append(f"| {tag} | {s['tag']} | " + " | ".join(p) + f" | {scenes} |")
        rec_rows.append(f"| {tag} | {s['tag']} | " + " | ".join(r) + f" | {scenes} |")
    md = ("### Pose accuracy (AUC, x100)\n\n" + "\n".join(pose_rows)
          + "\n\n### Reconstruction (F1 x100; DTU: Chamfer distance, mm)\n\n" + "\n".join(rec_rows) + "\n")
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
