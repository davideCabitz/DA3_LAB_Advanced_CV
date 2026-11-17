"""Exact-resume check: N uninterrupted steps vs N/2 steps + `--resume` for N/2 steps must give
identical losses (model, optimizer, scheduler, scaler, RNG and data order all restored).

  python scripts/resume_check.py --config configs/train/sanity_synthetic.yaml --steps 20 \
      --work /tmp/resume_check --out experiments/resume_check.json
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def run(cfg_path, out_dir, steps=None, resume=False):
    cmd = [sys.executable, "scripts/train.py", "--config", str(cfg_path), "--out-dir", str(out_dir)]
    if steps:
        cmd += ["--steps", str(steps)]
    if resume:
        cmd.append("--resume")
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    # "step k | ... total X | ..." -> {k: line without throughput / memory}
    return {int(m[1]): m[2] for m in re.finditer(r"^step (\d+) \| (.*?) \| \d+ tok/s", out, re.M)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--work", default=tempfile.mkdtemp())
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    half = args.steps // 2
    cfg = yaml.safe_load(open(args.config))
    cfg["train"].update(steps=args.steps, ckpt_every=half, log_every=1)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    cfg_path = work / "resume_check.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    a = run(cfg_path, work / "A")
    run(cfg_path, work / "B", steps=half)
    b = run(cfg_path, work / "B", resume=True)
    steps = sorted(k for k in b if k > half)
    same = all(a[k] == b[k] for k in steps)
    rep = {"config": args.config, "steps": args.steps, "resumed_at": half, "compared_steps": steps,
           "identical": same, "uninterrupted_last": a[args.steps], "resumed_last": b[args.steps]}
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2))
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
