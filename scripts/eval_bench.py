"""Phase 4: Sec. 6 benchmark (pose AUC, TSDF reconstruction F1 / Chamfer) on DA3-BENCH data.

The protocol code (dataset loaders, frame sampling with seed 42 and max 100 frames, evo Umeyama +
RANSAC alignment, Open3D TSDF fusion, metrics) is the released benchmark harness, imported from
--official-src. The model under test is OURS (`--engine ours`, default), loaded either with released
weights (--weights official:<preset>, setting [official]) or with our training checkpoint
(--weights path/to/last.pt, setting [reduced-scale]). `--engine official` runs the released API
instead, for cross-checking.

Data: <bench_root>/{hiroom,7scenes,dtu,dtu64,eth3d,scannetpp} extracted from HF
`depth-anything/DA3-BENCH`. The harness resolves paths relative to the CWD
(workspace/benchmark_dataset/...), so we run inside <work_root> with a symlink to <bench_root>.

  python scripts/eval_bench.py --official-src /kaggle/working/Depth-Anything-3/src \
      --bench-root /tmp/da3bench --work-root /kaggle/working/eval \
      --weights official:da3-small --datasets hiroom 7scenes --modes pose recon_unposed \
      --max-scenes 5 --tag official_da3-small
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def build_ours(weights: str, device: str):
    from da3 import DA3
    from da3.checkpoint_io import official_model

    if weights.startswith("official:"):
        model, info = official_model(weights.split(":", 1)[1])
        return model, "official", {"weights": weights}
    state = torch.load(weights, map_location="cpu", weights_only=False)
    model = DA3(state["config"]["model"])
    model.load_state_dict(state["model"])
    return model, "reduced-scale", {"weights": weights, "step": state.get("step"), "meta": state.get("meta")}


def cross_check_pose(work_dir: str, data: str, scenes) -> dict:
    """Recompute pose AUC with OUR metric code from the harness' saved predictions + GT meta."""
    from da3.eval.metrics import pose_metrics

    out = {}
    for scene in scenes:
        d = Path(work_dir) / "model_results" / data / scene / "unposed"
        res, gt = d / "exports" / "mini_npz" / "results.npz", d / "exports" / "gt_meta.npz"
        if res.exists() and gt.exists():
            out[scene] = pose_metrics(np.load(res)["extrinsics"], np.load(gt, allow_pickle=True)["extrinsics"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-src", required=True)
    ap.add_argument("--bench-root", required=True)
    ap.add_argument("--work-root", default="outputs/eval")
    ap.add_argument("--engine", choices=["ours", "official"], default="ours")
    ap.add_argument("--weights", default="official:da3-small")
    ap.add_argument("--datasets", nargs="+", default=["hiroom"])
    ap.add_argument("--modes", nargs="+", default=["pose", "recon_unposed", "recon_posed"])
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--max-scenes", type=int, default=0, help="0 = all scenes (limited runs are reported as such)")
    ap.add_argument("--use-ray-pose", action="store_true")
    ap.add_argument("--fusion-workers", type=int, default=4)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    work_root = Path(args.work_root).resolve()
    (work_root / "workspace").mkdir(parents=True, exist_ok=True)
    link = work_root / "workspace" / "benchmark_dataset"
    if not link.exists():
        try:
            os.symlink(Path(args.bench_root).resolve(), link, target_is_directory=True)
        except OSError:  # Windows without symlink privilege: directory junction
            import subprocess

            subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(Path(args.bench_root).resolve())], check=True)
    os.chdir(work_root)
    sys.path.insert(0, str(Path(args.official_src).resolve()))
    from depth_anything_3.bench.evaluator import Evaluator  # noqa: E402

    work_dir = str(work_root / "workspace" / "evaluation" / args.tag)
    probe = Evaluator(work_dir=work_dir, datas=args.datasets, modes=args.modes, max_frames=args.max_frames)
    scenes = None
    if args.max_scenes > 0:
        scenes = sorted({s for d in args.datasets for s in probe.datasets[d].SCENES[: args.max_scenes]})
    ev = Evaluator(work_dir=work_dir, datas=args.datasets, modes=args.modes, max_frames=args.max_frames,
                   scenes=scenes, ref_view_strategy="first", num_fusion_workers=args.fusion_workers)

    t0 = time.time()
    if args.engine == "ours":
        from da3.eval.bench_adapter import OursAsOfficialAPI

        model, setting, winfo = build_ours(args.weights, args.device)
        api = OursAsOfficialAPI(model, args.device, use_ray_pose=args.use_ray_pose)
    else:
        from depth_anything_3.api import DepthAnything3  # noqa: E402

        repo = {"official:da3-small": "depth-anything/DA3-SMALL", "official:da3-base": "depth-anything/DA3-BASE",
                "official:da3-large": "depth-anything/DA3-LARGE-1.1"}[args.weights]
        api = DepthAnything3.from_pretrained(repo).to(args.device)
        setting, winfo = "official (released code)", {"weights": repo}
    if not args.eval_only:
        ev.infer(api)
    t_inf = time.time() - t0
    metrics = ev.eval()
    t_all = time.time() - t0

    cc = {}
    if "pose" in args.modes:
        for d in args.datasets:
            ours = cross_check_pose(work_dir, d, [s for s in metrics.get(f"{d}_pose", {}) if s != "mean"])
            harness = metrics.get(f"{d}_pose", {})
            cc[d] = {"max_abs_diff_auc03": max((abs(ours[s]["auc03"] - harness[s]["auc03"]) for s in ours), default=None),
                     "max_abs_diff_auc30": max((abs(ours[s]["auc30"] - harness[s]["auc30"]) for s in ours), default=None)}

    from da3.train_utils import append_registry, environment_info, git_commit

    summary = {
        "tag": args.tag, "setting": setting, "engine": args.engine, **winfo,
        "datasets": args.datasets, "modes": args.modes, "max_frames": args.max_frames,
        "scenes": scenes or "all", "use_ray_pose": args.use_ray_pose,
        "means": {k: v.get("mean") for k, v in metrics.items()},
        "per_scene": metrics, "metric_crosscheck_ours_vs_harness": cc,
        "time_inference_s": t_inf, "time_total_s": t_all,
        "commit": git_commit(REPO), "env": environment_info(),
    }
    out = REPO / "experiments" / "eval" / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=float))
    append_registry(REPO / "experiments" / "registry.jsonl",
                    {"kind": "eval", **{k: summary[k] for k in ("tag", "setting", "engine", "datasets", "modes", "scenes", "means", "commit")}})
    print(json.dumps(summary["means"], indent=2, default=float))
    print("cross-check:", cc)
    print("wrote", out)


if __name__ == "__main__":
    main()
