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

Outputs: experiments/eval/<tag>.json (legacy summary, read by make_tables.py) and the standard
result record experiments/results/<enhancement>/<tag>.json (da3.compare; read by notebook
09_compare_all). `--enhancement/--variant` label the run (default: baseline / head).

Enhancement hooks:
* `--dump-audit`: also write the Self-Audit dump next to each unposed export (enhancement #1;
  official RANSAC ray poses + E_c re-pass). The exported results are unchanged.
* `--eval-only`: score exports that already exist in <work-root>/workspace/evaluation/<tag>
  (e.g. pose-source variants written by scripts/self_audit_eval.py). No model is loaded.
* `--infer-only`: the opposite: run the model and write the exports, no scoring. Used when the
  scoring must run in another Python (Kaggle / Colab Python 3.13 only get Open3D 0.20.0, whose TSDF
  fusion is broken, so the notebooks score in a Python 3.12 + Open3D 0.19.0 side environment).
* `--process-res R`: run the original DA3 at another uniform resolution (default 504, the paper's).
* `--meter`: also write per-scene efficiency meta (tokens, time, peak memory) next to each export;
  predictions unchanged (enhancement #2 uses it for the uniform-resolution rows).
* `--foveated SPEC`: enhancement #2, Foveated DA3: mixed per-view resolutions in one pass, e.g.
  "fraction=0.25,focus_res=504,context_res=252[,balance=1]" (da3.enhancements.foveated).
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


def official_ray_solver(ray: torch.Tensor, ray_conf: torch.Tensor, hw):
    """Ray map -> (w2c, K) with the released confidence-weighted RANSAC routine (the code path of
    the official API's `use_ray_pose=True`). ray: (B, S, h, w, 6) in OUR [origin, direction] order."""
    from depth_anything_3.utils.ray_utils import get_extrinsic_from_camray

    from da3.camera import affine_inverse

    H, W = hw
    raw = torch.cat([ray[..., 3:6], ray[..., 0:3]], dim=-1).float()   # released order [direction, origin]
    torch.manual_seed(0)  # the routine samples RANSAC hypotheses
    ext, focal, pp = get_extrinsic_from_camray(raw, ray_conf.float(), ray.shape[-3], ray.shape[-2])
    w2c = affine_inverse(ext)[..., :3, :]
    K = torch.zeros(w2c.shape[:-2] + (3, 3), dtype=w2c.dtype, device=w2c.device)
    K[..., 0, 0] = focal[..., 0] / 2 * W
    K[..., 1, 1] = focal[..., 1] / 2 * H
    K[..., 0, 2] = pp[..., 0] * W * 0.5
    K[..., 1, 2] = pp[..., 1] * H * 0.5
    K[..., 2, 2] = 1.0
    return w2c, K


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
    ap.add_argument("--infer-only", action="store_true",
                    help="run the model and write the exports, but do not score them (no metrics, no record)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="resume: do not run the model again on scenes whose exports (and audit dump) are complete")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # standard result record (da3.compare) and enhancement hooks
    ap.add_argument("--enhancement", default="baseline")
    ap.add_argument("--variant", default=None, help="default: 'ray_dlt' with --use-ray-pose, else 'head'")
    ap.add_argument("--baseline-tag", default=None)
    ap.add_argument("--setting", default=None, help="default: 'official' for official:* weights, else 'reduced-scale'")
    ap.add_argument("--results-root", default=str(REPO / "experiments" / "results"))
    ap.add_argument("--dump-audit", action="store_true", help="write the Self-Audit dump (enhancement #1)")
    ap.add_argument("--no-repass", action="store_true", help="with --dump-audit: skip the E_c re-pass")
    ap.add_argument("--process-res", type=int, default=504, help="uniform processing resolution (longest side)")
    ap.add_argument("--meter", action="store_true", help="write per-scene efficiency meta (enhancement #2)")
    ap.add_argument("--foveated", default="", help="enhancement #2 spec, e.g. 'fraction=0.25,context_res=252'")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    if args.foveated and (args.dump_audit or args.engine != "ours"):
        ap.error("--foveated needs --engine ours and cannot be combined with --dump-audit")
    if args.eval_only and args.infer_only:
        ap.error("--eval-only and --infer-only are mutually exclusive")

    from da3.eval.harness import prepare_workspace

    work_root = prepare_workspace(args.work_root, args.bench_root)
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
    api, dumper = None, None
    if args.eval_only:
        setting = args.setting or ("official" if args.weights.startswith("official:") else "reduced-scale")
        winfo = {"weights": args.weights}
    elif args.engine == "ours":
        from da3.eval.bench_adapter import OursAsOfficialAPI

        model, setting, winfo = build_ours(args.weights, args.device)
        if args.dump_audit:
            from da3.enhancements.self_audit import AuditDumper

            dumper = AuditDumper(ray_solver=official_ray_solver, repass=not args.no_repass)
        if args.foveated:
            from da3.enhancements.foveated import FoveationSpec
            from da3.enhancements.foveated.adapter import FoveatedAPI

            api = FoveatedAPI(model, args.device, spec=FoveationSpec.parse(args.foveated),
                              use_ray_pose=args.use_ray_pose)
        elif args.meter:
            from da3.enhancements.foveated.adapter import MeteredAPI

            api = MeteredAPI(model, args.device, process_res=args.process_res, use_ray_pose=args.use_ray_pose)
            api.dumper = dumper
        else:
            api = OursAsOfficialAPI(model, args.device, use_ray_pose=args.use_ray_pose, dumper=dumper,
                                    process_res=args.process_res, skip_existing=args.skip_existing)
    else:
        from depth_anything_3.api import DepthAnything3  # noqa: E402

        repo = {"official:da3-small": "depth-anything/DA3-SMALL", "official:da3-base": "depth-anything/DA3-BASE",
                "official:da3-large": "depth-anything/DA3-LARGE-1.1"}[args.weights]
        api = DepthAnything3.from_pretrained(repo).to(args.device)
        setting, winfo = "official (released code)", {"weights": repo}
    if not args.eval_only:
        ev.infer(api)
        if getattr(api, "skipped", 0):
            print(f"[eval_bench] resumed: {api.skipped} scene(s) already done, not run again")
    t_inf = time.time() - t0
    if args.infer_only:  # scoring happens in a separate --eval-only run (e.g. the Python 3.12 environment)
        if dumper is not None and dumper.failures:
            print("[WARN] audit dump issues:\n  " + "\n  ".join(dumper.failures))
        print(f"[eval_bench] inference done in {t_inf:.0f} s; exports in {work_dir}. "
              f"Score them with the same arguments and --eval-only.")
        return
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

    efficiency = api.summary() if hasattr(api, "summary") else {}
    summary = {
        "tag": args.tag, "setting": setting, "engine": args.engine, **winfo,
        "datasets": args.datasets, "modes": args.modes, "max_frames": args.max_frames,
        "scenes": scenes or "all", "use_ray_pose": args.use_ray_pose,
        "process_res": args.process_res, "foveated": args.foveated or None, "efficiency": efficiency,
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
    from da3.compare import make_record, protocol, save_record

    variant = args.variant or ("ray_dlt" if args.use_ray_pose else "head")
    run_notes = [args.notes] if args.notes else []
    if args.engine != "ours":
        run_notes.append(f"engine={args.engine}")
    if args.process_res != 504 and not args.foveated:
        run_notes.append(f"process_res={args.process_res}")
    if args.foveated:
        run_notes.append(f"foveated: {args.foveated}")
    record = make_record(
        enhancement=args.enhancement, variant=variant, tag=args.tag, setting=setting, weights=args.weights,
        protocol=protocol(args.datasets, scenes or "all", args.max_frames, args.modes, "first"),
        sec6=summary["means"], sec6_per_scene=metrics, baseline_tag=args.baseline_tag,
        extra={"efficiency": {"all": efficiency}} if efficiency else None,
        commit=summary["commit"], env=summary["env"], notes="; ".join(run_notes),
    )
    rec_path = save_record(record, args.results_root)
    if dumper is not None and dumper.failures:
        print("[WARN] audit dump issues:\n  " + "\n  ".join(dumper.failures))
    print(json.dumps(summary["means"], indent=2, default=float))
    print("cross-check:", cc)
    print("wrote", out, "and", rec_path)


if __name__ == "__main__":
    main()
