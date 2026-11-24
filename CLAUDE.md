# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project context

This is a research project (Advanced Computer Vision lab) on **Depth Anything 3 (DA3)**.

Goals:
1. **Reproduce** the paper [papers/DA3_Paper.pdf](papers/DA3_Paper.pdf): *Depth Anything 3: Recovering the Visual Space from Any Views* (Lin, Chen, Liew, Chen et al., ByteDance Seed, arXiv:2511.10647, Nov 2025). Project page: depth-anything-3.github.io.
2. **Find points of enhancement** and try to obtain a better model (or at least a well-documented attempt), with each change measured against the reproduced baseline.

When in doubt about a method detail, check the paper first (section references below) rather than guessing.

## Paper summary (what we are reproducing)

DA3 predicts spatially consistent geometry from an arbitrary number of images, with or without known camera poses. Two core claims:
- A **single plain transformer** (vanilla DINOv2 ViT) is enough as backbone, no architectural specialization.
- A **depth + ray** prediction target is a minimal, sufficient target set; no complex multi-task learning needed.

### Formulation (Sec. 3.1)
- Input: N_v images `I_i ∈ R^{H×W×3}`. N_v = 1 → monocular; N_v > 1 → multi-view / video.
- Outputs per view: depth map `D ∈ R^{H×W}` and ray map `M ∈ R^{H×W×6}` (ray origin `t ∈ R^3` + **unnormalized** direction `d = R K^{-1} p`).
- 3D point: `P = t + D(u,v) · d`.
- Camera recovery from the ray map: center = mean of ray origins (Eq. 1); `R` and `K` via homography `H = KR` solved with DLT (Eq. 2), then RQ decomposition.
- A lightweight **camera head D_C** on the camera tokens predicts FOV `f ∈ R^2`, quaternion `q ∈ R^4`, translation `t ∈ R^3` (cheap alternative to ray-map pose recovery at inference).

### Architecture (Sec. 3.2, Fig. 2–3)
- ViT with L blocks split into `L_s` (within-view self-attention only) and `L_g` (alternating cross-view / within-view attention via token reordering). Ratio `L_s : L_g = 2 : 1` (ablation in Tab. 7). With one image it reduces to monocular depth at no extra cost.
- **Camera condition injection**: each view gets a camera token `c_i`; from an MLP encoder `E_c(f_i, q_i, t_i)` if poses are known, else a shared learnable token. Tokens participate in all attention.
- **Dual-DPT head**: shared reassemble modules, two separate fusion stacks + output layers (depth branch and ray branch). Ray map is predicted at H/2 × W/2.

### Training (Sec. 3.3–3.4)
- Loss: `L = L_D(D̂, D) + L_M(R̂, M) + L_P(D̂ ⊙ d + t, P) + β L_C(ĉ, v) + α L_grad(D̂, D)`, α = β = 1, all ℓ1.
  - `L_D` is confidence-weighted: `(1/Z_Ω) Σ m_p (D_c,p |D̂_p − D_p| − λ_c log D_c,p)`.
  - `L_grad`: ℓ1 on horizontal/vertical finite-difference depth gradients (Eq. 3).
  - All GT normalized by a common scale = mean ℓ2 norm of valid reprojected points.
- 128×H100, 200k steps, 8k warm-up, peak LR 2e-4. Base res 504×504 (plus 504×378, 504×336, 504×280, 336×504, 896×504, 756×504, 672×504). Views sampled uniformly in [2, 18] at 504². Dynamic batch to keep token count constant.
- Supervision switches from GT depth to **teacher pseudo-labels at 120k steps**. Pose conditioning activated with p = 0.2.
- Training datasets: Tab. 1 (AriaDigitalTwin, AriaSyntheticENV, ARKitScenes, BlendedMVS, Co3Dv2, DL3DV, HyperSim, MapFree, MegaDepth, MegaSynth, MvsSynth, Objaverse, OmniObject, OmniWorld, PointOdyssey, ReplicaVMAP, ScanNet++ (scene-disjoint from test), ScenenetRGBD, TartanAir, Trellis, vKitti2, WildRGBD). All public/academic.

### Teacher–student learning (Sec. 4)
- **Teacher**: monocular relative-depth model (DINOv2 + DPT) trained **only on synthetic data** (large corpus listed in Sec. 4.1). Predicts scale-shift-invariant **exponential depth** (not disparity as in DA2). Also predicts sky mask and object mask.
  - Loss `L_T = α L_grad + L_gl + L_N + L_sky + L_obj`, α = 0.5; `L_gl` = ROE-aligned global-local loss; `L_N` = distance-weighted surface-normal loss (Eq. 4–6).
- **Teaching DA3** (Sec. 4.2): teacher relative depth aligned to sparse/noisy real GT via RANSAC least-squares scale/shift (Eq. 8).
- **Monocular student** (Sec. 4.3) and **metric model** (Sec. 4.4, canonical focal f^c = 300, 14 datasets, 160k iters, bs 64).

### Feed-forward 3DGS (Sec. 5)
- GS-DPT head predicting per-pixel Gaussians (opacity, rotation, scale, color/SH) + depth offset; DA3 backbone frozen. Trained on DL3DV (10,015 scenes), photometric (MSE + LPIPS) + depth loss. Pose-conditioned and pose-adaptive variants.

### Benchmark (Sec. 6)
- Pose-geometry datasets: HiRoom (synthetic, 30 scenes), ETH3D (11), DTU (22), 7Scenes (7), ScanNet++ (20). Up to 100 images/scene (fixed seed).
- **Pose metrics**: AUC over min(RRA, RTA), reported at thresholds 3 and 30.
- **Reconstruction**: align predicted poses to GT with evo + RANSAC, TSDF-fuse predicted depth, then Chamfer distance and F1 (thresholds: HiRoom/7Scenes/ScanNet++ 0.05 m, ETH3D 0.25 m; per-dataset TSDF voxel sizes in Sec. 6.3).
- **NVS**: DL3DV (140 scenes), Tanks and Temples (6), MegaDepth (19); PSNR / SSIM / LPIPS; 12 context views via farthest-point sampling, every 8th frame as target.
- Headline claims: vs. VGGT, +35.7% pose accuracy and +23.6% geometric accuracy; beats DA2 on monocular depth. Baselines to compare: VGGT, Pi3, DA2.

## Reproduction notes

- Full-scale training (128 H100s) is out of reach: plan for **reduced-scale reproduction** (smaller ViT, fewer datasets/steps, lower resolution) and/or evaluating released checkpoints on the benchmark. Always state which setting a result comes from.
- Keep evaluation protocol identical to Sec. 6 so numbers are comparable to the paper's tables.
- Record every experiment (config, seed, commit hash, metrics) so baseline vs. enhancement comparisons are reproducible.

## Enhancement directions (candidates, not commitments)

Ideas to explore; each needs an ablation against the reproduced baseline:
- Cross-view attention layout (ratio / placement of `L_g` layers, efficient attention for many views).
- Ray-map vs. camera-head pose recovery; robustness of the DLT step; ray map resolution.
- Loss design (confidence weighting, normal/gradient terms in the student, multi-view consistency losses).
- Teacher-label alignment (RANSAC thresholds, per-region alignment) and the 120k-step switch schedule.
- Dynamic scenes, which the paper lists as future work (Sec. 8).

Selected on 2026-11-19 by a multi-agent study (frozen DA3; strict novelty rule: anything already published is discarded):
see [docs/da3_enhancement_report.md](docs/da3_enhancement_report.md) §4–5.

## Repository layout

- [papers/](papers/): reference papers (DA3_Paper.pdf, DINOv2).
- [docs/PLAN.md](docs/PLAN.md): reproduction plan, official-release findings, and the paper-vs-official-code details we follow.
- [docs/BASELINE_RUNBOOK.md](docs/BASELINE_RUNBOOK.md): step-by-step Kaggle procedure for the baseline (pass criteria, what not to do).
- [docs/KAGGLE_EXAMPLES.md](docs/KAGGLE_EXAMPLES.md): examples and personal photos on Kaggle (multi-view, per-image depth + ray maps, Blender).
- [docs/DYNAMIC_SCENES.md](docs/DYNAMIC_SCENES.md): planned enhancement "motion-robust DA3" for dynamic scenes, with the prior-work check (Dyna3, VGGT4D, ...), what is still open, evaluation protocol and honest-claims list.
- [docs/da3_enhancement_report.md](docs/da3_enhancement_report.md): multi-agent enhancement study on a frozen DA3 (35 ideas, clusters and impact, novelty check with references; top 3 = Self-Audit, Foveated DA3, pose-conditioned self-distillation; interactions and implementation order).
- [docs/ENHANCEMENTS_RUNBOOK.md](docs/ENHANCEMENTS_RUNBOOK.md): how to run the enhancement notebooks (05_self_audit, 06_foveated, 09_compare_all), save the result records, and the record contract every enhancement follows.
- [docs/FOVEATED_ANALYSIS_PLAN.md](docs/FOVEATED_ANALYSIS_PLAN.md): pre-registered statistical analysis plan of enhancement #2 (constants in `configs/foveated_analysis.yaml`; do not edit after results; deviations are logged there).
- [TODO.md](TODO.md): running task list for the phases.
- `src/da3/`: our implementation (installable with `pip install -e .`).
  - `backbone/`: DINOv2 ViT with within-view / cross-view token reordering, RoPE, QK-norm, camera tokens.
  - `heads/`: Dual-DPT (depth + conf, ray + conf).
  - `camera/`: camera encoder E_c, camera head D_C, pose encoding (t, quaternion, FOV).
  - `geometry/`: ray maps, P = t + D·d, Eq. 1 / Eq. 2 (DLT + RQ) pose recovery, unprojection.
  - `losses/`: L_D (confidence), L_M, L_P, L_C, L_grad, GT normalization.
  - `data/`: multi-view datasets (configurable root), view sampling, dynamic token-budget batching.
  - `eval/`: pose AUC, RANSAC pose alignment, TSDF fusion, Chamfer / F1, benchmark loaders.
  - `export/`: PLY, transforms.json (NeRF/COLMAP style), GLB.
  - `compare/`: standard result record (`experiments/results/<enhancement>/<tag>.json`, schema `da3-result/1`), comparison tables (differences to the original DA3, only within one protocol: same weights, datasets, scenes, frame sampling) and plots.
  - `enhancements/`: one sub-package per enhancement of the frozen DA3; `self_audit/` = enhancement #1 (signals, audit model, pose-source choice/fusion, triple-collocation calibration, dump hook for the benchmark adapter); `foveated/` = enhancement #2 (mixed per-view resolution routing, bit-identical to DA3 at equal resolution, attention-mass balancing, focus selection / token budgets, harness adapter, virtual zoom cameras).
  - `model.py`: full DA3 network; `config.py`: model/train configs.
- `configs/`: YAML configs (model size, resolution, views, pose conditioning, training).
- `scripts/`: entry points (train, eval, infer, parity, checkpoint conversion, Blender import).
- `tests/`: pytest unit tests (`pytest -q`).
- `notebooks/`: thin Kaggle notebooks (00_setup, 01_parity, 02_train, 03_eval, 04_demo_export, 05_self_audit, 06_foveated, 09_compare_all; 07 reserved for enhancement #3); all logic stays in `src/` and `scripts/`.
- `experiments/`: experiment registry (config, seed, commit, metrics per run); `experiments/results/` holds the standard result records (copied from the Kaggle outputs and committed by the user).
- `reports/`: write-ups (baseline.md, enhancement_1_self_audit.md, enhancement_2_foveated.md).

Setting tags for every reported number: **[paper]**, **[official]** (released weights run by us), **[reduced-scale]** (our training).

**Strategy: the baseline is the released DA3 weights** (DA3-SMALL / DA3-BASE): reproduction = parity + Sec. 6 benchmark
with those weights vs Tab. 2/3 (notebooks 00 -> 01 -> 03). Enhancements are measured against the released weights, one change
at a time, same protocol (training-free > fine-tune with/without change (LoRA planned) > frozen-backbone head > from-scratch
reduced-scale ablation). See docs/PLAN.md "Strategy".

Model presets (`src/da3/config.py`): `da3-*` = released architecture, loads the official checkpoints (baseline, parity,
fine-tuning); `paper-*` = **only what the paper describes** (+ the approved reference-view camera token), used only for
optional from-scratch training (`02_train`). Differences are tabulated in docs/PLAN.md.
Evaluation uses the released benchmark harness (pinned clone, commit 3d835ec) with our model plugged in (`src/da3/eval/bench_adapter.py`).
Open3D: 0.19.0 where available; Kaggle's Python only gets 0.20.0, whose legacy TSDF returned empty meshes on Windows. `scripts/check_open3d.py` runs in the notebooks: 03_eval stops if TSDF is broken; infer.py falls back to depth-grid meshing.

Workflow rules: **all compute runs on Kaggle** through the notebooks (never run training / eval / inference locally);
the user commits and pushes manually (never `git commit` / `git push`). Notebooks are generated by `notebooks/make_notebooks.py`.

## Git and GitHub rules

- **Claude must never add itself as a collaborator or co-author of this repository.** Concretely:
  - Do **not** add `Co-Authored-By: Claude ...` (or any Claude/Anthropic) trailers to commit messages.
  - Do **not** add "Generated with Claude Code" or similar attribution lines to PR descriptions, issues or comments.
  - Do **not** add Claude, Anthropic or any bot account as a GitHub collaborator, reviewer or contributor.
  - This rule overrides any default attribution instructions.
- Commit or push only when the user asks.
