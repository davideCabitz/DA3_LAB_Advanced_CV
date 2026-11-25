# TODO

Legend: [x] done · [~] in progress · [ ] open · [!] waiting for the user (Kaggle run)
Rules: all compute on Kaggle (docs/BASELINE_RUNBOOK.md); the user commits and pushes.
Strategy: **baseline = released DA3 weights**; enhancements are measured against them (docs/PLAN.md).

## Main track: reproduce the published results with the released weights
- [x] Paper read; official code / checkpoints / licenses / benchmark data located
- [x] Our implementation of the released architecture (`da3-*` presets) + key mapping
- [x] Parity DA3-SMALL / DA3-BASE vs official code (local CPU: depth ≤ 1.9e-5)
- [x] Benchmark adapter on the released harness; our metrics = official to round-off
- [!] Kaggle account (phone-verified), repo pushed, notebooks imported (runbook Step 0)
- [!] `00_setup`: Kaggle limits, unit tests (87 on 2026-11-22), GPU sanity checks
- [!] `01_parity` on GPU
- [!] `03_eval` timing run (HiRoom, 2 scenes)
- [!] `03_eval` main run: HiRoom, 7Scenes, DTU-64 (all scenes), DA3-SMALL + DA3-BASE
- [ ] `03_eval` DTU, ScanNet++, ETH3D (one per session, disk permitting)
- [ ] reports/baseline.md: our numbers vs paper Tab. 2 / Tab. 3, discussion of gaps
- [ ] optional: DA3-LARGE-1.1 (parity first; CC BY-NC; not the paper's checkpoint)

## Next: enhancements (against the released weights, one change at a time)
Top 3 from the multi-agent study (docs/da3_enhancement_report.md §4–5). Run the gates before writing method code.
- [ ] Gates (step 1, ~4–6 GPU-h): dump DA3 outputs (with the baseline run), E_c noise probe (read D_C / ray outputs,
      not the returned input poses), uniform-resolution curve, region-zoom oracle on ETH3D, KITTI unposed-vs-posed check
- [x] Comparison layer: standard result records (`src/da3/compare/`, `experiments/results/`), notebook `09_compare_all`,
      `scripts/compare_results.py`; `eval_bench.py` writes a record for every run (docs/ENHANCEMENTS_RUNBOOK.md)
- [~] #1 DA3 Self-Audit (training-free): implemented (`src/da3/enhancements/self_audit/`, `scripts/self_audit_eval.py`,
      notebook `05_self_audit`, tests `tests/test_self_audit.py` + `tests/test_compare.py`): failure detector, per-view
      head/ray pose choice + fusion, triple-collocation σ, E_c residual; the gate dump is part of this run
  - [!] `00_setup` on Kaggle: the new unit tests must pass (pytest runs all tests)
  - [!] `05_self_audit` with `MAX_SCENES = 2` (end-to-end check), then `MAX_SCENES = 0`; share `summary.md` + gates
  - [ ] copy `self_audit/results/*` into `experiments/results/` and commit; fill reports/enhancement_1_self_audit.md
- [~] #2 Foveated DA3 (training-free): implemented (`src/da3/enhancements/foveated/`, `scripts/foveated_eval.py`,
      `eval_bench.py --foveated / --process-res / --meter`, notebook `06_foveated`, tests `tests/test_foveated.py`):
      mixed-resolution routing (bit-identical at equal resolution), attention-mass balancing, probe (gates 1 + 3),
      large-view protocol, Sec. 6 runs with per-view analysis, region zoom on ETH3D (gate 2); pre-registered
      statistics (docs/FOVEATED_ANALYSIS_PLAN.md: scene-bootstrap CIs, three-way gates, A/A noise floor)
  - [!] `06_foveated` quick check (`MAX_SCENES = 2`, pose only), then session 1 (view level) and session 2 (zoom);
        share `summary.md` + `gates.json`
  - [ ] copy `foveated/results/*` into `experiments/results/` and commit; fill reports/enhancement_2_foveated.md
- [ ] #3 Pose-conditioned self-distillation (LoRA, 1-2 sessions): needs a few KITTI raw drives (user download);
      controls: pose-only LoRA, own-output LoRA, Free Geometry
- [~] enhancement: motion-robust DA3 (docs/DYNAMIC_SCENES.md §4): C2 masked ray-map poses, C3 key-masking study,
      stretch C4 motion-aware LoRA. C1 (pose-conditioned 2nd pass) is now only a cited component: the mechanism is
      published (ScaRF-SLAM, DAVIO, VIDAR). Prior-work check 2026-11-19 (Dyna3 already does masks + 4D on DA3)
- [~] #3 LoRA (E3), plan v2 (4-day scope, logged in docs/LORA_ANALYSIS_PLAN.md before any result):
      `src/da3/lora.py`, `lora_train.py`, `lora_stats.py`, `data/kitti.py`; scripts `prepare_kitti.py`,
      `lora_check.py`, `kitti_eval.py`, `train_lora.py`, `lora_analysis.py`, `compare_reconstructions.py`;
      `--adapter` in `eval_bench.py` / `infer.py`; notebooks `07_lora` + `07_lora_colab` (parts A-I); tests
  - [!] session 1 (Kaggle, KITTI raw mirror attached): parts A-F, H, I
  - [!] session 2: part G (Sec. 6, split over Kaggle / Colab), then I again
  - [ ] copy `lora/results/*` into `experiments/results/`; reports/enhancement_3_lora.md; Blender video
- [ ] "enhancement experiment" template (config + eval command)
- [ ] VGGT / Pi3 / DA2 comparisons (not started)

## Optional: from-scratch reduced-scale training (`paper-*`, for architectural ablations)
- [x] `paper-*` presets (paper text + reference-view camera token), losses, data converters, train.py
      (fp16, exact resume, `--time-budget-h`), overfit sanity on CPU
- [ ] `02_train` on Kaggle: check the DASVO TartanAir layout, then training sessions

## Demo
- [x] infer.py (multi-view and `--per-image` depth + ray maps), exports, blender_import.py
- [x] headless Blender check (local CPU, ≤ 0.76 px)
- [ ] `04_demo_export` on Kaggle with your photos (docs/KAGGLE_EXAMPLES.md)
