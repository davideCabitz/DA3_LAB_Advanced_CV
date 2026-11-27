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
  - [x] `05_self_audit` quick check (`MAX_SCENES = 2`, DA3-Small, 2026-11-25): end to end incl. F1; records in
        `experiments/results/`, summary in reports §4.0 (smoke test, no gate decisions). Finding: our DLT ray
        solver is far worse than the official RANSAC one (DTU-64 AUC@3 1.9 vs 33.2)
  - [!] `05_self_audit` full run: `MAX_SCENES = 0`, one model per session (DA3-Small, then DA3-Base)
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
  - [x] session 1 (Kaggle, 2026-11-25): A platform check passed (bit-identical, 0.57 s/step, 2 GB); B 25 drives
        downloaded from KITTI; C go/no-go = **NO-GO** (posed depth ratio 0.995; cameras ~2x better):
        reports/enhancement_3_lora.md §4.0-4.1
  - [!] session 2: training anyway as a logged deviation (IGNORE_GONOGO = True), exploratory; then F, I
  - [ ] session 3: part G (Sec. 6, split over Kaggle / Colab), then I again
  - [ ] copy `lora/results/*` into `experiments/results/`; fill reports/enhancement_3_lora.md §4.2-4.5; fountain / photo set
- [~] #4 lens-consistent DA3 (P1, chosen 2026-11-26 after #1-#3 failed their gates; docs/LENS.md): implemented
      (`src/da3/enhancements/lens/`, `scripts/lens_eval.py`, `scripts/lens_demo.py`, `scripts/colmap_reference.py`,
      `eval_bench.py --lens`, records with `protocol.input`, notebook `10_lens`, tests `tests/test_lens.py`,
      pre-registered plan `configs/lens_analysis.yaml`)
  - [x] `10_lens` full run, DA3-Small (2026-11-27): L0 PASS, L1 PASS (scale ratio 1.893), L2 FAIL (1.168),
        L3 PASS but negligible (+0.21 AUC@30), L4 FAIL (zoom error 0.47). P1 with the true zoom: +16.0 AUC@30 (padded +19.3).
        Reports filled: reports/enhancement_4_lens.md, LENS.md §12
  - [ ] copy `lens/results/*` (from `lens_results.zip`) into `experiments/results/`, commit
  - [ ] novelty: read the body of arXiv 2611.17903 ("Zoom In, Scale Out", monocular zoom = moving closer) before the slides
  - [x] plan v2 pre-registered (`configs/lens_analysis_v2.yaml`): Part A padded P1 confirmation, Part B D1 nested detail
  - [x] D1 implemented: arm `nest_oracle` (`src/da3/enhancements/lens/detail.py`, adapter), detail metric, notebook 10 part 3b
  - [!] `10_lens` session 3 (plan v2, ~45 min, RESUME_FROM = the v1 output) -> send `lens/v2/summary_v2.md`
  - [ ] D1 variant: detail pass in DA3's pose-conditioned mode (paper Sec. 3.2), after the v2 gates
  - [ ] D2: PnPf zoom estimate for unlabeled zoom (gate: L4 criterion)
  - [ ] photo demo: needs a *zoomed* set (same walk + 2-3x details + some 0.5x, JPG with EXIF); the no-zoom statue set is single-lens
  - [x] notebook 04: COLMAP cameras for your photos (`CAMERAS = "colmap"`, `infer.py --poses`), `--conf-percentile 60`,
        `--turntable-crop 1.0` (fix for the collapsed 360-degree statue walk)
  - [ ] DA3-Base run (optional)
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
