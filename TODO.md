# TODO

Legend: [x] done · [~] in progress · [ ] open · [!] waiting for the user (Kaggle run)
Rules: all compute on Kaggle (docs/BASELINE_RUNBOOK.md); the user commits and pushes.

## Phase 0 – Recon
- [x] Paper read; official code / checkpoints / licenses / benchmark data located (docs/PLAN.md)
- [x] Decision: own implementation; official repo = reference, parity oracle, benchmark harness
- [!] Kaggle limits: run `00_setup`, send env_report.json

## Phase 1 – Implementation
- [x] backbone, Dual-DPT, E_c, D_C, geometry (Eq. 1, Eq. 2), losses, data, configs
- [x] `paper-*` presets (baseline) vs `da3-*` presets (released, loading only)
- [x] camera-token decision: reference + shared tokens in the baseline
- [x] 25 unit tests (passed on local CPU); [!] rerun on Kaggle (`00_setup`)

## Phase 2 – Parity
- [x] key mapping + parity DA3-SMALL / DA3-BASE (local CPU, ≤ 1.9e-5 depth)
- [!] parity on Kaggle GPU (`01_parity`)

## Phase 3 – Reduced-scale training
- [x] converters (TartanAir w/ App. A cleaning, vKITTI2), reprojection check, token-budget sampler
- [x] train.py: fp16, grad checkpointing, exact resume, `--time-budget-h`, overfit mode
- [!] GPU sanity (overfit + resume) in `00_setup`
- [!] check the DASVO TartanAir layout (first cells of `02_train`), then training sessions

## Phase 4 – Evaluation
- [x] harness adapter, our metrics (= official to round-off), make_tables.py
- [!] `03_eval`: HiRoom / 7Scenes / DTU-64 first, then DTU / ScanNet++ / ETH3D
- [ ] VGGT / Pi3 / DA2 baselines (not started)

## Phase 5 – Demo + Blender
- [x] infer.py (multi-view and `--per-image` depth + ray maps), exports, blender_import.py
- [x] headless Blender check (local CPU, ≤ 0.76 px)
- [!] `04_demo_export` on Kaggle with your photos (docs/KAGGLE_EXAMPLES.md)

## Phase 6 – Report
- [~] reports/baseline.md: fill in the Kaggle results
- [x] docs/BASELINE_RUNBOOK.md, docs/KAGGLE_EXAMPLES.md
