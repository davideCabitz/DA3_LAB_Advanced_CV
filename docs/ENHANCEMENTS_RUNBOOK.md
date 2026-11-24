# Enhancements runbook (Kaggle)

How to run the enhancement notebooks, save their results, and compare the original DA3 with all
enhancements. The enhancements, their novelty check and the plan are in
[da3_enhancement_report.md](da3_enhancement_report.md).

| Notebook | What it does | Accelerator |
|---|---|---|
| `05_self_audit` | Enhancement #1, DA3 Self-Audit: original DA3 vs DA3 + Self-Audit | GPU (one pass per model), the rest CPU |
| `06_…`, `07_…` | reserved for enhancements #2 (Foveated DA3) and #3 (pose-conditioned self-distillation) | – |
| `09_compare_all` | original DA3 vs every enhancement, from the saved result records | None (CPU) |

## 1. How results are stored (the contract every enhancement follows)

Every run, original or enhanced, writes one **standard result record** (JSON, schema
`da3-result/1`, see `src/da3/compare/schema.py`):

```
experiments/results/<enhancement>/<tag>.json
  enhancement  "baseline" | "self_audit" | "foveated" | "pose_distill"
  variant      e.g. "head" (API default), "ray_official", "audit_fuse"
  weights, setting ([official] / [reduced-scale]), protocol (datasets, scenes, max frames)
  sec6         paper-protocol means from the released harness (pose AUC, F1 / Chamfer)
  extra        enhancement-specific metric groups
  baseline_tag the original-DA3 run it is compared with
```

- `scripts/eval_bench.py` writes a record for every benchmark run; `--enhancement` and `--variant`
  label it.
- An enhancement adds its own metrics in `extra`.
- `da3.compare` only compares records with the **same weights, datasets, scenes and frame
  sampling**, so every difference shown is measured under one protocol.
- The reference row of each group is the original DA3 with its API default: enhancement
  `baseline`, variant `head`.

## 2. Run `05_self_audit`

1. **Push the repo** (the notebook clones `main`), then import `notebooks/05_self_audit.ipynb` on
   Kaggle.
2. **Settings:** Accelerator = GPU (T4 or P100), Internet = on.
3. **Configuration cell:** the default is a quick end-to-end check (`MAX_SCENES = 2`,
   `MODELS = ["da3-small"]`). For the real numbers set `MAX_SCENES = 0` and run **one model per
   session**: `["da3-small"]`, then `["da3-base"]`, so each run stays well inside Kaggle's 12-hour
   limit.
   - `DATASETS = ["hiroom", "7scenes", "dtu64"]`. ETH3D (15 GB) and ScanNet++ (11 GB) belong in a
     separate session, because of disk space.
   - `WORK` (harness exports + dumps) is in `/tmp`, so it does not count against the 20 GB output
     limit.
   - Only the three main pose variants (`ray_official`, `audit_select`, `audit_fuse`) are also
     scored on reconstruction (TSDF, slow). `ray_dlt` and the oracle get pose metrics only.
4. **Save Version → Save & Run All.** The notebook runs these steps:
   1. **Original DA3 on the paper protocol + dump (GPU).** The released harness with camera-head
      poses. `--dump-audit` also saves the official RANSAC ray poses, our DLT ray poses, a pooled
      ray map, and one extra E_c pass per scene. The exported depth and poses are unchanged.
   2. **Analysis (CPU).**
      - Self-Audit signals, per-view GT errors, and the audit model fitted leave-one-dataset-out.
      - Failure detection (A) and depth uncertainty (C).
      - Pose variants (B), written with the same depth and scored by the harness (`--eval-only`).
   3. **Results.** Summary tables (original vs Self-Audit), Sec. 6 difference plots,
      risk–coverage curves, depth-interval reliability, triple-collocation vs GT σ, audit
      coefficients, example flagged views.
   4. **Go/no-go gates** from the report (§4.2), printed as PASS / FAIL.
5. **Splitting GPU and CPU work (optional).** To save GPU quota:
   1. Run with `PHASE = "infer"` and `WORK = "/kaggle/working/eval_self_audit"` on GPU, so the dump
      is saved with the version.
   2. Then start a CPU session with `PHASE = "analyse"` and
      `RESUME_FROM = "/kaggle/input/<that version>"` (attach it as input first).
   3. The analysis downloads DA3-BENCH again, because the harness needs the GT meshes.

## 3. Save the results (needed for 09)

- Download `self_audit_results.zip` from the notebook output. It contains `self_audit/<tag>/`
  (summary, figures, CSV) and `self_audit/results/` (the records).
- Copy `self_audit/results/*` into the repo under **`experiments/results/`**, keeping the
  sub-folders (`baseline/`, `self_audit/`). Commit them; they are small JSON files.
- Keep the figures you want in the write-up (e.g. under `reports/figures/self_audit/`). Send me
  `summary.md` and the gate output, and I'll fill in
  [reports/enhancement_1_self_audit.md](../reports/enhancement_1_self_audit.md).

## 4. Run `09_compare_all`

- Accelerator = None, Internet = on.
- It reads the committed records in `experiments/results/`, plus any attached notebook output that
  contains a `results/` folder.
- It writes `comparison.md`: Sec. 6 tables with differences to the original DA3, then each
  enhancement's own metrics, plus difference plots.
- Locally (no compute, just JSON), the same report is
  `python scripts/compare_results.py --out reports/comparison.md`.

## 5. What not to do

- ❌ **Don't change `MAX_SCENES` / `DATASETS` between the baseline run and the analysis.** The
  analysis reads the baseline run's exports by tag. Records with different scene sets are never
  compared (by design).
- ❌ **Don't compare Self-Audit pose rows with the 03_eval baseline rows of another commit or
  solver.** Compare inside one protocol group; the reference row is in the same table.
- ❌ **Don't read the depth-uncertainty numbers of 7Scenes like HiRoom's.** 7Scenes GT depth is
  Kinect depth, not registered to the RGB camera, so per-pixel errors include that offset. HiRoom
  (synthetic) is the clean case.
- ❌ **Don't tune anything on the test data.** The audit is fitted leave-one-dataset-out. With a
  single dataset it falls back to leave-one-scene-out, and the record says so.
- ❌ **Don't delete `eval_self_audit/` before the analysis has run.** It holds the dumps.
