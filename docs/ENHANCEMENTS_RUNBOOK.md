# Enhancements runbook (Kaggle)

How to run the enhancement notebooks, save their results, and compare the original DA3 with all
enhancements. The enhancements, their novelty check and the plan are in
[da3_enhancement_report.md](da3_enhancement_report.md).

| Notebook | What it does | Accelerator |
|---|---|---|
| `05_self_audit` | Enhancement #1, DA3 Self-Audit: original DA3 vs DA3 + Self-Audit | GPU (one pass per model), the rest CPU |
| `06_foveated` | Enhancement #2, Foveated DA3: mixed per-view resolutions in one pass (view level + region zoom) | GPU; region zoom in its own session |
| `07_…` | reserved for enhancement #3 (pose-conditioned self-distillation) | – |
| `09_compare_all` | original DA3 vs every enhancement, from the saved result records | None (CPU) |

## 1. How results are stored (the contract every enhancement follows)

Every run, original or enhanced, writes one **standard result record** (JSON, schema
`da3-result/1`, see `src/da3/compare/schema.py`):

```
experiments/results/<enhancement>/<tag>.json
  enhancement  "baseline" | "self_audit" | "foveated" | "pose_distill"
  variant      e.g. "head" (API default), "ray_official", "audit_fuse", "head_res336", "fov25_504-252"
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

## 3. Run `06_foveated`

Plan and novelty: [da3_enhancement_report.md](da3_enhancement_report.md) §4.3; write-up:
[reports/enhancement_2_foveated.md](../reports/enhancement_2_foveated.md). Five parts, chosen with
`RUN` in the configuration cell:

| Part | What | Time (T4) |
|---|---|---|
| 0 · routing check | equal resolutions must give outputs bit-identical to the original DA3; the notebook stops otherwise | ~5 min |
| 1 · probe (gates 1 and 3) | 3 × 16 views per scene, 8 focus at 504 px + 8 context at 252 / 280 px, vs uniform resolutions, the focus views alone and an A/A run | ~1 h |
| 2 · large-view protocol | 100–1000 views per pass on all 7 7Scenes sequences: time (repeated), memory, out-of-memory, focus-view accuracy | ~1.5 h (DA3-Small) |
| 3 · Sec. 6 protocol | released harness with uniform (504 / 336 / 252) and foveated runs, plus the per-view analysis with paired CIs | 3–6 h with TSDF; ~1 h with `SEC6_MODES = ["pose"]` |
| 4 · region zoom (gate 2) | virtual zoom cameras on the full-resolution photos of 11 ETH3D scenes; oracle, random and texture regions | ~1.5 h + 15 GB download |

**Statistics are pre-registered**
([FOVEATED_ANALYSIS_PLAN.md](FOVEATED_ANALYSIS_PLAN.md), constants in
`configs/foveated_analysis.yaml`):
- every gate is PASS / FAIL / INCONCLUSIVE, read off a 95% scene-bootstrap CI, with the margins at
  which the decision would change;
- the margin is the report's 3%, raised only by the noise floor of a treatment-blind A/A run and
  capped at 6%.

**Do not edit the plan after the runs.** The analysis compares the plan's SHA-256 with the one each
part was run under, and prints a warning if they differ.

1. **Push the repo**, import `notebooks/06_foveated.ipynb` on Kaggle. Accelerator = GPU (T4 or
   P100), Internet = on.
2. **Quick end-to-end check first:** `MAX_SCENES = 2`, `SCALING_VIEWS = [100, 250]`,
   `SEC6_MODES = ["pose"]`. Results are tagged `_limited2` and are not reportable.
3. **Session 1 (view level):** the defaults (`RUN` with `zoom: False`, `MAX_SCENES = 0`).
   **Save Version → Save & Run All.**
   - If the session stops early, attach its output and set
     `RESUME_FROM = ["/kaggle/input/<version>/foveated"]`: finished parts and Sec. 6 runs are skipped.
   - Every Sec. 6 run goes through `scripts/eval_bench.py` (`--process-res` + `--meter` for uniform
     rows, `--foveated` for mixed rows) and writes a standard record; the reference row is the
     original DA3 (504 px, camera-head poses). Its exports (~2 GB) are deleted after the per-view
     analysis unless `KEEP_EXPORTS = True`.
4. **Session 2 (region zoom):** `RUN = {"parity": False, "probe": False, "scaling": False,
   "sec6": False, "zoom": True}` and `RESUME_FROM = [<session 1 output>]`. It downloads ETH3D
   (15 GB) to `/tmp` and extracts only the 5 scenes' photos and the GT depth of the source views.
   The final analysis cell then reports every part.
5. **Outputs** in `/kaggle/working/foveated/` (zipped as `foveated_results.zip`):
   - `summary.md` and `gates.json`;
   - figures: `gates_*.png` (decision figure), `probe_*.png`, `sec6_*.png`, `scaling_*.png`,
     `gates_zoom_*.png`, `zoom_*/zoom_examples.png`;
   - one JSON per part;
   - `results/` with the records: `baseline/` for the uniform rows, `foveated/` for the rest.

Gates (printed and in `gates.json`):
- G0 routing parity;
- G1 uniform-resolution curve (G1a headroom, G1b context informative);
- G3 cross-scale probe: primary, then secondary, robustness and the rescue, each labelled;
- G2 region zoom: oracle (primary), random and texture (secondary).

## 4. Save the results (needed for 09)

- Download the zip from the notebook output: `self_audit_results.zip` (05) or
  `foveated_results.zip` (06). They contain the summaries, figures and a `results/` folder with
  the records.
- Copy `<…>/results/*` into the repo under **`experiments/results/`**, keeping the sub-folders
  (`baseline/`, `self_audit/`, `foveated/`). Commit them; they are small JSON files.
- Keep the figures you want in the write-up (e.g. under `reports/figures/self_audit/` or
  `reports/figures/foveated/`). Send me `summary.md` and the gate output, and I'll fill in
  [reports/enhancement_1_self_audit.md](../reports/enhancement_1_self_audit.md) /
  [reports/enhancement_2_foveated.md](../reports/enhancement_2_foveated.md).

## 5. Run `09_compare_all`

- Accelerator = None, Internet = on.
- It reads the committed records in `experiments/results/`, plus any attached notebook output that
  contains a `results/` folder.
- It writes `comparison.md`: Sec. 6 tables with differences to the original DA3, then each
  enhancement's own metrics, plus difference plots.
- Locally (no compute, just JSON), the same report is
  `python scripts/compare_results.py --out reports/comparison.md`.

## 6. What not to do

- ❌ **Don't change `MAX_SCENES` / `DATASETS` between the baseline run and the analysis** (or between
  06 sessions combined with `RESUME_FROM`). The analysis reads runs by tag, and the tags contain
  them. Records with different scene sets are never compared (by design).
- ❌ **Don't put the 06 probe, large-view or region-zoom numbers into Sec. 6 tables.** They come from
  new protocols (16 views per scene, 100–1000 views, oracle regions). Only the 06 Sec. 6 runs are
  comparable with the paper protocol, and only within their own group (same datasets).
- ❌ **Don't set `KEEP_EXPORTS = True` for all 06 Sec. 6 runs**: each run keeps ~2 GB in `/tmp`.
- ❌ **Don't change `configs/foveated_analysis.yaml` after seeing 06 results** (margins, primary
  configurations, selectors). A different plan needs a new version, the deviation logged in
  [FOVEATED_ANALYSIS_PLAN.md](FOVEATED_ANALYSIS_PLAN.md), and both analyses reported.
- ❌ **Don't report an INCONCLUSIVE gate as PASS or FAIL**, and don't drop the secondary,
  robustness or rescue rows from the write-up.
- ❌ **Don't compare Self-Audit pose rows with the 03_eval baseline rows of another commit or
  solver.** Compare inside one protocol group; the reference row is in the same table.
- ❌ **Don't read the depth-uncertainty numbers of 7Scenes like HiRoom's.** 7Scenes GT depth is
  Kinect depth, not registered to the RGB camera, so per-pixel errors include that offset. HiRoom
  (synthetic) is the clean case.
- ❌ **Don't tune anything on the test data.** The audit is fitted leave-one-dataset-out. With a
  single dataset it falls back to leave-one-scene-out, and the record says so.
- ❌ **Don't delete `eval_self_audit/` before the analysis has run.** It holds the dumps.
