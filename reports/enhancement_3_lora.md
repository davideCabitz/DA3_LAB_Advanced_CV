# Enhancement 3: LoRA on DA3, pose-conditioned self-distillation (E3)

**Status (2026-11-26):**
- Platform check and go/no-go done on Kaggle (§4.0, §4.1). **The pre-registered gate said NO-GO:** the
  pose-conditioned DA3 (E3's teacher) gives **no better depth** than the unposed DA3 on KITTI, while its
  **cameras are about twice as accurate**. E3's novel claim (depth taught by the pose-conditioned pass) is
  therefore not supported.
- **Reproduced:** a fresh session on 2026-11-26 (new download and conversion of all 25 drives) gave the
  platform check and the go/no-go table **digit for digit**, with the same NO-GO (§4.0, §4.1).
- **No LoRA has been trained yet, so no LoRA result exists.** The exploratory training after the NO-GO
  (`IGNORE_GONOGO = True`, a logged deviation) was attempted on 2026-11-25, but all six runs stopped
  before step 1 because of a notebook bug (§4.2, fixed). The 2026-11-26 session used the default
  `IGNORE_GONOGO = False` and skipped training, as pre-registered.
- **Available for the comparison:** the reference arm of the Sec. 6 no-regression check, i.e. the released
  DA3-Small on HiRoom, 7Scenes and DTU-64 (§4.4).
- **Still open:** whether to run the exploratory training (§4.2–4.5 stay empty until then).
- Setting tags: **[paper]**, **[official]** (released weights run by us), **[official+LoRA]** (released
  weights + our adapter).

Plan and method: [docs/LORA.md](../docs/LORA.md). Statistics pre-registered before any run:
[docs/LORA_ANALYSIS_PLAN.md](../docs/LORA_ANALYSIS_PLAN.md) (plan v2, `configs/lora_analysis.yaml`).
Origin: rank 3 of the multi-agent study, [docs/da3_enhancement_report.md](../docs/da3_enhancement_report.md) §4.4.

## 1. Idea

In many domains camera poses are cheap (GNSS/INS on a car, robot kinematics) but dense depth is not.
DA3 is the only feed-forward 3D model that can take poses as input (camera encoder E_c). E3 uses this:
- **teacher:** the frozen released DA3, conditioned on the known poses;
- **student:** the same DA3, unposed, plus a LoRA (switchable; off = bit-identical to the release);
- the student learns the teacher's depth and the known poses, so at test time it needs no poses.

The idea is new only if the teacher's **depth** is better than the student's: otherwise E3 reduces to
ordinary pose-supervised fine-tuning, which is published. The go/no-go checks exactly this before training
(report §4.4: GO only if conditioned depth is ≥ 5% better).

## 2. What was implemented

| Part | Code | What it does |
|---|---|---|
| LoRA | `src/da3/lora.py` | A, B per linear layer (`attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2`) of backbone blocks 4–11, inserted by module name (E_c, heads, blocks 0–3 untouched); B = 0 at init; off = exact base path; merge / exact unmerge; adapter-only files tied to the base weights' SHA-256 |
| Training | `src/da3/lora_train.py`, `scripts/train_lora.py` | arms `e3` (teacher depth + known poses), `pose_only` (known poses only), `own_output` (frozen DA3's own outputs: no new information); gauge: reference view 0 for teacher and student, teacher depth to pose units with the released API's Umeyama scale, `normalize_gt`, detached per-clip scale; anchor to the frozen outputs; cached teacher; exact resume; snapshots |
| KITTI | `src/da3/data/kitti.py`, `scripts/download_kitti.py`, `scripts/prepare_kitti.py` | OXTS → camera poses, LiDAR → depth (evaluation only), drive split with the Eigen-test leakage guard; only the split's drives are downloaded |
| Evaluation | `src/da3/eval/clip_metrics.py`, `scripts/kitti_eval.py` | clip-scale AbsRel / δ1 vs LiDAR, ATE / RPE, pose AUC; all arms on identical clips |
| Statistics | `src/da3/lora_stats.py`, `scripts/lora_analysis.py` | pre-registered tests: drive / scene units, sign-flip permutation, cluster bootstrap, Holm, non-inferiority (IUT) |
| Same photos | `scripts/compare_reconstructions.py` | released vs adapters on one photo set: differences, side-by-side video, aligned PLYs |
| Notebooks | `07_lora`, `07_lora_colab` | parts A–I, resumable (Colab: Drive checkpoints) |

Unit tests: `tests/test_lora*.py`, `test_kitti.py`, `test_download_kitti.py`, `test_compare_reconstructions.py`
(incl. an end-to-end run of the scripts on a tiny random model).

## 3. Setup

- **Model:** DA3-Small, released weights (Apache-2.0). LoRA r = 16, α = 16, blocks 4–11, 786,432 trainable
  parameters (≈ 2.3% of the model).
- **Data:** KITTI raw (CC BY-NC-SA 3.0), official download, synced + rectified, camera 2, OXTS, Velodyne.
  Split drawn by the pre-registered rule (`configs/lora_analysis.yaml`, seed 0) from the Eigen test drives
  (test) and Eigen train drives (train / val), zips ≤ 4 GB: **12 test, 3 validation, 10 training drives**
  (25 of 52 eligible). Validation drives (go/no-go): 2011_09_26_drive_0039, _0057, _0079 (from the download
  log; see `lora/kitti/split.json`). No KITTI image is in DA3's training set (Tab. 1 has only the synthetic
  vKITTI2) or in any Sec. 6 test set.
- **Clips:** 8 frames, every 3rd frame, ≤ 10 clips per drive, 1.8:1 centre crop at 504×280; cameras from the
  camera head D_C; depth vs LiDAR ≤ 80 m with one median scale per clip.

## 4. Results

### 4.0 Platform check [official] (Kaggle T4; commit `7e12db1`, repeated in a fresh session at `7779f8a`)

- LoRA switched off and LoRA at init (B = 0): outputs **bit-identical** to the released DA3 on the GPU
  (depth, confidence, ray map, camera encoding); the released model itself is deterministic (repeat run
  identical). Same result in both sessions; the 2026-11-26 session also passed the 32 LoRA / KITTI unit tests.
- Measured cost of one training step (forward + backward into A / B, gradient checkpointing, fp16):

| Views × resolution | Peak memory | Time / step (`7e12db1`) | Time / step (`7779f8a`) |
|---|---|---|---|
| 8 × 280×504 (the training setting) | 2.00 GB | 0.57 s | 0.59 s |
| 16 × 280×504 | 3.69 GB | 1.29 s | 1.34 s |
| 8 × 504×504 | 3.47 GB | 1.13 s | 1.20 s |

Peak memory is identical in both sessions; time varies by 2–6% between T4 sessions. The estimates in
docs/LORA.md (4–6 GB, ~1.5 s) were pessimistic. At ≈ 0.6 s/step, one 2000-step run takes ≈ 20 min of
training (plus the shared cache).

### 4.1 Go/no-go [official]: does the pose-conditioned DA3 give better depth?

3 validation drives (24 clips: 10 + 10 + 4, drive 0079 has only 100 frames); released DA3 unposed vs
conditioned on the OXTS poses, identical clips.
Ratios = posed / unposed, ratio of drive means, 95% drive-bootstrap CI (descriptive: with 3 drives the
bootstrap CI is too narrow, so read the numbers, not the intervals).

| Metric | Unposed (released) | Posed (teacher) | Ratio [95% CI] |
|---|---|---|---|
| AbsRel, one scale per clip ↓ | 0.1239 | 0.1234 | **0.995** [0.974, 1.016] |
| δ1, one scale per clip ↑ | 0.891 | 0.884 | 0.993 [0.979, 1.004] |
| AbsRel, one scale per image ↓ | 0.1227 | 0.1214 | 0.990 [0.964, 1.018] |
| ATE (m) ↓ | 0.109 | 0.064 | **0.587** [0.446, 0.627] |
| RPE translation (m) ↓ | 0.143 | 0.076 | **0.533** [0.469, 0.646] |
| RPE rotation (°) ↓ | 0.376 | 0.184 | **0.489** [0.460, 0.660] |
| Pose AUC@3 ↑ | 13.3 | 45.2 | **3.39** [2.30, 7.80] |
| Pose AUC@30 ↑ | 63.5 | 69.1 | 1.09 [0.85, 1.17] |

**Decision: NO-GO** (rule: posed / unposed clip-scale AbsRel ≤ 0.95; observed 0.995).

**Reproduction (2026-11-26, commit `7779f8a`):** a fresh session that downloaded and converted the 25
drives again (same split, Eigen list SHA-256 `e226d4ca…`, no Eigen test drive in train / val) gave every
number of the table above identically, and the same NO-GO. The gate is deterministic, so this confirms that
the pipeline is reproducible end to end. It is not new statistical evidence: same drives, same clips.

What it means:
- **Depth:** conditioning on the true poses changes DA3's depth on KITTI by less than 1%, so the teacher
  has nothing to teach about depth. E3's novel term has no headroom here.
- **Cameras:** with the poses, DA3's own camera head is about **twice as accurate** (ATE −41%, RPE −47 to
  −51%, AUC@3 ×3.4). This also shows that DA3 really uses the pose input (we read D_C, not the input poses):
  the poses are absorbed into the camera estimate but not into the depth.
- This matches the paper's mixed evidence that conditioning does not always help geometry (Tab. 7: +2.9 to
  +8 F1 on three datasets, −12.4 on 7Scenes [paper]).
- Consequence (report §4.4): any gain of a LoRA trained here can only come from the poses, i.e. ordinary
  pose-supervised fine-tuning, which is not new. The novelty claim of E3 is **rejected by its own gate**.

### 4.2 Training (exploratory, logged deviation)

**No run has completed; no adapter exists.**
- **2026-11-25, commit `7e12db1`, `IGNORE_GONOGO = True`:** the deviation was logged in
  `lora/deviations.txt`. All six launches (3 arms × 2 seeds) then stopped at argument parsing, before step 1
  (`train_lora.py: error: argument --arm: invalid choice: '{arm}'`).
  - Cause: the notebook line ended in `--time-budget-h {budget:.2f}`. IPython evaluates a `{...:.2f}` field
    on a `!` line as code, fails, and then passes the whole command unexpanded.
  - The failures were silent because a `!` command does not raise on a non-zero exit.
  - Fixed in commit `29871ce`: the value is formatted before the call, and the notebook now stops on a
    failed run.
  - The `deviations.txt` entry of that session therefore records a training that produced nothing.
- **2026-11-26, commit `7779f8a`, defaults (`IGNORE_GONOGO = False`):** training skipped, as pre-registered
  ("go/no-go = NO-GO: training skipped").

Planned when run: arms `e3`, `pose_only`, `own_output` × seeds 0, 1, 2000 steps each. Fill from
`lora/runs/*/train_summary.json` and the logs (final losses, s/step).

| Arm | Seed 0: final loss | Seed 1: final loss | Time / run |
|---|---|---|---|
| e3 | – | – | – |
| pose_only | – | – | – |
| own_output | – | – | – |

### 4.3 Held-out KITTI test drives [official] vs [official+LoRA]

_Pending: fill from `lora/kitti_test_da3-small.md` and `lora/analysis/summary.md`. Because of the NO-GO and
the deviation, every decision here is reported but read as **exploratory**._

| ID | Question | Ratio (LoRA / reference) | 95% CI | p | Decision |
|---|---|---|---|---|---|
| H1 | E3 depth (AbsRel, clip scale) vs released | – | – | – | – |
| S1 | E3 ATE vs released | – | – | – | – |
| S2 | E3 RPE translation vs released | – | – | – | – |
| S3 | E3 RPE rotation vs released | – | – | – | – |
| S4 | E3 vs pose-only (AbsRel): does the teacher's depth add anything? | – | – | – | – |
| S5 | E3 vs own-output (AbsRel): more than generic fine-tuning? | – | – | – | – |

Expected from §4.1 (stated before the training results): H1 null, S4 null; any gain in S1–S3.

### 4.4 No regression on the paper's benchmarks (Sec. 6) [official] vs [official+LoRA]

_Pending for the E3 adapters (part G): HiRoom, 7Scenes, DTU-64, released vs E3 adapters (2 seeds);
non-inferiority margins fixed in the plan (AUC −1 point, F1 −1 point, Chamfer +3%). ETH3D, ScanNet++ and
the monocular check are not evaluated (plan v2)._

**Reference arm done: released DA3-Small [official]** (2026-11-25, commit `7e12db1`, Kaggle T4; inference
in Python 3.13, scoring in the Python 3.12 + Open3D 0.19.0 side environment; record
`lora/results/baseline/lora_sec6_da3-small_released_hiroom-7scenes-dtu64.json`). Harness pinned at
`3d835ec`, up to 100 frames per scene, reference view "first", cameras from D_C. AUC and F1 in points.

| Dataset (scenes) | AUC@3 | AUC@30 | F1 (mean) | F1 (median) | Chamfer "overall" (mean) | Chamfer "overall" (median) |
|---|---|---|---|---|---|---|
| HiRoom (30) | 9.67 | 75.21 | 18.32 | 14.06 | 0.370 | 0.172 |
| 7Scenes (7) | 14.58 | 78.22 | 38.91 | 36.30 | 0.160 | 0.090 |
| DTU-64 (13) | 30.80 | 91.16 | – | – | – | – |

- **HiRoom Chamfer is dominated by one scene.** In `828785/cam_sampled_12` the accuracy term is 10.79
  (overall 5.47); without it the mean overall is 0.195. F1, which the plan uses (N3), is not affected by
  this. Chamfer is reported as a median next to the mean.
- **The harness scores DTU-64 for pose only.** The log says "(pose evaluation only)" for every DTU-64
  scene, so **N4 (DTU-64 Chamfer) cannot be computed as written in the plan.** See §6 and the deviations
  log of the plan; nothing has been changed yet.
- **Internal check:** our AUC re-computation agrees with the harness on every scene (max difference
  0.0002, i.e. 0.02 points). This is a check of the metric code within one run, not a repeat evaluation.
- **A/A repeat (plan `aa_checks`, descriptive).** The only other released DA3-Small evaluation of the
  same scenes is the 2-scene Self-Audit smoke test (`experiments/results/baseline/sa_da3-small_hiroom-7scenes-dtu64_limited2.json`,
  commit `e8676fd`). Reconstruction F1 on the shared scenes:

| Scene | Smoke test (`e8676fd`) | This run (`7e12db1`) | Difference (points) |
|---|---|---|---|
| HiRoom 828738/cam_sampled_08 | 5.111 | 5.114 | +0.003 |
| HiRoom 828744/cam_sampled_04 | 13.603 | 13.687 | +0.084 |
| 7Scenes chess | 36.322 | 36.296 | −0.026 |
| 7Scenes fire | 65.449 | 65.571 | +0.122 |

  On HiRoom, the scenes the plan names for the A/A check, the difference stays below the 0.1-point flag.
  7Scenes fire exceeds it slightly (0.12). This is evaluation non-determinism (TSDF / RANSAC): the model is
  deterministic (§4.0). It is about 8× smaller than the N3 margin (1 point). Per-scene pose AUC is not
  printed in the log, so the pose A/A comparison needs the record file and is still open.

| ID | Metric | Estimate | 95% CI | Decision |
|---|---|---|---|---|
| N1 | AUC@3 (pooled) | – | – | – |
| N2 | AUC@30 (pooled) | – | – | – |
| N3 | F1 (HiRoom, 7Scenes) | – | – | – |
| N4 | Chamfer (DTU-64) | not computable as planned (harness: DTU-64 pose only), decision open | | |
| N5, N6 | monocular δ1, AbsRel | not evaluated (plan v2) | | |

### 4.5 Same photos (Phase 5)

_Pending: a photo set of one static scene (released vs adapters vs training snapshots), part H._

## 5. Discussion

_After the training results. Points prepared from §4.1:_
- Why the poses help the cameras but not the depth: on KITTI's forward motion the scale-normalised depth of
  DA3 is driven by single-view cues and cross-view matching, which the camera token does not change much;
  the clip-scale metric also removes any global-scale benefit of metric poses by design.
- What a pose-supervised LoRA can and cannot claim (no novelty; a measured domain-adaptation result with a
  forgetting check and a switchable adapter).

## 6. Limitations and honest claims

- **Go/no-go on 3 drives** (24 clips): the depth effect is small and consistent across them, but 3 drives
  cannot exclude a small effect on other routes. The decision rule was fixed before the run and is not
  revisited. Its exact reproduction (§4.1) shows that the pipeline is deterministic. It does not add drives,
  so it does not narrow the uncertainty.
- **Deviation:** the training after a NO-GO was not part of the plan; it is logged and its results are
  exploratory. The first attempt (2026-11-25) trained nothing (§4.2), so the deviation entry of that
  session has no result attached. Plan v2 (fewer seeds, fewer Sec. 6 datasets, no monocular check) was
  fixed before any result, because of the deadline.
- **N4 not computable as planned:** the pinned harness evaluates DTU-64 for pose only, so the DTU-64
  Chamfer of N4 does not exist. Any replacement (e.g. DTU with reconstruction) must be fixed and logged
  **before** a LoRA result is seen. Otherwise N4 is reported as "not evaluated", and the no-regression
  claim is restricted to N1–N3.
- **A/A floor:** the repeat evaluation of the released model differs by up to 0.12 F1 points (7Scenes
  fire; ≤ 0.08 on HiRoom). Effects of this size are noise.
- **May claim:** the measured effect of DA3's pose conditioning on KITTI (cameras ~2× better, depth
  unchanged), which the paper does not report for this domain; a switchable, bit-exact LoRA platform for
  DA3 with pre-registered evaluation.
- **Must not claim:** E3's novelty (rejected by its gate); any LoRA gain as new (pose-supervised fine-tuning
  is published).

## 7. Reproduce

Notebook `07_lora` on Kaggle (GPU T4, Internet on), see
[docs/ENHANCEMENTS_RUNBOOK.md](../docs/ENHANCEMENTS_RUNBOOK.md) §3b: run 1 with defaults (parts A–C gave
the NO-GO); run 2 with `RESUME_FROM = <run 1 output>/lora` and `IGNORE_GONOGO = True`.
The settings cell is reset to its defaults (`IGNORE_GONOGO = False`, `"sec6": False`, `RESUME_FROM = ""`)
every time the notebook is regenerated, so set them again after importing a new version.
