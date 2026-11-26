# Enhancement 1: DA3 Self-Audit, original DA3 vs DA3 + Self-Audit

**Status (2026-11-25): smoke test done, full run pending.**
- Method, setup and claims below are final.
- A limited end-to-end run on Kaggle (`MAX_SCENES = 2`, DA3-Small) completed: every step ran, including
  the reconstruction scores. Its numbers are in §4.0, labelled as a smoke test: 2 scenes per dataset
  are too few for any conclusion, so the gates are **not** decided from it.
- The result sections §4.1–4.4 are filled from the full run (`MAX_SCENES = 0`, DA3-Small then
  DA3-Base), setting **[official]**. Until then they hold placeholders (–).

Origin of the idea: rank 1 of the multi-agent study,
[docs/da3_enhancement_report.md](../docs/da3_enhancement_report.md) §4.2 (merges ideas A2, C1, F3
and F2). It passed the novelty check: the signals and the GT-free calibration below were not
found in the literature (§3.4 there).

## 1. Idea in one paragraph

DA3 estimates every camera twice: once with the camera head D_C and once through the ray map. The
ray map has six numbers per pixel, far more than a pinhole camera needs. DA3 can also take its own
cameras back in through the camera encoder E_c. VGGT and π³ have neither a second camera decoder
nor a pose input. The Self-Audit turns this redundancy into three things the original DA3 does not
provide, without retraining anything:
1. a per-view reliability score;
2. a per-view choice between head and ray poses;
3. a depth uncertainty calibrated per scene without ground truth.

## 2. What was implemented

| Part | Code | What it does |
|---|---|---|
| Signals | `src/da3/enhancements/self_audit/signals.py` | Per view, all oriented so larger = less reliable: (a) head-vs-ray gap in rotation, centre and focal, after a Sim(3) alignment of the ray poses; (b) spread of the ray origins (Eq. 1 expects one centre); (c) angular residual of the best pinhole fit to the ray field (Eq. 2); (d) DA3's own depth / ray confidence; (e) cross-view depth reprojection disagreement (occlusion-aware); (f) **E_c fixed-point residual**: how much depth, camera and rays move when DA3 is re-run conditioned on its own cameras (one extra pass, global rotation removed) |
| Audit model | `models.py` | Ridge regression of log(1 + per-view pose error) on the signals. Fitted **leave-one-dataset-out**: it is never fitted on the dataset it is tested on |
| Pose source | `pose_sources.py` | Ray poses Sim(3)-aligned onto the head poses. Per view: **choice** of the source predicted to be more accurate, or **inverse-variance fusion** (quaternion average, centre, intrinsics). Oracle choice (uses GT) as upper bound |
| GT-free calibration | `triple_collocation.py` | σ(c) = a·c^(−b) for the log-depth error, per scene, by triple collocation on points seen by three views; gross outliers (occlusions) trimmed per triplet |
| Dump | `dump.py` + `scripts/eval_bench.py --dump-audit` | During the GPU benchmark run: official RANSAC ray poses (pinned official routine), our DLT ray poses, pooled ray map, E_c re-pass. The exported benchmark results are unchanged; a unit test checks they are bit-identical |
| Analysis | `scripts/self_audit_eval.py` | Builds the per-view table, fits the models, computes A/B/C and the gates, replays the pose variants through the released harness, writes records, figures, CSV and a summary |
| Comparison | `src/da3/compare/`, notebook `09_compare_all` | Standard result records shared by all enhancements; tables with differences to the original DA3 under one protocol |

## 3. Experimental setup

- **Weights:** DA3-Small and DA3-Base, released (Apache-2.0); network unchanged.
- **Data and protocol:**
  - DA3-BENCH HiRoom, 7Scenes and DTU-64 (DTU-64 counts for pose only, as in the paper).
  - Released harness at commit 3d835ec: ≤ 100 frames per scene (seed 42), evo Umeyama + RANSAC,
    TSDF, F1 / Chamfer.
  - Reference view "first", as in the 03_eval baseline.
- **Scope compared with the paper's Sec. 6:** same metrics (AUC@3, AUC@30, F1) and the same
  harness, on 3 of the paper's 5 datasets. Not run:
  - **ETH3D and ScanNet++**: compute budget, see §6;
  - **reconstruction with GT poses** ("w/ pose" column of Tab. 3): the Self-Audit only changes the
    predicted cameras, so with GT cameras the result equals the original DA3 by construction;
  - **full DTU (Chamfer in mm)**: DTU-64 is scored on pose only;
  - **novel view synthesis and monocular depth**: depth and Gaussians are unchanged.
- **Per-view GT error:** for view i, the median over the other views j of max(RRA_ij, RTA_ij).
  This is the paper's pairwise definition behind pose AUC. Failure = rotation error > 15° (gross)
  or > 5°.

The three comparisons, each against what the original DA3 offers:

| | Original DA3 | DA3 + Self-Audit | Metrics |
|---|---|---|---|
| **A. Failure detection** | depth confidence (−mean log c per view) | audit score | AUROC (> 15°, > 5°), Spearman, AURC (risk–coverage, error clipped at 30°) |
| **B. Pose source** | camera head (API default); ray map with the official RANSAC (official option); ray map with our DLT | per-view choice; per-view fusion; oracle choice (upper bound) | Sec. 6 AUC@3 / AUC@30, F1 without poses: depth identical in all rows, only the cameras differ |
| **C. Depth uncertainty** (HiRoom, 7Scenes) | confidence calibrated with GT on the *other* datasets (σ = a·c^(−b)) | per-scene triple collocation, no GT | empirical coverage of 50% / 90% Gaussian intervals and its error, NLL; upper bound = per-scene GT fit; AUSE of the confidence ranking (same for all) |

**Go/no-go gates** (report §4.2):
- **G1:** Spearman(head–ray gap or pinhole residual, rotation error) ≥ 0.3 and AUROC(> 15°) ≥ 0.7
  on ≥ 2 datasets.
- **G2:** oracle − better single source ≥ 2 AUC@3, or fusion ≥ +0.5 AUC@3, on most datasets.
- **G3:** triple-collocation σ / true σ within 0.7–1.4 in ≥ 75% of the confidence bins.
- **G4:** the E_c residual raises the audit AUROC by ≥ 0.03.

## 4. Results [official]

### 4.0 Smoke test (limited run, not reportable)
Kaggle, notebook `05_self_audit`, 2026-11-25, commit `e8676fd`; DA3-Small, 2 scenes each of HiRoom,
7Scenes, DTU-64 (≤ 100 frames); tag `sa_da3-small_hiroom-7scenes-dtu64_limited2`. Records in
`experiments/results/{baseline,self_audit}/`, summary and figures in
[figures/self_audit/limited2/](figures/self_audit/limited2/summary.md).

**What it shows about the pipeline (this is its purpose):**
- the GPU run, the dump, the leave-one-dataset-out audit, the harness replays of the pose variants
  and the TSDF / F1 scoring all ran end to end (F1 present for HiRoom and 7Scenes; DTU-64 counts for
  pose only, as planned);
- the numbers are in the range of the paper for DA3-Small [paper, Tab. 2: HiRoom 9.5 / 75.2,
  DTU 30.6 / 91.2, 7Scenes 14.0 / 78.7 AUC@3 / AUC@30]: ours on 2 scenes are HiRoom 3.9 / 70.3,
  DTU-64 40.7 / 93.1, 7Scenes 29.3 / 88.3. With 2 scenes the gaps are within scene-to-scene variation;
  DTU-64 is a subset of DTU. The full run decides.

**Sec. 6, pose source (only the cameras differ; depth identical):**

| Variant | 7Scenes AUC@3 / @30 / F1 | DTU-64 AUC@3 / @30 | HiRoom AUC@3 / @30 / F1 |
|---|---|---|---|
| camera head (original, API default) | 29.3 / 88.3 / 50.9 | 40.7 / 93.1 | 3.9 / 70.3 / 9.4 |
| ray map, official RANSAC (original option) | 32.4 / 89.2 / 53.8 | 33.2 / 91.9 | 3.6 / 70.5 / 17.4 |
| ray map, our DLT | 22.6 / 87.1 / – | 1.9 / 79.6 | 1.0 / 60.9 / – |
| Self-Audit per-view choice | 29.4 / 88.3 / 50.3 | 35.1 / 92.1 | 2.8 / 67.2 / 11.3 |
| Self-Audit per-view fusion | 31.6 / 88.9 / 53.5 | 38.1 / 92.8 | 3.9 / 69.3 / 8.5 |
| oracle choice (uses GT, upper bound) | 33.9 / 89.5 / – | 43.3 / 93.4 | 3.7 / 72.8 / – |

**Observations to verify in the full run (not conclusions):**
- **Our DLT ray solver is much worse than the official RANSAC solver** (DTU-64 AUC@3 1.9 vs 33.2). The
  report expected a 0.3–0.6° difference; this is far larger, so the DLT poses should not be used for
  any headline number, and the cause (no confidence weighting / no outlier rejection in Eq. 2) is worth
  a note in the baseline report.
- The oracle beats the best single source by +1.5 (7Scenes) and +2.6 (DTU-64) AUC@3: there is some
  headroom in choosing the pose source per view, but the audit's choice does not capture it yet
  (it is fitted on very few views here).
- Failure detection (A): on the pooled views the audit's AUROC(>15°) is 0.986 vs 0.948 for DA3's
  confidence, but only HiRoom has gross failures (1 of the 3 datasets), so G1 is "n.a." in this run.
- Depth uncertainty (C): triple collocation gives 90% coverage 0.93 (HiRoom) / 0.72 (7Scenes) without GT,
  vs 0.997 / 0.82 for the GT-calibrated confidence; G3 = 62% of bins within [0.7, 1.4] (needs ≥ 75%).
- Gates on this run (informative only): G1 n.a., G2 FAIL, G3 FAIL, G4 FAIL (the E_c residual adds
  +0.003 AUROC, criterion +0.03).

_Pending the Kaggle run. Fill from `self_audit/<tag>/summary.md`; the records are in
`experiments/results/`._

### 4.1 A. Failure detection
| Model | Dataset | AUROC > 15° (DA3 conf → audit) | AUROC > 5° | Spearman | AURC (deg) |
|---|---|---|---|---|---|
| DA3-Small | HiRoom | – → – | – → – | – → – | – → – |
| DA3-Small | 7Scenes | – → – | – → – | – → – | – → – |
| DA3-Small | DTU-64 | – → – | – → – | – → – | – → – |
| DA3-Base | … | | | | |

### 4.2 B. Pose source (Sec. 6)
| Model | Variant | HiRoom AUC@3 / AUC@30 / F1 | 7Scenes AUC@3 / AUC@30 / F1 | DTU-64 AUC@3 / AUC@30 |
|---|---|---|---|---|
| DA3-Small | camera head (original) | – | – | – |
| DA3-Small | ray map, official RANSAC (original option) | – | – | – |
| DA3-Small | Self-Audit choice | – | – | – |
| DA3-Small | Self-Audit fusion | – | – | – |
| DA3-Small | oracle choice (upper bound) | – | – | – |

### 4.3 C. Depth uncertainty
| Model | Dataset | Method | coverage@50% | coverage@90% | NLL |
|---|---|---|---|---|---|
| DA3-Small | HiRoom | DA3 conf + GT calibration from other datasets | – | – | – |
| DA3-Small | HiRoom | Self-Audit triple collocation (no GT) | – | – | – |
| DA3-Small | HiRoom | upper bound: per-scene GT fit | – | – | – |

### 4.4 Gates
| Gate | DA3-Small | DA3-Base |
|---|---|---|
| G1 detector signals | – | – |
| G2 pose-source choice | – | – |
| G3 GT-free calibration | – | – |
| G4 E_c residual | – | – |

## 5. Discussion

_After the run: which signals carry the information (coefficients figure), where the audit
fails, whether the per-view pose choice moves Tab. 2 numbers, and how close the GT-free
calibration gets to the per-scene GT fit._

## 6. Limitations and honest claims

- **Shared backbone.** Head and ray are predicted from the same features. Failures where both are
  wrong in the same way (symmetric or low-overlap scenes) are invisible to the head–ray gap; the
  reprojection and E_c signals are partly independent.
- **Triple collocation cannot see errors shared by all views** (global scale drift, bending). It
  also absorbs camera errors into the depth σ.
- **7Scenes GT depth** is Kinect depth that is not registered to the RGB camera, so per-pixel depth
  errors include that offset. HiRoom (synthetic) is the clean case for C.
- **Small fitting set.** The audit model is trained on three datasets. It is fitted
  leave-one-dataset-out, so every reported number is out-of-dataset.
- **3 of the 5 benchmark datasets (ETH3D and ScanNet++ not evaluated), because of compute.**
  - **Measured cost** on Kaggle (one T4, 4 CPU cores, DA3-Small, full run): ~2.5 h per model for
    HiRoom + 7Scenes + DTU-64. That is 10 min of model inference, then CPU-bound scoring: 32 min
    for the original DA3, ~30 min for each of the three reconstruction replays of the Self-Audit
    variants.
  - **Cost of the missing datasets:** ETH3D (15 GB) and ScanNet++ (11 GB) do not fit in the same
    session's disk next to the others. Each would need its own session per model, with the same
    four reconstruction scorings: at least four more multi-hour T4 sessions for DA3-Small and
    DA3-Base.
  - **Why that did not fit:** this was not possible within the project deadline and the weekly
    Kaggle GPU quota, with long T4 queues on Kaggle and Colab during the project.
  - **What the omission does and does not mean:** it is a budget decision, not a technical limit.
    The same notebook runs them unchanged (`DATASETS = ["eth3d"]`, then `["scannetpp"]`). The
    conclusions hold for HiRoom, 7Scenes and DTU-64; how far they generalise to ETH3D (large
    outdoor and indoor scenes) and ScanNet++ is untested.
- **May claim** (not found in the literature, report §3.4):
  - DA3-specific redundancy signals: head-vs-ray gap, ray-origin spread, pinhole residual, E_c
    fixed-point residual;
  - per-view head/ray pose choice and fusion;
  - per-scene GT-free triple-collocation calibration;
  - the first reliability audit of DA3.
- **Must cite, not claim:**
  - trained reliability heads (GeoCond);
  - distractor rejection (RobustVGGT);
  - perturbation variance (ReVar3R);
  - GT-fitted calibration audits (arXiv 2608.29705);
  - the ray-to-camera solver (official RANSAC, Cameras-as-Rays).

## 7. Reproduce

Notebook `05_self_audit` on Kaggle (GPU); see [docs/ENHANCEMENTS_RUNBOOK.md](../docs/ENHANCEMENTS_RUNBOOK.md). The two commands it runs per model:

```
python scripts/eval_bench.py --official-src <official>/src --bench-root /tmp/da3bench --work-root <work> \
    --weights official:da3-small --datasets hiroom 7scenes dtu64 --modes pose recon_unposed \
    --tag sa_da3-small_hiroom-7scenes-dtu64 --enhancement baseline --variant head --dump-audit \
    --results-root <out>/results
python scripts/self_audit_eval.py --official-src <official>/src --bench-root /tmp/da3bench --work-root <work> \
    --baseline-tag sa_da3-small_hiroom-7scenes-dtu64 --weights official:da3-small \
    --datasets hiroom 7scenes dtu64 --results-root <out>/results --out-dir <out>/sa_da3-small_hiroom-7scenes-dtu64
```
