# Enhancement 1: DA3 Self-Audit, original DA3 vs DA3 + Self-Audit

**Status (2026-11-22): implemented, not yet run.**
- Method, setup and claims below are final.
- The result sections (§4) are filled from the Kaggle run of `notebooks/05_self_audit`, setting
  **[official]** (released DA3-Small / DA3-Base weights run by us).
- Until then every number in this file is a placeholder (–).

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
