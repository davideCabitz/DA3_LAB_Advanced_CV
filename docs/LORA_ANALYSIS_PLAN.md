# LoRA / enhancement #3: pre-registered statistical analysis plan

**Version 1, fixed on 2026-11-25, before any LoRA training run and before any LoRA result exists.
Version 2 (same day, still before any run): reduced scope for a 4-day deadline, see the deviations log.**
The numbers live in [configs/lora_analysis.yaml](../configs/lora_analysis.yaml). Every result
record stores that file's SHA-256, and the analysis warns when a run was made under a different
plan. The git history is the timestamp. After results are in, the plan is not edited: a changed plan
gets a new version, and both analyses are reported.

Method and expectations: [LORA.md](LORA.md) (Phases 1–3) and
[da3_enhancement_report.md](da3_enhancement_report.md) §4.4. Template:
[FOVEATED_ANALYSIS_PLAN.md](FOVEATED_ANALYSIS_PLAN.md).

## 1. Comparison and hypotheses

**Comparison:** released DA3 (reference arm, no LoRA, bit-identical to LoRA off) vs DA3 + LoRA, same
weights, same model size, same protocol. Confirmatory decisions use **DA3-Small**; DA3-Base is a
descriptive replication.

| ID | Role | Question | Estimand | Decision type |
|---|---|---|---|---|
| H1 | **primary** | Does the E3 LoRA reduce depth error on held-out KITTI drives? | AbsRel vs LiDAR (one median scale per clip), ratio of drive means E3 / released | superiority; null = equivalence within ±3% |
| S1–S3 | secondary (Holm) | Does it improve trajectories? | ATE, RPE-translation, RPE-rotation on the same clips, ratio of drive means | superiority |
| S4 | secondary (Holm) | Is the teacher's depth the reason (novelty)? | AbsRel, E3 / pose-only LoRA | superiority |
| S5 | secondary (Holm) | Is it more than generic fine-tuning? | AbsRel, E3 / own-output LoRA | superiority |
| N1–N6 | non-inferiority (IUT) | Does the LoRA leave the paper's benchmarks intact? | Sec. 6 AUC@3, AUC@30, F1, DTU Chamfer; monocular δ1, AbsRel | non-inferiority with fixed margins (§6) |

**Why AbsRel is primary and not ATE.** E3's claim rests on the teacher's depth: if only the poses
improve, E3 is ordinary pose-supervised fine-tuning (report §4.4). ATE has the larger expected gain
(−20–35% vs −3–8%), so choosing it as primary would favour a positive result for the less novel part.
The primary may therefore be underpowered at the low end of the expected range; that is reported,
not fixed afterwards.

## 2. Units, pairing and seeds

- **Unit of analysis:** the scene for Sec. 6 (HiRoom 30, ETH3D 11, DTU 22, 7Scenes 7, ScanNet++ 20
  scenes, up to 100 images, the harness' fixed seed); the **drive** for KITTI (12 test drives,
  10 clips each, averaged within the drive); the scene's image set for monocular depth. Views,
  clips and pixels within a unit are correlated and are never treated as independent.
- **Paired design:** both arms see identical scenes, frames, frame order, resolution (504 px longest
  side for Sec. 6, 504×280 for KITTI), reference view ("first"), camera source (D_C) and inference
  seeds. The released arm is evaluated once per protocol; the LoRA arms once per training seed.
- **Seeds:** 3 training seeds (minimum 2). The per-unit effect is
  d_j = mean_s(B_{s,j}) − A_j (or the ratio analogue below). Inference is over units, conditional on
  the seed set. Seed variance is reported separately and never counted as an effect:
  - the estimate for each seed alone, their range and standard deviation;
  - a "seed-robust" label only if every seed's point estimate lies on the same side as the pooled one;
  - a two-stage bootstrap (units, then seeds with replacement) as a sensitivity CI. With 3 seeds this
    is crude and is labelled as such.

## 3. Estimation, tests and the decision rule

- **CI:** 95% percentile bootstrap over units (cluster bootstrap), 10,000 resamples, seed 0. Pooled
  Sec. 6 estimands resample scenes **within** each dataset (stratified).
- **Test:** paired **sign-flip permutation test** on the estimand itself. Exact (all 2^n sign
  patterns) for n ≤ 16, e.g. 7Scenes (128 patterns) or 12 KITTI drives (4096); Monte Carlo with
  100,000 flips (seed 0) otherwise. One-sided α = 0.025.
  - Why this test: it targets the mean (the estimand), it is exact for small n, and it assumes only
    that the paired differences are symmetric under the null. The bootstrap percentile CI undercovers
    for n ≈ 7–12, so decisions do not rest on it alone. The Wilcoxon signed-rank p-value is reported
    but never decides: it tests a pseudo-median, not the mean.
  - Ratio estimands are linearised per unit so that the test stays a paired mean test:
    "mean(B)/mean(A) < c" ⇔ mean(B_j − c·A_j) < 0, with e_j = B_j − c·A_j sign-flipped.
- **Effect sizes:** mean paired difference, relative change (ratio of means − 1), d_z = mean(d)/sd(d),
  fraction of units improved.
- **Decision (three-way plus null), for superiority:**

| Outcome | Rule |
|---|---|
| PASS (improvement) | permutation p < α (Holm-adjusted where applicable) **and** the 95% CI excludes no effect on the good side |
| FAIL (regression) | the 95% CI lies entirely on the bad side |
| NULL (no meaningful effect) | the 95% CI lies entirely within the equivalence margin (H1: ratio in [0.97, 1.03]) |
| INCONCLUSIVE | anything else, including test and CI disagreeing |

- **Decision for non-inferiority** (component Nk with margin m; "higher is better" shown):
  PASS if the shifted test (H0: Δ ≤ −m, sign flips of d_j + m) gives p < 0.025 **and** the CI lower
  bound > −m; FAIL if the CI upper bound < −m; otherwise INCONCLUSIVE. Relative margins use
  e_j = B_j − (1 + m)·A_j for lower-is-better metrics.
- For every decision the margins at which it would change are reported (as in the #2 plan).

## 4. Multiplicity

- **H1** alone at one-sided α = 0.025.
- **S1–S5:** one Holm–Bonferroni family at α = 0.025, **gated by H1** (tested confirmatorily only if
  H1 PASSes). If H1 does not pass, S1–S5 are still computed and shown, labelled exploratory, with
  unadjusted p-values and no claim.
- **N1–N6:** an intersection–union test. "No regression on the paper's benchmarks" is claimed only if
  every component PASSes; each component is tested at α = 0.025 and the IUT needs no correction.
  A component whose data could not be evaluated is reported as "not evaluated" and the claim is
  restricted to the components that ran; nothing is imputed.
- **Descriptive** (no decision): per-dataset Sec. 6 rows (Holm-adjusted across the 5 datasets per
  metric), posed reconstruction, KITTI per-image AbsRel / δ1 / pose AUC, the ray-solver camera source,
  KITTI under the Tab. 4 protocol (in domain), the DA3-Base replication, Free Geometry, seed spread.

## 5. Benchmarks and metrics

| Benchmark | Data | Metrics | Status |
|---|---|---|---|
| Sec. 6 pose | HiRoom, ETH3D, DTU, 7Scenes, ScanNet++ | AUC@3, AUC@30 | released harness (pinned 3d835ec) via `bench_adapter.py` |
| Sec. 6 reconstruction | same | evo + RANSAC alignment, TSDF with the paper's voxel sizes; F1 (0.05 m HiRoom / 7Scenes / ScanNet++, 0.25 m ETH3D), Chamfer (DTU) | harness; Open3D 0.19.0 side environment when needed |
| Monocular depth (N_v = 1) | `mono_datasets` (v1: ETH3D from DA3-BENCH) | AbsRel, δ1 after least-squares scale + shift in depth; disparity alignment as sensitivity | loader to write; NYU / Sintel / DIODE only if added **before** any LoRA result (a later addition is a deviation) |
| In domain | 12 held-out KITTI drives | AbsRel, δ1 vs raw LiDAR (≤ 80 m, within the crop); ATE (Sim(3)-aligned, m), RPE-trans (m), RPE-rot (°) vs OXTS | converter and metrics to write |
| NVS | DL3DV, Tanks and Temples, MegaDepth | PSNR / SSIM / LPIPS | **not evaluated**: no GS-DPT head or 3DGS pipeline exists in the repository; no number is reported |

**Paper-level aggregates.** Next to the paired analysis, each Sec. 6 dataset mean is reported as in
Tab. 2 / Tab. 3 with its tag: **[paper]** (copied, where the paper lists the model),
**[official]** (released weights run by us), **[official+LoRA]** (mean over seeds, with the seed
range). These rows are for comparability only; decisions come from §3–4.

**A/A checks (treatment-blind).**
- LoRA off must give exports byte-identical to the released model; otherwise the run stops.
- The released model is evaluated twice on HiRoom in two sessions; any difference is evaluation
  non-determinism and is reported (flag above 0.1 point).
- The own-output LoRA (no new information) is the training-noise control; it is *not* used to widen
  any margin, because it can itself cause forgetting.

## 6. Non-inferiority margins (fixed now)

| ID | Metric | Margin | Reason |
|---|---|---|---|
| N1 | AUC@3, pooled over 5 datasets | −1.0 point | reported in points at 0.1 precision; at DA3-Small's AUC@3 level (≈ 15) this is ≈ 7% relative, wider than #2's 3% because AUC@3 is a thresholded, high-variance metric; it is half the report's upper forgetting estimate (−2) so a model with that forgetting does not pass |
| N2 | AUC@30, pooled | −1.0 point | ≈ 1–1.5% of typical AUC@30 values |
| N3 | F1, pooled over 4 datasets | −1.0 point | same reasoning as N2 |
| N4 | DTU Chamfer | +3% relative | the 3% tolerance of the #2 plan |
| N5 | monocular δ1 | −0.5 point | δ1 is close to its ceiling (Tab. 4: 75–99) |
| N6 | monocular AbsRel | +3% relative | the 3% tolerance of the #2 plan |

## 7. Leakage

- LoRA training and validation drives are disjoint from every test unit:
  - Sec. 6 test sets contain no KITTI data (disjoint by construction);
  - KITTI test drives are drawn from the Eigen test list; the converter refuses any training or
    validation drive on that list, and the split (drive names + SHA-256 of the list) is stored in
    every record;
  - the anchor replay set (TartanAir subset) is in DA3's Tab. 1 training data and in no test set.
- DA3's own training data contain vKITTI2 (synthetic clones of KITTI tracking scenes); real KITTI
  images are not in Tab. 1. This is stated in the report, not hidden.
- Hyper-parameters are frozen in the YAML; checks during development use the validation drives only.
  The test drives and Sec. 6 are evaluated once per trained adapter.

## 8. Records

Every run is one `da3-result/1` record under `experiments/results/lora/` (pending the owner's
choice of folder name, [LORA.md](LORA.md) "Open decisions"):
- `weights = "official:da3-small"` for both arms, so `da3.compare` pairs them (it groups by weights);
- `setting = "official+LoRA"`, `variant` = arm (`e3`, `pose_only`, `own_output`, `free_geometry`);
- new optional fields: `adapter` {path, sha256, rank, alpha, blocks, seed}, `split` {train, val, test
  drives, list sha256, overlap checks}, `plan_sha256`;
- `extra`: `kitti` (per drive), `mono` (per scene), seed-level summaries;
- config, seed, commit hash, environment and GPU as in every record. All records appear in
  `09_compare_all`.

## 9. Reporting rules

- Every hypothesis is reported, including FAIL, NULL and INCONCLUSIVE, with estimate, CI, p-value,
  effect sizes, seed spread and the margins at which the decision changes.
- KITTI numbers are never mixed into Sec. 6 tables; KITTI under the Tab. 4 protocol is in-domain for
  the LoRA and never counts towards "no regression".
- No improvement is claimed that §3–4 do not support.

## Deviations log

**1. 2026-11-25, plan version 2, before any LoRA run or result** (no number of any arm had been seen;
only part 0 of notebook 07, which trains nothing, had been written). Reason: a 4-day deadline.
Hypotheses, estimands, tests, margins and decision rules are unchanged. Changed:
- seeds 3 → **2** (the plan's minimum);
- Sec. 6 non-inferiority on **HiRoom, 7Scenes, DTU-64** only (N1, N2 on all three; N3 F1 on HiRoom +
  7Scenes; N4 Chamfer on DTU-64). ETH3D and ScanNet++ are **not evaluated** (disk / time); the
  "no regression" claim is restricted to these three datasets;
- monocular N5 / N6: **not evaluated** (no loader in time);
- dropped: DA3-Base replication, Free Geometry;
- anchor: frozen unposed DA3 outputs on the KITTI training clips instead of a TartanAir replay set
  (no extra download); a weaker guard against forgetting, stated as such;
- data: KITTI raw from a public Kaggle mirror (CC BY-NC-SA allows non-commercial redistribution) instead
  of the official download; test drives = all available Eigen test drives up to 12, by the same rule.
- training drives: at most 10, drawn with the split seed from the non-Eigen drives (v1: all of them);
  only the drives of the split are converted.
