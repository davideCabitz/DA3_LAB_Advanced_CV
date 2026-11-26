# DA3 enhancement report: improving a frozen Depth Anything 3

> **Update 2026-11-26.** The three selected ideas were run: #2 Foveated and #3 LoRA (E3) are no-go by their
> pre-registered gates, #1 Self-Audit fails gates G2-G4 in its smoke test (full run pending). Their measured findings
> led to a new enhancement, #4 lens-consistent DA3: [LENS.md](LENS.md) (§1 there explains the route). This report is
> kept unchanged as the record of the original selection.

**Date:** 2026-11-19.
**Status:** planning document. No enhancement has been run yet, so every gain below is an **estimate**, not a measurement.

**Constraint.** DA3 itself stays as released, because we cannot retrain it on its 22 training datasets.
- **Allowed:** training-free methods, add-on modules on frozen features, switchable LoRA.
- **Compute:** Kaggle T4/P100 (16 GB), about 30 GPU-h per week, sessions of up to ~12 h.

**Discard rule (user, 2026-11-19).** Everything already implemented in the literature is discarded, applied strictly:
- If an idea's core mechanism is published on *any* feed-forward 3D model (VGGT, π³, DUSt3R/MASt3R, MapAnything, ...), the idea is discarded, even if nobody has applied it to DA3 yet.
- A partially covered idea survives only if the unpublished part is its core, and it is re-scoped to that part.

## TL;DR

| Rank | Idea | Route | In one line |
|---|---|---|---|
| **1** | **DA3 Self-Audit** (A2 + C1 + F3 + F2) | training-free | DA3 estimates each camera twice and over-parameterises it; the disagreement becomes a per-view failure score, a per-view choice between head and ray poses, and a per-scene uncertainty calibrated without ground truth. |
| **2** | **Foveated DA3** (B2, with H4 as its region variant) | training-free | One pass where views have different resolutions: focus views (or zoom views warped from the full-resolution photo) at 504 px, context views at 252 px. |
| **3** | **Pose-conditioned self-distillation** (E3) | LoRA, 1–2 Kaggle sessions | In a new domain with poses but no depth labels, DA3's own pose-conditioned pass teaches a LoRA on the unposed model; no poses needed at test time. |

- **Runners-up:** E_c camera-input optimisation (F1(b)) and self-calibrating undistortion (F4) (§4.5).
- **Discarded:** 23 of the 35 ideas, because they are already published (§3.3).
- **Not run yet:** the [official] DA3-Small/Base baseline on Kaggle. It is step 0 of the plan (§5.5).

## About this report

How the study was run (14 agents in three phases):

| Phase | Agents | Output |
|---|---|---|
| 1. Idea generation | 8 in parallel, one direction each (A–H) | 35 ideas, each with description, motivation and benefit, complexity, risks |
| 2a. Grouping | 1 | 8 clusters (K1–K8), merged "canonical" ideas, cross-cluster dependencies |
| 2b. Impact | 4 in parallel, 2 clusters each | expected gains (pessimistic / realistic / optimistic), metrics, cost, failure modes, first go/no-go experiment, impact and feasibility scores (1–5) |
| 3. Novelty | 1, with web search (arXiv, Hugging Face, GitHub) | per idea: published / partially covered / novel, with references |
| Selection | orchestrator | top 3 among the survivors, their interactions, implementation order |

All agents worked from the same sources:
- the DA3 paper and its tables;
- this repository: the parity-checked `da3-*` implementation, and the benchmark adapter on the released harness (commit 3d835ec).

Caveats:
- **Gains are estimates.** Phase 2 extrapolated from the paper's tables and from the repository; nothing was run.
- **Most 2026 papers were read at abstract level** (read level per reference in §3.6). Re-check the arXiv IDs before they go on slides, and repeat the novelty search the week before the exam.
- **Setting tags** are the ones used in the rest of the repo: **[paper]** = numbers from the DA3 paper; **[official]** = released weights run by us (none yet).

DA3 numbers the analysis relies on **[paper]**:
- **Pose is the weak point of the Apache-2.0 models.**
  - HiRoom AUC@3 (Tab. 2): Small 9.5, Base 19.0, Large 58.7, Giant 80.3.
  - Mean AUC@3 over the five pose datasets: Small 14.7, Base 27.9, i.e. 52.5 points below Giant.
- **Better poses would lift reconstruction a lot.**
  - HiRoom F1 with GT poses (Tab. 3): Small 18.3 → 52.2, Base 25.9 → 71.4.
  - Giant reaches 85.1 without poses.
- **Pose conditioning through E_c helps, but not everywhere** (ViT-L with GT poses, Tab. 7 f→g):
  - HiRoom F1 65.8 → 73.8;
  - DTU CD 3.65 → 2.14;
  - but 7Scenes F1 58.4 → 46.0.

---

## 1. Phase 1: ideas from each agent

Each agent explored one direction under the same constraint (frozen DA3). For every idea it returned a description, the motivation and expected benefit, the implementation complexity and the main risks.

The last column anticipates the Phase 3 verdict:
- **kept**: survives the novelty check;
- **kept (weak)**: survives, with a thin novelty margin;
- **merged**: survives inside another idea;
- **discarded**: already published.

### A. Inference-time and test-time techniques (agent's top pick: A1)
| Idea | What it does | Motivation and expected benefit | Complexity | Main risks | Phase 3 |
|---|---|---|---|---|---|
| A1 Refine-and-Recondition | Unposed pass → light bundle adjustment (ALIKED/LightGlue matches, tracks lifted with DA3 depth, pycolmap) → second pass conditioned on the refined cameras through E_c; 1–3 rounds. Variant without BA: is DA3 a pose denoiser? | Attacks the Small/Base pose gap; E_c adds 3–8 F1 with GT poses (Tab. 7); noise-sensitivity curve as a by-product | Medium, no training, ~10–15 GPU-h | E_c saw only GT poses, so BA residuals may be amplified (7Scenes −12 F1 in Tab. 7); BA fails on low texture; BA on feed-forward poses is prior art | kept (weak): only the iterated loop |
| A2 DA3 checks itself | Per view: head-vs-ray pose gap, spread of ray origins (Eq. 1), pinhole residual of the homography fit (Eq. 2), confidence statistics, cross-view depth reprojection → flag bad views, reject distractors, pick head or ray pose per view | DA3 gives no per-view reliability; selective prediction and robustness to contaminated photo sets | Low, training-free (optional 5-feature logistic fit), 3–6 GPU-h | Shared backbone, so both branches can be wrong together; DINOv2 retrieval may already catch distractors; overlap with RobustVGGT, ReVar3R | **kept**, merged into #1 |
| A3 Coarse global + fine chunks | Low-res pass over all views for global poses, then ≤ 18-view chunks at 504–756 px conditioned on them through E_c | Quadratic cross-view cost; DA3-Streaming drifts and peaks at 18.7–28.3 GB | Medium, 6–10 GPU-h | Inputs below 504 px are out of distribution; seams; HD-VGGT | discarded |
| A4 Partially-posed inference | Per-view camera tokens: E_c tokens for posed views, learnable token for the others → localise new views in a posed map | New capability: relocalisation, incremental mapping | Low–Medium (~30 lines), 3–6 GPU-h | Mixed tokens never seen in training; MapAnything and Pow3R are trained for this | discarded |

### B. Efficiency and scalability (top pick: B1)
Agent's prior-work note: plain token merging and KV pruning on DA3 are already published (Spark3R: ~17× faster at 1000 frames, accuracy unchanged). These ideas therefore rely on what only DA3 has.

| Idea | What it does | Motivation and expected benefit | Complexity | Main risks | Phase 3 |
|---|---|---|---|---|---|
| B1 Skeleton-anchored long video | One pass over 100–300 keyframes gives a pose skeleton; the other frames run in chunks together with their nearest keyframes, posed through E_c (V1 mixed tokens, V2 two passes) | DA3-Streaming chains Sim(3) alignments: TUM ATE 0.087 m at chunk 120 vs 0.227 m at chunk 30, at 18.7–28.3 GB (more than a T4) | Medium, 8–10 GPU-h | V1 out of distribution; skeleton errors propagate | discarded |
| B2 Mixed-resolution ("foveated") inference | Focus views at 504 px + many context views at 252/168 px in one pass: within-view blocks and DPT per resolution group, cross-view blocks over the concatenated variable-length token set | 10 focus + 190 context views need ≈ 12× less cross-view attention than 200 full-res views; detail only where needed | Low–Medium, 4–8 GPU-h | Training used one resolution per batch (short side ≥ 280 px); reference-view choice may depend on resolution; plain downscaling may tie | **kept (novel)** → #2 |
| B3 Covisibility-sparse attention | Each view's patches attend only to itself, the reference, k covisible views and all camera tokens (as relays) | Linear in the number of views for fixed k; 1–2k frames per T4 pass | Medium, ~8 GPU-h | Missed loop-closure neighbours; Block-Sparse, AVGGT, FlashVGGT are close | discarded |
| B4 Efficiency LoRA | LoRA on the cross-view blocks so that compressed DA3 reproduces uncompressed DA3 | Compression is lossless at moderate ratios but lossy at the ratios a T4 needs | Medium, 2–3 runs × 3–4 h | No headroom if compression is lossless; transfer to many views | discarded |

### C. Robustness and generalization (top pick: C1)
| Idea | What it does | Motivation and expected benefit | Complexity | Main risks | Phase 3 |
|---|---|---|---|---|---|
| C1 Self-diagnosis + calibrated reliability | A2's signals + **E_c fixed-point residual** (feed DA3's own poses back through E_c, measure how much depth and rays move) → isotonic/logistic calibrator → drop flagged views, re-run | DA3 fails silently (low overlap, mixed scenes, corruptions); no confidence audit of DA3 exists | Low, 4–8 GPU-h | Self-consistency is not accuracy; per-scene calibration may stay poor; overlap with GeoCond, ReVar3R | **kept**, merged into #1 |
| C2 Fisheye / 360° | (i) search the lens distortion that makes DA3 most self-consistent; (ii) split wide frames into rotated pinhole views conditioned jointly through E_c | E_c, D_C and Eq. 2 assume a pinhole camera; undistort-and-crop loses field of view | Medium, 5–8 GPU-h | Rotation-only view sets are out of distribution; VGGT-360, Fisheye3R, Wid3R | discarded ((i) folded into F4) |
| C3 Sky head | ~0.5M-parameter head distilled from DA3MONO-LARGE's sky output; sky pixels become points at infinity | Small/Base put the sky at finite depth (floaters) | Low, one session | Engineering; confidence may already remove the sky | discarded |
| C4 Physics-in-the-loop haze / underwater | Fit a scattering model shared by all views using DA3 depth and cross-view constraints, restore the images, re-run DA3, iterate | Contrast drops with distance, so far-range geometry degrades first | Medium, 3–6 GPU-h | DA3 may already be robust; noise where transmission → 0; Sea-thru-style prior art | kept (weak) |

### D. New capabilities via add-on heads (top pick: D1)
| Idea | What it does | Motivation and expected benefit | Complexity | Main risks | Phase 3 |
|---|---|---|---|---|---|
| D1 Correspondence head + BA through E_c | MASt3R-style descriptor branch on frozen features → matches → BA → cameras back through E_c | AUC@3 is the weakest Small/Base number; also gives DA3 matching | Medium–High, 4–6 GPU-h for the head | Stride-14 features too coarse; Glob3R, Selfi | discarded |
| D2 Normals + sky ("Tri-DPT") | Third fusion stack for normals (with uncertainty) and sky, distilled from MoGe-2/DSINE and DA3MONO; cross-view normal consistency | The teacher had normal and sky losses, but the any-view model outputs neither | Low, 2–4 GPU-h | Blurry normals; Surflo, WorldMirror, Dens3R | discarded |
| D3 Mirror / glass head | Mask mirrors and glass, replace their depth with a plane fitted at the border, remove phantom points | Reflections and transmissions give confidently wrong geometry | Low–Medium, 3–4 GPU-h | Curved glass; Mirror3D | discarded |
| D4 Gravity + metric scale head | Small MLP on camera tokens → per-view gravity and log metric scale at a canonical focal | Metric, gravity-up frames without a second model | Low, 1–2 GPU-h | Scale is ambiguous across domains; AMB3R, G3T | discarded |
| D5 Query views without images (moonshot) | A generator turns a query camera's rays into tokens; frozen DA3 predicts depth where no camera looked | Planning and view selection need unseen geometry | High, 8–10 GPU-h | Frozen attention matches by appearance and may ignore generated tokens; CUT3R | discarded |

### E. Adaptation on a frozen base (top pick: E1)
Agent's prior-work note: test-time LoRA self-distillation on DA3 is published (Free Geometry: full- vs partial-view consistency, +3.7% pose AUC).

| Idea | What it does | Motivation and expected benefit | Complexity | Main risks | Phase 3 |
|---|---|---|---|---|---|
| E1 Giant → Small/Base pose distillation | Cache Giant/Large-1.1 outputs on ~3k unlabelled clips; LoRA on blocks 4–11 + a trainable copy of D_C on Small/Base | Small is ~4× faster than Giant but far worse at poses; close part of that gap at the same inference cost | Medium, 2 sessions | Capacity limit; Giant/Large are CC BY-NC; Distill3R | discarded |
| E2 Adapter for partial / noisy priors | Trainable copy of E_c accepting any subset of {intrinsics, gravity, GPS, VIO poses} + an availability mask + a noise level | Phones record EXIF and gravity; the paper only tests clean poses | Low–Medium, 4–8 GPU-h | Partial tokens out of distribution; Pow3R, MapAnything, OmniVGGT | discarded |
| E3 Self-distillation from known poses | Teacher: frozen DA3 conditioned on known poses through E_c. Student: unposed DA3 + LoRA, trained on the teacher's depth and the known poses. No poses at test time | Domains with poses but no dense depth (driving, robot kinematics, COLMAP videos); the teacher is DA3 itself | Medium, 6–10 GPU-h | Conditioned depth may not beat unposed depth out of domain; forgetting | **kept** → #3 |
| E4 Fisheye through the ray map | Probe whether rays bend on distorted images; LoRA on the ray branch + a generic central-camera solver | Fisheye and action cameras without calibration | Medium–High, 6–10 GPU-h | Wid3R, Fisheye3R; no peripheral training data | discarded |

### F. Losses and self-supervision without ground truth (top pick: F1)
Agent's prior-work note: consistency-based LoRA test-time adaptation of DA3 is published (Free Geometry, Self-Geometry).

| Idea | What it does | Motivation and expected benefit | Complexity | Main risks | Phase 3 |
|---|---|---|---|---|---|
| F1 Closed-loop camera refinement through E_c | Optimise only the cameras fed to E_c so that the depth DA3 predicts under them is consistent across views: (a) alternate BA and conditioned re-runs; (b) backpropagate into E_c's inputs | No weights adapted (unlike Self-/Free Geometry, TCO); VGGT and π³ have no pose input | Medium; (b) is limited to ~24 views or 336 px on a T4; 5–8 GPU-h | Depth may bend to fit any input pose (flat objective); the gain may come only from BA | kept: variant (b), runner-up |
| F2 Per-scene calibration by triple collocation | Fit σ = a·c^(−b) per scene from pairwise depth residuals of points seen in three views; same estimator on three pose sources | Feed-forward confidences are overconfident (≥ 2.4× in a 2026 audit); GT-fitted fixes fail per scene | Low, 1–2 GPU-h | Errors shared by all views are invisible, so σ comes out too low | **kept**, merged into #1 |
| F3 Internal redundancy as a signal | Head-vs-ray disagreement and the pinhole residual of the ray field used (a) to fuse pose sources, (b) to detect failures, (c) as a test-time loss on the camera tokens | Ray-map poses beat a camera-only model in the ablation (Tab. 6: 48.7 vs 10.8 AUC@3), yet the API defaults to head poses | Low for (a, b) | Correlated errors; co-adaptation in (c) | merged into #1 |
| F4 Self-calibrating non-pinhole inputs | Gradient-free search over 1–3 lens parameters that maximises frozen-DA3 self-consistency, then undistort | DA3 on uncalibrated wide-angle or fisheye footage | Low–Medium, 2–3 GPU-h | Mild distortion is harmless and strong distortion gives no signal; Fisheye3R, 2512.15608 | kept (weak), runner-up |

### G. Geometry and camera pipeline (top pick: G3)
| Idea | What it does | Motivation and expected benefit | Complexity | Main risks | Phase 3 |
|---|---|---|---|---|---|
| G1 Robust uncertainty-aware ray solver | Gauss–Newton on SO(3) × focal, confidence-weighted Cauchy IRLS, D_C as prior, per-view covariance | The solver alone moves poses by 0.3–0.6°; DA3 gives no pose uncertainty | Low, ~2 GPU-h | A solver can't fix network errors; official RANSAC exists | discarded |
| G2 Self-calibration beyond pinhole | Probe principal point, aspect ratio and distortion on the ray map; joint fit over a video; optional residual head | D_C fixes the principal point at the image centre | Low–Medium, 2–6 GPU-h | No zero-shot signal; AnyCalib, GeoCalib | discarded |
| G3 Closed-loop DA3 | A1 with a pluggable pose updater + a noise-slope probe; E_c-LoRA fallback if E_c copies its input | Pose bottleneck; 7Scenes suggests E_c follows its input even when it is wrong | Medium; probe ~1 GPU-h | If E_c copies, the whole gain is BA (prior work) | merged into A1/F1 |
| G4 Anchor-conditioned chunking | Each chunk includes already reconstructed anchor frames, posed through E_c | Scale drift of Sim(3) chunking | Medium, ~8 GPU-h | Mixed tokens out of distribution; MapAnything | discarded |
| G5 Metric-scale head | ~0.5M head → scene log-scale ± σ; baseline: align to DA3METRIC-LARGE | Metres for AR/robotics with the Apache models | Low–Medium, ~6 GPU-h | Ill-posed; AMB3R | discarded |

### H. Multimodal and extra inputs (top pick: H1)
Agent's own discards (already published): thermal-input adapters for feed-forward 3D models, metric scale from language.

| Idea | What it does | Motivation and expected benefit | Complexity | Main risks | Phase 3 |
|---|---|---|---|---|---|
| H1 Partial / noisy sensor priors | Fill missing camera fields from a first pass aligned to the sensor fields, condition through E_c; a trust gate falls back to DA3's own estimate | Devices give partial priors; imperfect priors can hurt (7Scenes) | Low–Medium, 6–10 GPU-h | E_c may copy its input; VIDAR finds decoupled alignment better | discarded |
| H2 Click-to-fix | User clicks → pose-graph correction → an E_c re-pass spreads the fix to all pixels | Interaction is named as future work in the paper (Sec. 8) | Medium, 5–8 GPU-h | Click3R; simulated users are optimistic | discarded |
| H3 Sparse metric anchors | Zero-initialised encoder adds sparse depth (LiDAR/ToF/SfM) at block 4, with cross-view propagation | Metric scale, textureless regions, mixed rigs | Medium, 8–12 GPU-h | Crowded: PromptDA, CAPA, OmniVGGT | discarded |
| H4 Text/click zoom via virtual cameras | Warp a region of the full-resolution photo into a rotated, narrow-FOV virtual pinhole; posed pass together with the scene views through E_c | DA3 runs at ~504 px and throws away most DSLR pixels | Low, 2–4 GPU-h | Narrow FOV out of distribution; E_c may not follow | kept (weak) → region variant of #2 |
| H5 Pose-conditioned chunk chaining | Condition chunk k+1 on the poses of the overlap frames from chunk k | Drift on long sequences | Low–Medium, 4–8 GPU-h | Copy risk; pose graphs (HiP3R) may match it | discarded |

**Where the agents converged.**
- **E_c.** Six of the eight top picks (A1, B1, D1, F1, G3, H1) use E_c, DA3's camera-pose input, which VGGT and π³ lack. Most of these E_c-feedback ideas fell under the strict rule, because feeding external poses into DA3 is already published (§3.5).
- **Self-diagnosis.** Three directions independently proposed self-diagnosis from DA3's redundant outputs (A2, C1, F3). This theme survived.

---

## 2. Phase 2: grouping and impact analysis

### 2.1 Clusters
| Cluster | Theme | Members | Canonical merged idea |
|---|---|---|---|
| K1 Re-conditioning loop | Refine cameras outside DA3, feed them back through E_c | A1, G3, F1, D1, H2 | Refine-and-Recondition (A1 + G3 + F1(a) + D1's BA step), gated by an E_c noise probe |
| K2 Imperfect priors | Use partial or noisy priors without ever doing worse than unposed | H1, A4, E2, H3 | Prior-tolerant conditioning (H1 + A4 + E2) |
| K3 Anchored long video | Condition chunks on one global pose skeleton instead of chaining Sim(3) | B1, G4, H5, A3 | Skeleton-anchored chunking |
| K4 Token budget | Spend resolution and attention where they matter, within one pass | B2, H4, B3 | Foveated DA3 (B2 + H4); B3 kept separate |
| K5 Self-audit and uncertainty | Turn DA3's redundant outputs into reliability scores, a pose-source choice and calibrated σ | A2, C1, F3, G1, F2 | DA3 Self-Audit (A2 + C1 + F3(b)); fused camera recovery (G1 + F3(a) + A2, calibrated by F2) |
| K6 Image-formation correction | Estimate and undo optics or media that DA3 never saw | F4, C2, G2, E4, C4 | Self-calibrating undistortion; generic-camera ray adapter |
| K7 Add-on heads | New outputs for Small/Base, read from frozen features | C3, D2, D4, G5, D3, D5 | Sky + normal head; world-anchoring head |
| K8 Privileged-teacher LoRA | Distil a teacher that has information unavailable at test time into a switchable LoRA | E1, E3, B4 | One distillation platform, three teachers |

### 2.2 Impact per cluster
How to read the tables:
- **Gains** are pessimistic / realistic / optimistic.
- **Cost** is Kaggle GPU-h; development days.
- **I/F** is impact / feasibility, each from 1 to 5.

**K1 Re-conditioning loop**
| Variant | Expected gain | Metrics | Cost | Main failure modes | First go/no-go | I/F |
|---|---|---|---|---|---|---|
| Step 0: E_c noise probe (GT poses + controlled noise; output error vs input error; F1 vs noise) | A gate, not a gain. Predicted slope 0.8–1.0 for D_C (it copies), 0.5–0.9 for ray-map poses; F1 falls to the unposed level at ~3–6° noise (Small) | AUC, F1 on HiRoom, DTU-64, 7Scenes | 1–3; 1–2 | Gauge leak that mimics denoising | — (it is the gate) | 3/5 |
| Refine-and-Recondition | Whole system, Small: HiRoom AUC@3 9.5 → 13 / 25–40 / 55; HiRoom F1 18.3 → 19 / 30–38 / 46. E_c pass on top of BA alone: F1 −3 / +2–5 / +8; 7Scenes −10 / −4…0 / +2 | Tab. 2 + Tab. 3 without poses | 10–20; 10–14 | Whole gain from BA; BA diverges on low texture; E_c follows BA errors | GT-pose isolation (~2–3 GPU-h); drop the E_c part if it gives < +2 F1 | 3 (4 with BA)/4 |
| BA-free self-loop | AUC@3 −2 / 0…+1 / +3 | Tab. 2/3 | 1–2; 1–2 | No new information enters the loop | Probe slope < 0.7 | 2/5 |
| F1(b) gradient search over E_c inputs | On top of BA: AUC@3 0 / +1–3 / +5; F1 −2 / +1–2 / +4 | Subsampled Tab. 2/3 (≤ 24 views or 336 px) | 6–12; 5–7 | Flat objective; ~15 min per scene | Along GT-perturbation directions, is the loss minimum closer to GT than the start? | 2/2 |
| D1 descriptor head as BA front-end | vs LightGlue: AUC@3 −15 / −5…0 / +3 | Through BA | 6–10; 7–10 | No sub-pixel precision at stride 14 | Match precision vs LightGlue | 2/2 |
| H2 click-to-fix | Worst scenes, 5–20 clicks: AUC@3 +2 / +5–15 / +25 | Clicks-to-accuracy curves | 3–6; 6–9 | Simulated clicks are too precise | Oracle GT poses for the worst views | 2/3 |

Verdict:
- Do Refine-and-Recondition, gated by the probe and by the GT-pose isolation (≤ 5 GPU-h together).
- The large system gain comes mostly from BA. If the isolation shows < +2 F1, K1 reduces to "BA on DA3 poses", which is prior art.

**K2 Imperfect priors**
| Variant | Expected gain | Metrics | Cost | Main failure modes | First go/no-go | I/F |
|---|---|---|---|---|---|---|
| Prior-tolerant conditioning (fill-in + trust gate) | Small, HiRoom, VIO-like poses: F1 18.3 → 22 / 30–42 / 48. Intrinsics only: +0 / +1–3 / +4 F1. Gain over decoupled alignment: −3 / 0…+4 / +6 | Simulated priors on HiRoom, DTU-64, 7Scenes, ETH3D | 6–12; 7–10 | E_c copies its input; mixed inputs are out of distribution; decoupled alignment wins (as in VIDAR) | Mixed-input probe vs decoupled alignment (~1 GPU-h) | 3/4 |
| A4 mixed per-view tokens | Hidden views, AUC@3 vs unposed + Sim(3): −10 / −2…+3 / +8 | Hidden-pose rows | 2–5; 3–5 | Tokens ignored | Hide 30% of the poses on 10 scenes | 2/5 |
| E2 trained adapter | F1 at 2–5° noise over the training-free tier: 0 / +2–4 / +8 | Same | 10–18; 8–12 | Collapses to the learnable token | 2-h training run | 3/3 |
| H3 sparse-depth adapter | AbsRel on anchored views: −5 / −10–25 / −40% | New sparse-anchor protocol | 8–14; 8–12 | No propagation to other views | Per-view scale/shift baseline | 2/3 |

Verdict:
- Do the training-free tier with a trust gate, compared against decoupled alignment.
- Train E2 only if zero-shot mixing breaks.

**K3 Anchored long video**
| Variant | Expected gain | Metrics | Cost | Main failure modes | First go/no-go | I/F |
|---|---|---|---|---|---|---|
| Skeleton-anchored chunking, two passes | ATE vs chained Sim(3) at chunk 18–30: −10 / −40 / −60%. vs DA3-Streaming at chunk 120: +10 / −10 / −30% | TUM fr1, full 7Scenes (ATE/RPE, F1) | 10–15; 7–10 | The gap comes from per-chunk quality, not from the chain; E_c pushes skeleton errors into depth | DA3-Streaming at chunk 30 vs 120 vs a strided full batch (~1 GPU-h) | 3/4 |
| Mixed-token mode (V1) | New-frame ATE vs Sim(3) to the skeleton: +20% / −5% / −20% | Same | +2–3; 1–2 | Out of distribution | Anchors must come back within ~1° | 3/4 |
| Resolution split (A3) | Skeleton capacity ×3; chunk F1 at 756–896 px: −1 / +2 / +5 | + ETH3D/DTU | +3–5; +2 | Below 280 px out of distribution | Uniform-resolution curve | 3/4 |
| Online topologies (G4, H5) | ATE vs chained Sim(3): +10 / −10 / −25% | Online ATE | 4–8 | Error propagation | Reuse the V1 probe | 2/3 |

Verdict:
- GO as one study.
- For Small/Base on a T4, K3 is about accuracy, not memory.
- A control without E_c (Sim(3) to the skeleton) is mandatory. In posed mode the released API returns the input extrinsics, so the ATE comes from the skeleton.

**K4 Token budget**
| Variant | Expected gain | Metrics | Cost | Main failure modes | First go/no-go | I/F |
|---|---|---|---|---|---|---|
| B2 view-level foveation | 200–1000 views, 10% focus views: T4 speed ×3 / ×6 / ×9; max views (Small) ~750 → ~2000; focus-view AbsRel vs all at 504 px: +10 / +2 / 0% (worse) | Needs a 300–1000-view protocol: at the Sec. 6 size (≤ 100 views) a T4 is not token-bound | 4–8; 3–5 | Cross-scale token matching; reference view biased by resolution; uniform 336 px may tie | After the resolution curve: 8 views at 504 + 8 at 252 vs all-504 vs all-252 | 2/3 |
| H4 region-level zoom | Region Chamfer at fine thresholds: 0 / −15 / −35%; region F1: +0 / +4 / +10 | ETH3D, DTU, ScanNet++ DSLR; region metrics | 2–4; 2–4 | Narrow FOV out of distribution; detail mostly monocular; a global 896-px pass may capture the gain | Oracle regions on 5 ETH3D scenes vs global 504/896 px and a naive crop | 3/4 |
| B3 covisibility-sparse attention | End-to-end ×3 / ×8 / ×12 at 1000 views | 500–2000 views | 6–10; 5–8 | Missed loop-closure neighbours | ≥ 90% of the attention mass in the top-16 views | 2/2 |

Verdict:
- H4: GO.
- B2: conditional on the resolution curve and a cross-scale probe.
- B3: NO-GO.
- The efficiency bar is Spark3R (~17× on DA3, accuracy unchanged).

**K5 Self-audit and uncertainty**
| Variant | Expected gain | Metrics | Cost | Main failure modes | First go/no-go | I/F |
|---|---|---|---|---|---|---|
| Self-Audit detector (A2 + C1 + F3(b)) | AUROC for gross failures (> 15°): 0.62 / 0.75–0.85 / 0.9; for > 5°: 0.55 / 0.6–0.7 / 0.75. Spearman vs pose error: 0.15 / 0.3–0.45 / 0.6. Depth AUSE vs raw confidence: 0 / −10–20% / −30% | New AUROC and risk–coverage tables; Tab. 2/3 untouched | 3–5; 5–7 | Head and ray wrong together; features encode the dataset, not the error | Dump (~1 GPU-h): Spearman ≥ 0.3 and AUROC ≥ 0.7 on ≥ 2 datasets | 3/5 |
| Drop-and-rerun, distractors | With 30% distractors, share of the inlier-AUC drop recovered: 30% / 60–90% / ~100% | Contaminated protocol | +3–5; 4–8 | DINOv2 retrieval already reaches ≥ 0.95 AUROC | Inlier AUC@30 drops ≥ 5 and DA3's signals beat retrieval on look-alike rooms | 3/4 |
| E_c fixed-point residual (C1) | ΔAUROC over the detector: 0 / +0.02–0.05 / +0.08 | As above | +3–5; 1–2 | E_c ignores or copies its input | After the noise probe: ΔAUROC ≥ 0.03 | 2/4 |
| Fused camera recovery (G1 solver + per-view head/ray fusion) | AUC@3 vs the better official source: −0.5 / +1–3 / +5; F1 without poses: 0 / +0.5–2 / +4 | Tab. 2 AUC@3, Tab. 3 | 1–3; 6–10 | Correlated head and ray errors cap fusion | Per-view oracle − best ≥ 2, or fusion ≥ +0.5 on ≥ 3 of 5 datasets | 3/5 |
| F2 triple-collocation σ | Per-scene error of the 90%-interval coverage: none / 5–12 / < 5 points (a global GT fit is off by 10–25). NLL: 0 / −10–20% / −30% | Calibration table; σ-weighted TSDF | 1–2; 4–6 | Shared errors make σ too low; correlated sources bias the estimate | Estimated σ / true σ within 0.7–1.4 per confidence bin | 2/4 |
| F3(c) correspondence-free TTA | AUC@3: −1 / +0.5–2 / +4 | Tab. 2 | 3–6; 7–10 | Co-adaptation to a wrong consensus | 5 scenes, camera tokens only | 2/3 |

Verdict:
- Do it first: one dump of DA3 outputs (~1 GPU-h) runs four go/no-go tests.
- Fused camera recovery is the only K5 item that moves a Tab. 2 number.

**K6 Image-formation correction**
| Variant | Expected gain | Metrics | Cost | Main failure modes | First go/no-go | I/F |
|---|---|---|---|---|---|---|
| Probe (synthetic distortion, crops, stretches) | Diagnostic. Raw→oracle gap at moderate distortion: AUC@3 −20–50% relative | Distorted DA3-BENCH | 1–2; 3–4 | Resampling blur confounds | — (gate) | 2/5 |
| Self-calibrating undistortion (F4 + C2(i) + G2) | Share of the raw→oracle gap recovered: 10–30% / 50–80% / 90% | Distorted HiRoom/ETH3D/7Scenes, TUM-VI | 4–8; 8–12 | Flat objective; loses to single-image calibration (AnyCalib/GeoCalib) followed by DA3 | 9-point grid: objective minimum within ±25% of the true distortion in ≥ 4 of 5 scenes | 3/4 |
| Generic-camera ray adapter | Synthetic AUC@30 gap closed: 20% / 50–80% / 90% | Fisheye sets | 8–14; 12–18 | Frozen features locked to pinhole | Only if the probe is negative | 3/2 |
| Virtual pinhole views + E_c rig (C2(ii)) | Seam disagreement −30–70% (realistic) | 360° / fisheye | 5–8; 8–12 | Rotation-only conditioning out of distribution | After the noise probe | 3/3 |
| C4 haze / underwater loop | Far-range AbsRel (synthetic fog): 0 / −5–15% / −25% | Fogged ETH3D, SQUID/FLSea | 3–6; 6–10 | DA3 already robust; inverse crime with synthetic fog | Stop if raw far-range AbsRel degrades < 10% at medium fog | 2/3 |

Verdict:
- Probe first.
- No K6 variant moves Tab. 2/3, because the benchmark images are pinhole.

**K7 Add-on heads**
| Variant | Expected gain | Metrics | Cost | Main failure modes | First go/no-go | I/F |
|---|---|---|---|---|---|---|
| Sky + normal head | Sky IoU vs teacher: 0.85 / 0.92 / 0.96. Normal error vs depth-derived normals: −1° / −3–5° / −7° | Sky IoU, NYUv2 normals; in Sec. 6 only ETH3D | 6–9; 5–8 (sky only: 2–3) | Confidence may already remove the sky | 0.5-GPU-h probe | 2/4 |
| World-anchoring head | Metric δ1 indoor: 0.70 / 0.85 / 0.92 | NYUv2, KITTI, ETH3D | 5–7; 4–6 | Aligning to DA3METRIC-LARGE is likely better | Run that baseline first | 2/4 |
| Mirror / glass | Region AbsRel: −5 / −30 / −60% | MSD/GDD, Booster | 4–6; 5–8 | Non-planar glass | Oracle masks | 2/3 |
| Query views | Unobserved-pixel AbsRel: worse / ±10% / −25% | New protocol | 10–14; 8–12 | Generated tokens ignored | 2-h run | 3/1 |

Verdict:
- Low-risk side deliverables.
- They leave DA3's outputs bit-exact, so they cannot move Tab. 2/3.

**K8 Privileged-teacher LoRA**
| Variant | Expected gain | Metrics | Cost | Main failure modes | First go/no-go | I/F |
|---|---|---|---|---|---|---|
| Shared platform (cached teacher, LoRA on blocks 4–11, repo losses, anchor to frozen outputs) | Enabler | — | ~1; 4–6 | Gauge mismatch (reference view, scale normalisation) | LoRA-off bit-exact + overfit 50 clips | –/4 |
| E1 Giant → Small/Base | Mean AUC@3 (Small 14.7): +0.5 / +2–5 / +8–10, i.e. 1 / 4–10 / 15–19% of the gap to Giant | Sec. 6 | 12–20; 5–7 | Capacity gap; adapters inherit CC BY-NC | ≥ +2 AUC@3 on HiRoom and ≥ +1.5 over a control trained on its own outputs | 4/4 |
| E3 posed DA3 → unposed LoRA | Held-out KITTI drives: ATE/RPE −5 / −20–35 / −50%; AbsRel vs LiDAR 0 / −3–8 / −15%. Sec. 6 with the LoRA on: −0–2 AUC@3 | KITTI (outside Sec. 6) + HiRoom no-regression | 10–15; 5–8 | Conditioned depth no better out of domain (Tab. 7) | Unposed vs pose-conditioned DA3 on 2 drives (~1 GPU-h) | 3/3 |
| B4 compressed ← uncompressed | Recovers 0 / 30–50 / 70% of the compression loss | Accuracy–speed Pareto | 10–15 (+ K4) | No headroom (Spark3R is lossless) | Compression curve | 2/2 |

Verdict:
- Build the platform once.
- Do E1 first, E3 second. Phase 3 later discarded E1 (§3.3).

### 2.3 Findings across clusters
- **One E_c noise probe gates many ideas** (1–3 GPU-h):
  - what it gates: K1, K2, K3, the E_c residual (K5), C2(ii), the zoom cameras of H4, and E3's teacher;
  - it must read D_C and the ray map directly, because in posed mode the released API and our `bench_adapter.py` return the input extrinsics.
- **One prediction dump** serves all the K5 tests and the K6 baseline. It holds head poses, ray maps with confidence, depth with confidence, and the poses from both solvers.
- **One uniform-resolution curve** (504 / 378 / 280 / 252 px, no new code) gates A3, B2 and H4.
- **Backpropagation through frozen DA3 plus LoRA** is one shared platform for K8, E2, F1(b), F3(c), H3 and D5.
- **Fix one pose solver first.** Our DLT and the official RANSAC differ by 0.3–0.6° in rotation, which is worth a few AUC@3 points. Run the [official] baseline with the solver held fixed before attributing any gain.
- **Cheaper evaluation:** predict in GPU sessions, score TSDF/Chamfer in CPU-only sessions (≈ 3–5× fewer GPU-h).
- **Budget:** each cluster needs about one Kaggle week (15–35 GPU-h). All eight do not fit, so a selection is unavoidable.

---

## 3. Phase 3: novelty check

### 3.1 Criteria
- **Published** (the core mechanism exists on any feed-forward 3D model) → discard.
- **Partially covered** → keep only if the unpublished part is the core, and re-scope the idea to it.
- **Novel** → keep.

The search ran on 2026-11-19 (arXiv, Hugging Face, GitHub). "Conf." is the agent's confidence, which depends on how much of each paper it read.

### 3.2 Verdict for each idea
| Idea | Verdict | Closest prior work | What is still new | Conf. | Survives? |
|---|---|---|---|---|---|
| A1 | Partially covered | ScaRF-SLAM (SLAM poses into DA3's pose input); 2603.20428 (BA from VGGT-Ω / π³-Long poses); the official DA3 `colmap` mode accepts poses | Iterating DA3 → BA → E_c to a fixed point, with DA3 as its own BA initialiser; a DA3-specific noise-response curve (the protocol exists in Pow3R / WorldMirror). One BA → E_c pass is effectively published | Med | Yes, weak: must beat one BA → conditioned pass |
| A2 | Partially covered | GeoCond (trained reliability head, 6 backbones, not DA3); RobustVGGT (distractor rejection) | Training-free signals from DA3's two pose decoders (head-vs-ray gap, ray-origin spread, pinhole residual); per-view head/ray choice | Med | Yes, merged with C1 + F3 |
| A3 | Partially covered | HD-VGGT (trained low-res global + high-res branch) | Training-free resolution split; anchoring through a prior input is published (GeoFF3D, DAVIO). Thin | Med | No |
| A4 | Published | MapAnything (optional poses for a subset of views); MultiLoc (pose tokens for posed references + zero token for the query) | Only zero-shot transfer to DA3 | High | No |
| B1 | Published | Eco3R (training-free pinned keyframes); GeoFF3D (chunk priors through the prior input); DAVIO (pose-conditioned DA3 windows) | Pose tokens instead of a KV cache = variant | Med | No |
| B2 | **Novel** | Closest: TurboVGGT, PrePARE (uneven token budgets through learned sparsity / pruning); HD-VGGT | Native mixed per-view resolutions (focus vs context views) in one training-free pass | Med | **Yes** |
| B3 | Partially covered | Block-Sparse Global Attention; VGGT-Ω (register-token relays); FlashVGGT; view partitioning (2607.01885) | View-level kNN mask + camera-token relays. Thin | Med | No |
| B4 | Published | PrePARE (trained light modules on frozen VGGT/π³/DA3 recover the pruning loss) | Only the LoRA form differs | Med | No |
| C1 | Partially covered | As A2; Calibration Audit (no DA3, fitted with GT) | **E_c fixed-point residual (not found anywhere)**; auditing DA3 | Med | Yes, merged with A2 |
| C2 | Partially covered | VGGT-360 (training-free perspective slicing); Fisheye3R; Any3R | (i) = F4; (ii) rotation-only views through E_c = ordinary use of known poses | Med | No ((i) folded into F4) |
| C3 | Published | Sky heads of DA3MONO / DA3METRIC; head distillation is standard | — | High | No |
| C4 | Partially covered | Sea-thru; SeaThru-NeRF; Wat3R and WAT3R (trained underwater feed-forward models) | Training-free restore ↔ re-infer loop around frozen DA3, with a scattering fit constrained across views | Med | Yes, weak |
| D1 | Published | Glob3R (matching head on frozen Pi3X + tracks + BA); Selfi | The BA → E_c step = A1 | High | No |
| D2 | Published | Surflo (normal head on frozen VGGT); WorldMirror; Dens3R | — | High | No |
| D3 | Published | Mirror3D; glass detection with VGGT priors (2608.26752) | Multi-view phantom removal, minor | Med-High | No |
| D4 | Published | AMB3R; G3T; CalibAnyView (gravity) | — | High | No |
| D5 | Published | CUT3R (ray-map queries for unobserved regions) | Frozen DA3 + generator = transfer | High | No |
| E1 | Published | Distill3R; OTT3R (π³ → 102M student); FlashSSM-3D | LoRA on a smaller sibling = variant | High | No |
| E2 | Published | Pow3R, MapAnything, OmniVGGT, Pi3X; WorldMirror (prior dropout, noisy-prior tests); GeoFF3D | Adapter-only through DA3's camera token = transfer | High | No |
| E3 | Partially covered | Free Geometry (a privileged full-view pass as teacher of a LoRA on DA3); TCO (pose priors as test-time constraints, optimises weights) | **DA3's own pose-conditioned pass as teacher of an unposed LoRA student; poses needed only at training time** | Med | **Yes** |
| E4 | Published | Fisheye3R (tokens on frozen VGGT/π³/MapAnything); Any3R; Wid3R | — | High | No |
| F1 | Partially covered | TCO (optimises weights under pose constraints); variant (a) = A1 | **Variant (b): gradient optimisation of only the E_c camera inputs through frozen DA3, not found** | Med | **Yes**, variant (b) |
| F2 | Partially covered | Calibration Audit (power-law σ(c), per dataset, fitted with GT) | **Per-scene, GT-free triple-collocation fit; the same estimator on pose sources** | Med | **Yes** |
| F3 | Partially covered | GeoCond; Free Geometry, Self-Geometry, Test3R | Head–ray fusion = A2; head–ray agreement as a test-time loss is thin | Med | Only merged into A2/C1 |
| F4 | Partially covered | Dense-match calibration with radial distortion and VGGT init (2512.15608); CalibAnyView | Black-box distortion search that maximises frozen-DA3 self-consistency; value reduced by any-camera models | Med | Yes, weak |
| G1 | Published | Official DA3 confidence-weighted RANSAC; Cameras-as-Rays (least-squares camera from rays); IRLS / Laplace are textbook | — | Med | No |
| G2 | Published | AnyCalib; CalibAnyView | — | Med-High | No |
| G3 | Partially covered | As A1; noisy-prior training and evaluation (WorldMirror, GeoFF3D) | Same as A1 | Med | Merged into A1/F1 |
| G4 | Published | GeoFF3D (shared-view estimates as next-chunk priors); MapAnything; DAVIO | Pose vs depth priors = variant | Med | No |
| G5 | Published | AMB3R (scale head on frozen VGGT); UniScale; DA3-Metric's canonical focal | — | High | No |
| H1 | Partially covered | VIDAR, DAVIO, ScaRF-SLAM (external poses into DA3); MapAnything, GeoFF3D, WorldMirror (trained partial priors); GeoCond (gate) | Training-free fill-in + trust gate. Thin | Med | No |
| H2 | Partially covered | Click3R (clicks correct a stereo feed-forward model through a trained adapter) | Training-free N-view propagation through E_c = the A1/F1 mechanism | Med | No |
| H3 | Published | OmniVGGT (zero-init depth adapter); MapAnything, WorldMirror, Pi3X; PromptDA; CAPA | — | High | No |
| H4 | Partially covered | HD-VGGT (trained); PatchFusion, BoostingDepth (monocular tiles); Perspective Crop Layers | Rotated virtual zoom views from the full-resolution original, conditioned through E_c together with the scene views | Med-Low | Yes, weak |
| H5 | Published | GeoFF3D; DAVIO (two shared frames per pose-conditioned DA3 window); HiP3R | — | Med | No |

Main search queries behind the "novel / not found" verdicts:
- **B2:** "mixed/heterogeneous resolution views VGGT/π³/DA3 single pass"; "adaptive per-view resolution token budget".
- **F1(b):** "optimize input camera poses through frozen pose-conditioned network"; "camera token test-time optimisation".
- **F2:** "triple collocation depth uncertainty"; "GT-free per-scene confidence calibration feed-forward 3D".
- **E3:** "self-distillation pose-conditioned teacher pose-free student"; "DA3 LoRA adaptation with poses, no depth labels".
- **A2/C1:** "DA3 camera head vs ray map disagreement reliability"; "Depth Anything 3 uncertainty/failure detection".

### 3.3 Discarded ideas and the work that already implements them
| Idea | Implemented by |
|---|---|
| A3 coarse global + fine chunks | HD-VGGT (2603.27222); anchoring through a prior input: GeoFF3D (2608.28288), DAVIO (2609.27702) |
| A4 partially-posed inference | MapAnything (2509.13414); MultiLoc (2603.27170) |
| B1 skeleton-anchored long video | Eco3R (2611.18099); GeoFF3D; DAVIO |
| B3 covisibility-sparse attention | Block-Sparse Global Attention (2509.07120); VGGT-Ω (2605.15195); FlashVGGT (2512.01540); view partitioning (2607.01885) |
| B4 efficiency LoRA | PrePARE (2605.08371) |
| C2 fisheye / 360° | VGGT-360 (2603.18943); Fisheye3R (2603.28896); Any3R (2610.36150) |
| C3 sky head | Sky heads of DA3MONO / DA3METRIC (DA3 release) |
| D1 correspondence head + BA | Glob3R (2607.09225); Selfi (2512.08930) |
| D2 normals + sky | Surflo (2606.13644); WorldMirror (2510.10726); Dens3R |
| D3 mirror / glass | Mirror3D (2106.06629); 2608.26752 |
| D4 gravity + metric scale | AMB3R (2511.20343); G3T (2605.27372); CalibAnyView (2605.14615) |
| D5 query views without images | CUT3R (2501.12387) |
| E1 Giant → Small/Base distillation | Distill3R (2602.00865); OTT3R (2609.36374); FlashSSM-3D |
| E2 partial-prior adapter | Pow3R (2503.17316); MapAnything; OmniVGGT (2511.10560); Pi3X; WorldMirror; GeoFF3D |
| E4 fisheye through the ray map | Fisheye3R; Any3R; Wid3R (2602.05321) |
| G1 robust ray solver | Official DA3 confidence-weighted RANSAC; Cameras-as-Rays (2402.14817) |
| G2 self-calibration beyond pinhole | AnyCalib (2503.12701); CalibAnyView |
| G4 anchor-conditioned chunking | GeoFF3D; MapAnything; DAVIO |
| G5 metric-scale head | AMB3R; UniScale (2602.23224); DA3-Metric |
| H1 partial / noisy priors | VIDAR (2607.17171); DAVIO; ScaRF-SLAM (2606.00307); MapAnything; WorldMirror; GeoCond (2609.18465) |
| H2 click-to-fix | Click3R; propagation through E_c = A1/F1 |
| H3 sparse metric anchors | OmniVGGT; MapAnything; PromptDA (2412.14015); CAPA (2602.14751) |
| H5 chunk chaining | GeoFF3D; DAVIO; HiP3R (2610.31635) |

### 3.4 Survivors and what we may claim
| Survivor family | Ideas | May claim (not found in the literature) | Must cite, not claim |
|---|---|---|---|
| DA3 Self-Audit → **#1** | A2, C1, F3, F2 | DA3-specific redundancy signals (head-vs-ray gap, ray-origin spread, pinhole residual of the ray field); the **E_c fixed-point residual**; per-view choice between head and ray poses; **GT-free per-scene triple-collocation calibration** of depth and of pose sources; the first reliability audit of DA3 | Trained reliability heads (GeoCond); distractor rejection (RobustVGGT); perturbation variance (ReVar3R); GT-fitted calibration audits (2608.29705); the ray-to-camera solver (official RANSAC, Cameras-as-Rays) |
| Foveated DA3 → **#2** | B2 (+ H4) | **Native mixed per-view resolutions in one training-free pass** (the only fully novel idea); virtual zoom cameras through E_c as its region-level use (weak on its own) | Token merging and pruning (Spark3R, PrePARE, TurboVGGT); trained high-res branches (HD-VGGT); monocular tiling (PatchFusion, BoostingDepth); perspective crops |
| Pose-conditioned self-distillation → **#3** | E3 | **DA3's own pose-conditioned pass as teacher of the unposed model; poses only at training time** | Privileged-view teachers for test-time adaptation (Free Geometry); pose priors as test-time constraints (TCO); plain fine-tuning with pose supervision |
| E_c camera-input optimisation → runner-up | F1(b), A1, G3 | **Gradient search over E_c's camera inputs through frozen DA3**; iterating the loop to a fixed point (weak) | Feeding external or BA poses into DA3 (ScaRF-SLAM, DAVIO, VIDAR, official `colmap` mode); BA from feed-forward poses (VGGT + BA, 2603.20428) |
| Self-calibrating undistortion → runner-up | F4 (+ C2(i)) | Distortion search that maximises frozen-DA3 self-consistency (weak) | Fisheye / any-camera models (Fisheye3R, Any3R, Wid3R); calibration (AnyCalib, CalibAnyView, 2512.15608) |
| Haze / underwater loop | C4 | Training-free restore ↔ re-infer loop with a cross-view scattering fit (weak) | Sea-thru, SeaThru-NeRF; trained underwater models (Wat3R, WAT3R) |

### 3.5 Findings that change our plans
- **Feeding external poses into DA3 through E_c is published.**
  - The official DA3 `colmap` mode accepts poses.
  - ScaRF-SLAM feeds SLAM poses into DA3's pose input and reports a gain over no pose input.
  - DAVIO conditions DA3 windows on VIO poses.
  - VIDAR feeds VIO poses into DA3-Streaming and finds injection *worse* than decoupled alignment on EuRoC (F@0.10: 0.463 vs 0.676).

  Any "pose-conditioned second pass" is therefore no longer a contribution. This includes C1 in [DYNAMIC_SCENES.md](DYNAMIC_SCENES.md), which has been updated.
- **The API hides what DA3 does with input poses.** In posed mode DA3's API overwrites the returned poses with the input poses, which ScaRF-SLAM and DAVIO also note; our `bench_adapter.py` does the same. Any test of E_c must read D_C or the ray map directly.
- **The "copy" risk is real.** SpatialBench (2605.27367) reports that DA3-Giant adheres closely to injected GT poses.
- **BA from feed-forward poses is published** (VGGT + BA). 2603.20428 runs BA from VGGT-Ω and π³-Long poses; DA3 is only a baseline there, contrary to a Phase 1 flag.
- **Efficiency on DA3 is crowded.**
  - Spark3R: training-free token reduction including DA3, ~17× at 1000 frames.
  - PrePARE: trained restoration on frozen VGGT/π³/DA3.
- **Test-time adaptation of DA3 is published.**
  - Free Geometry: LoRA where a full-view pass teaches partial views.
  - Self-Geometry: LoRA with correspondence and epipolar losses on DA3-S/B/L/G.
- **Reliability and calibration work exists for VGGT-type models, but none of it uses DA3.** Examples: GeoCond (6 backbones), Calibration Audit (7 backbones), ReVar3R, RobustVGGT, and a failure probe on VGGT (2610.35806). This is the gap #1 fills.

### 3.6 References
The read level is what the novelty agent actually read: abstract, HTML (full text or an excerpt), search snippet, or model card.

**DA3 and pose input**
- Depth Anything 3 (Lin et al., 2025): https://arxiv.org/abs/2511.10647 (the paper; project page depth-anything-3.github.io).
- ScaRF-SLAM (Zhang, RA-L 2026): https://arxiv.org/abs/2606.00307 (abstract + HTML).
- DAVIO (Mahmoud, Sep 2026): https://arxiv.org/abs/2609.27702 (abstract + HTML).
- VIDAR (Salih, Jul 2026): https://arxiv.org/abs/2607.17171 (abstract + HTML).
- VI3, IMU metric anchoring + refinement incl. DA3 (Lozano, Sep 2026): https://arxiv.org/abs/2609.03824 (abstract + snippet).
- SpatialBench (May 2026): https://arxiv.org/abs/2605.27367 (HTML excerpt).
- Benchmarking Pose Strategies for NVS (Meza, 2026): https://arxiv.org/abs/2603.20428 (abstract + HTML).

**Test-time adaptation and distillation**
- Free Geometry (Dai, Apr 2026): https://arxiv.org/abs/2604.14048 (abstract).
- Self-Geometry (Youn, Aug 2026): https://arxiv.org/abs/2608.10708 (abstract).
- TCO (Zhou, CVPR 2026): https://arxiv.org/abs/2604.03878 (abstract).
- Test3R (Yuan, 2025): https://arxiv.org/abs/2506.13750 (abstract).
- Distill3R (Leblanc, CRV 2026): https://arxiv.org/abs/2602.00865 (abstract).
- OTT3R (Sep 2026): https://arxiv.org/abs/2609.36374 (snippet).

**Priors, partial poses, long sequences, high resolution**
- MapAnything (Keetha, 3DV 2026): https://arxiv.org/abs/2509.13414 (abstract + HTML).
- Pow3R (Jang, CVPR 2025): https://arxiv.org/abs/2503.17316 (abstract).
- OmniVGGT (Peng, 2025): https://arxiv.org/abs/2511.10560 (abstract).
- WorldMirror (Hunyuan, ICML 2026): https://arxiv.org/abs/2510.10726 (snippets).
- Pi3X (π³ team, 2025): https://huggingface.co/yyfz233/Pi3X (README snippet).
- MultiLoc (Dang, 2026): https://arxiv.org/abs/2603.27170 (HTML).
- GeoFF3D (Yang, Aug 2026): https://arxiv.org/abs/2608.28288 (full text, pp. 1–5).
- Eco3R (Cao, Nov 2026): https://arxiv.org/abs/2611.18099 (abstract).
- HiP3R (Xu, Oct 2026): https://arxiv.org/abs/2610.31635 (abstract).
- KV-Tracker (Taher, 2025): https://arxiv.org/abs/2512.22581 (abstract).
- SAIL-Recon (Deng, 2025): https://arxiv.org/abs/2508.17972 (snippets).
- VGG-T³ (Elflein, CVPR 2026): https://arxiv.org/abs/2602.23361 (abstract).
- HD-VGGT (Chen, Mar 2026): https://arxiv.org/abs/2603.27222 (abstract).
- CAPA (Ke, Feb 2026): https://arxiv.org/abs/2602.14751 (abstract).
- PromptDA: https://arxiv.org/abs/2412.14015 (cited by the Phase 1 agent).

**Reliability and calibration**
- GeoCond (Ahmedt-Aristizabal, Sep 2026): https://arxiv.org/abs/2609.18465 (abstract + HTML).
- RobustVGGT (Han, 2025): https://arxiv.org/abs/2512.04012 (abstract).
- ReVar3R (Mahdi, Oct 2026): https://arxiv.org/abs/2610.07883 (abstract).
- Calibration Audit (Deng, Aug 2026): https://arxiv.org/abs/2608.29705 (abstract + HTML).
- Does VGGT Know What It Does Not Know? (Srinivasan, Oct 2026): https://arxiv.org/abs/2610.35806 (abstract).
- Cameras as Rays: https://arxiv.org/abs/2402.14817.

**Efficiency**
- Spark3R (Tang, May 2026): https://arxiv.org/abs/2605.06270 (abstract).
- PrePARE (Li, May 2026): https://arxiv.org/abs/2605.08371 (abstract).
- TurboVGGT (Huang, NeurIPS 2026): https://arxiv.org/abs/2605.14315 (abstract).
- Block-Sparse Global Attention (Wang, CVPR 2026): https://arxiv.org/abs/2509.07120 (abstract).
- AVGGT (Sun, CVPR 2026): https://arxiv.org/abs/2512.02541 (abstract).
- VGGT-Ω (Wang, CVPR 2026): https://arxiv.org/abs/2605.15195 (snippets).
- FlashVGGT: https://arxiv.org/abs/2512.01540.
- View partitioning: https://arxiv.org/abs/2607.01885.

**Cameras, distortion, wide field of view**
- AnyCalib (Tirado-Garín, ICCV 2025): https://arxiv.org/abs/2503.12701.
- CalibAnyView (Li, May 2026): https://arxiv.org/abs/2605.14615 (snippets).
- Robust Multi-view Calibration from Dense Matches (Hägerlind, 2025): https://arxiv.org/abs/2512.15608 (abstract).
- Fisheye3R (Duan, ECCV 2026): https://arxiv.org/abs/2603.28896 (HTML).
- Any3R (Chen, Oct 2026): https://arxiv.org/abs/2610.36150 (abstract + HF).
- Wid3R (Jung, Feb 2026): https://arxiv.org/abs/2602.05321 (abstract).
- VGGT-360 (Yuan, Mar 2026): https://arxiv.org/abs/2603.18943 (abstract).
- Perspective Crop Layers: https://arxiv.org/abs/2011.13607.

**Heads, scale, other**
- AMB3R (Wang, CVPR 2026): https://arxiv.org/abs/2511.20343 (HTML).
- G3T (Nagoor Kani, May 2026): https://arxiv.org/abs/2605.27372 (abstract).
- UniScale: https://arxiv.org/abs/2602.23224.
- Glob3R (Deng, NeurIPS 2026): https://arxiv.org/abs/2607.09225 (snippets).
- Selfi (Deng, 2025): https://arxiv.org/abs/2512.08930 (abstract).
- Surflo (Guédon, Jun 2026): https://arxiv.org/abs/2606.13644 (HF card).
- Mirror3D (Tan, CVPR 2021): https://arxiv.org/abs/2106.06629 (snippet).
- Glass detection grounded in 3D geometry (Lu, ACM MM 2026): https://arxiv.org/abs/2608.26752 (abstract).
- CUT3R (Wang, CVPR 2025): https://arxiv.org/abs/2501.12387 (snippets).
- Click3R (Gu, NeurIPS 2026): https://huggingface.co/uva-cv-lab/Click3R (model card / poster).
- Wat3R / WAT3R (Ren / Zheng, Jul 2026): https://arxiv.org/abs/2607.08772, https://arxiv.org/abs/2607.21023 (snippets).
- SeaThru-NeRF (Levy, CVPR 2023): https://arxiv.org/abs/2304.07743 (snippet).
- Cited by name only (no ID checked): Sea-thru (CVPR 2019), PatchFusion, BoostingDepth, Dens3R, FlashSSM-3D, GeoCalib.

---

## 4. The three selected ideas

### 4.1 How they were chosen
1. **Eligibility.** Only the survivors of the novelty check are eligible: 12 ideas, merged into 6 families (§3.4).
2. **Scores.** Each family gets three scores from 1 to 5:
   - expected impact and feasibility come from Phase 2b;
   - novelty comes from Phase 3: 5 = novel; 4 = core not found and several claimable parts; 3 = core not found but close prior work; 2 = weak survivor.
3. **Ranking.** By the sum of the three scores. Ties are broken first by novelty strength, because under the strict rule one new paper can wipe out a weak survivor. The second tie-breaker is fit with the plan: training-free work plus one or two LoRA sessions on Kaggle.

| Family | Impact | Feasibility | Novelty | Sum | Route | Decision |
|---|---|---|---|---|---|---|
| DA3 Self-Audit (A2 + C1 + F3 + F2) | 3 | 5 | 4 | **12** | training-free | **#1** |
| Foveated DA3 (B2 + H4) | 2 (3 if the zoom variant works) | 3 | 5 | **10** | training-free | **#2** |
| Pose-conditioned self-distillation (E3) | 3 | 3 | 3 | **9** | LoRA, 1–2 sessions | **#3** (ties with F4; wins on novelty and plan fit) |
| Self-calibrating undistortion (F4) | 3 | 4 | 2 | 9 | training-free | runner-up |
| E_c camera-input optimisation (F1(b) + A1/G3) | 2 | 2 | 4 | 8 | training-free | runner-up |
| Haze / underwater loop (C4) | 2 | 3 | 2 | 7 | training-free | not selected |

#2 and #3 are close:
- Foveated DA3 ranks higher because its novelty is the strongest.
- E3 has the higher expected impact and is the only surviving way to use the planned training sessions.
- Weighting impact more would swap them.

### 4.2 #1 DA3 Self-Audit: reliability, pose-source choice and calibrated uncertainty without ground truth

> **Status (2026-11-22): implemented, waiting for the Kaggle run.** See notebook `05_self_audit`, [ENHANCEMENTS_RUNBOOK.md](ENHANCEMENTS_RUNBOOK.md) and the results report [reports/enhancement_1_self_audit.md](../reports/enhancement_1_self_audit.md). Changes from the plan below:
> - pose choice and fusion use the official RANSAC solver, as planned; the custom IRLS solver (G1) is not used because it is published;
> - pose triple collocation is left for later, because it needs a third pose source that does not share the backbone;
> - triple collocation uses plain variances with per-triplet outlier trimming, because MAD-based variances are biased for the mixed-confidence partners.

**Idea.** DA3's camera estimates are redundant, and the disagreement between the redundant copies is a free signal of how reliable the prediction is.
- DA3 predicts every camera twice: once with the camera head D_C and once through the ray map.
- The ray map has 6 numbers per pixel, far more than an 11-DOF pinhole camera needs.
- DA3 can also take its own cameras back in through E_c.

VGGT and π³ have neither two camera decoders nor a pose input, so this is specific to DA3.

**How it works.**
1. **Per-view signals, no training:**
   - (a) head-vs-ray pose gap: rotation angle, normalised distance between camera centres, FOV ratio;
   - (b) spread of the ray origins, which should all be the same point (Eq. 1);
   - (c) pinhole residual of the ray field after the homography fit (Eq. 2);
   - (d) statistics of the depth and ray confidences;
   - (e) cross-view depth-reprojection agreement;
   - (f) **E_c fixed-point residual:** feed DA3's own cameras back through E_c and measure how much depth and rays move (one extra pass).
2. **Failure detector.** A logistic or isotonic model on these signals, fitted on one dataset and tested on the other four to check that it transfers. Flagged views can be dropped and DA3 re-run.
3. **Per-view pose source.** For each view, choose or fuse the head pose and the official ray-map pose (RANSAC solver), weighted by the audit. This is the only part expected to move a Tab. 2 number.
4. **Calibrated uncertainty without GT (F2).**
   - Per scene, fit σ = a·c^(−b) to DA3's depth confidence c by triple collocation: a surface point seen in three views gives three pairwise residuals, which separate each view's error variance.
   - The same estimator on three pose sources (head, ray, depth-ICP) gives per-view pose error bars, which also serve as fusion weights in step 3.

**Expected results** (Phase 2b estimates, pessimistic / realistic / optimistic):
- **Gross pose failures (> 15°):** AUROC 0.62 / 0.75–0.85 / 0.9. Moderate errors (> 5°) are near chance for Small, because most of its view pairs are already above 5°.
- **Depth error ranking (AUSE)** vs raw confidence: 0 / −10–20% / −30%.
- **Per-view pose source:** AUC@3 vs the better official source −0.5 / +1–3 / +5.
  - That estimate included a custom solver, which is published (G1 discarded).
  - With the official solvers, expect the lower end.
- **Per-scene coverage error of 90% intervals:** none / 5–12 / < 5 points, against 10–25 points for a global GT-fitted calibration.
- **E_c residual:** ΔAUROC 0 / +0.02–0.05 / +0.08 on top of the other signals.

**First experiments (go/no-go).** One dump of DA3-Small outputs on HiRoom, ETH3D and 7Scenes (~1 GPU-h); everything after that runs on CPU.
- **Detector: go** if Spearman(head–ray gap or pinhole residual, true rotation error) ≥ 0.3 and AUROC ≥ 0.7 on at least two datasets.
- **Pose-source choice: go** if a per-view oracle beats the better single source by ≥ 2 AUC@3, or the fusion gains ≥ 0.5 on 3 of 5 datasets.
- **F2: go** if the estimated σ divided by the true σ lies within 0.7–1.4 in each confidence bin on 5 HiRoom scenes.
- **E_c residual:** only after the noise probe (§5.4); **go** if ΔAUROC ≥ 0.03.

**Cost.**
- **Core** (dump, detector, pose-source choice, F2): about 5–8 GPU-h.
- **Extensions:** with the E_c residual (one extra pass per scene) and the distractor protocol, 10–15 GPU-h.
- **Development:** 13–19 days in total (Phase 2b), parallelisable.

**Risks.**
- **Shared failures.** Head and ray share the backbone, so they can be wrong the same way (symmetric scenes, low overlap) and their gap stays small.
  - Mitigation: the cross-view reprojection signal is less tied to the shared features.
  - Report the AUROC for each dataset; treat a spread above 0.15 as a warning.
- **Dataset leakage.** The detector may learn the dataset instead of the error.
  - Mitigation: fit on one dataset, test on the others.
- **Shared errors are invisible to triple collocation** (scale drift, bending), so σ comes out too low.
  - Mitigation: check against GT on HiRoom.
- **Noise.** The pinhole residual is noisy on the low-resolution ray map.

**Why #1:**
- the highest feasibility of all the survivors;
- a capability DA3 lacks (knowing when it is wrong), plus a chance at a Tab. 2 gain;
- three parts not found anywhere: the head-vs-ray signals, the E_c residual and triple collocation;
- it produces the dump and the signals that #2 and #3 reuse.

### 4.3 #2 Foveated DA3: mixed per-view resolutions in one pass

> **Status (2026-11-22): implemented, waiting for the Kaggle run.** See notebook `06_foveated`, [ENHANCEMENTS_RUNBOOK.md](ENHANCEMENTS_RUNBOOK.md) §3 and the results report [reports/enhancement_2_foveated.md](../reports/enhancement_2_foveated.md). Changes from the plan below:
> - the routing is checked to be bit-identical to the original at equal resolution (unit tests on CPU, GPU check in the notebook) before any experiment;
> - added a training-free mitigation of the cross-scale risk: attention-mass balancing (log(N_focus / N_view) on the context views' keys in the cross-view blocks, equivalent to replicating their tokens), tested in the probe next to 280-px context views;
> - region level: the zoom views go through E_c with DA3's own cameras (practical) and with GT cameras (oracle); the gate uses DA3's own cameras;
> - the Spark3R comparison at equal FLOPs is not run (no public code in this repo); the large-view protocol compares against uniform resolution and fewer views at equal tokens;
> - the gates below are made precise in a pre-registered analysis plan ([FOVEATED_ANALYSIS_PLAN.md](FOVEATED_ANALYSIS_PLAN.md)): scene-bootstrap CIs, PASS / FAIL / INCONCLUSIVE decisions, margins raised only by a treatment-blind A/A noise floor; gate 1's "collapse" is tested directly as "focus views with vs without the context views".

**Idea.** DA3 runs every view at the same resolution, so the token count grows with views × resolution and the cross-view cost grows quadratically with it. Instead, give a few *focus* views full resolution and many *context* views a quarter of the tokens, all in **one** forward pass. There are two uses:
- **View level (B2).** Focus views at 504 px, context views at 252 px. Focus views are the ones we care about, or the ones the audit flags. More views fit on a T4, and runs are faster.
- **Region level (H4).**
  - A region of the full-resolution photo (for example ETH3D's 6048×4032 DSLR images) is warped into a rotated, narrow-FOV virtual pinhole camera. The warp is a pure rotation, so it is exact.
  - That zoom view is processed together with the scene views, with its camera given through E_c. The result is high-resolution depth for the region, already in the scene frame.
  - Since the zoom rotation R_v is known exactly, checking that DA3 recovers it is a free test that DA3 follows E_c.

**Why this is possible in DA3.**
- **Cross-view blocks.** In the released architecture they use *constant* RoPE positions for all patch tokens ([src/da3/backbone/vit.py:178-186](../src/da3/backbone/vit.py#L178-L186)). Cross-view attention therefore does not depend on where a token sits in its image, and it is well defined over views with different token counts.
- **Everything else already works per view:** within-view blocks, the learned position embedding (interpolated for each resolution) and the DPT heads. They simply run once per resolution group.
- **Built-in check.** With all views at the same resolution, the output must be bit-identical to the original.

**Expected results** (Phase 2b):
- **View level** (200–1000 views, 10% focus views):
  - T4 speed: ×3 / ×6 / ×9.
  - Maximum views per T4 pass: ~750 → ~2000 for Small, ~400 → ~1100 for Base.
  - Focus-view AbsRel vs all views at 504 px: +10 / +2 / 0% (positive = worse).
  - Focus-view AUC@30 vs fewer full-resolution views at the same token budget: +0 / +2 / +5.
- **Region level:**
  - Region Chamfer at fine thresholds: 0 / −15 / −35%.
  - Region F1: +0 / +4 / +10.
  - Global Sec. 6 F1: unchanged (±1).

These gains do not show in Tab. 2/3: at the Sec. 6 size (≤ 100 views) a T4 is not limited by token count. Two new protocols are needed:
- 300–1000-view sequences (TUM RGB-D, full 7Scenes);
- region metrics on the ETH3D and DTU originals.

**First experiments (go/no-go).**
1. **Uniform-resolution curve:** 504 / 378 / 280 / 252 px on HiRoom and 7Scenes (~1 GPU-h, no new code; run it next to the baseline).
   - If 252 px is nearly lossless, plain downscaling wins and B2 is moot.
   - If it collapses, context views carry nothing.
2. **Region oracle:** 5 ETH3D scenes, zoom view vs global 504 px vs global 896 px vs a naive crop (~0.5 GPU-h). All views are at 504 px, so no new routing code is needed.
   - **Go** if region AbsRel is ≥ 10% better than global 896 px and R_v is recovered within 1°.
3. **Cross-scale probe** (needs the routing): 16 views, 8 at 504 px + 8 at 252 px, vs all at 504 px vs all at 252 px (~0.5 GPU-h).
   - **Go** if focus-view AbsRel is within 3% of all-504 and context-view poses are at least as good as all-252.

**Cost.**
- **GPU:** 6–12 GPU-h.
- **Development:** 5–9 days for the variable-length routing, the per-group DPT and the zoom warps, plus 1–2 days for optional text-prompted regions.

**Risks.**
- **Cross-scale token matching was never trained.** DA3 saw one resolution per batch, with the short side ≥ 280 px.
  - Mitigation: probe 3; try 280 px context views instead of 252 px.
- **Reference-view bias.** The released reference-view choice (cls statistics) may favour one resolution.
  - Mitigation: fix the reference to a focus view.
- **Uniform 336 px may tie** at the same token budget. That would still be a reportable finding.
- **The efficiency bar is Spark3R** (~17× on DA3 at 1000 frames, accuracy unchanged).
  - Compare at equal FLOPs and test stacking the two.
  - The claim is "detail where needed", not raw speed.
- **Zoom views are narrow-FOV with zero baseline,** so their detail is mostly monocular. A global 896-px pass may capture most of DTU's gain.

**Why #2:**
- the only idea judged fully **novel**;
- training-free;
- it uses an architectural property of DA3 that is easy to show at the exam;
- the region variant gives a visible demo: sharp detail where you zoom.

It ranks below #1 because its gains do not show on the paper's metrics and its core depends on probe 3.

**Data to provide:**
- full-resolution ETH3D DSLR and DTU originals, if DA3-BENCH only ships downscaled images (check first);
- two or three long TUM RGB-D sequences (public, ~0.5–1.5 GB each).

### 4.4 #3 Pose-conditioned self-distillation: adapt DA3 to a new domain without depth labels

**Idea.** In many domains poses are cheap but dense depth is not: driving (GNSS-INS), robots (kinematics), any video with COLMAP poses.
- **Teacher:** frozen DA3 in its posed mode, i.e. conditioned on the known poses through E_c. Its outputs are cached once.
- **Student:** the same DA3 run unposed, plus a LoRA on blocks 4–11 (rank 8–16).
- **Losses:**
  - L_C / L_M against the known poses;
  - confidence-weighted L_D + L_grad against the teacher's depth;
  - an anchor to the frozen outputs, against forgetting.

At test time the student needs no poses. With the LoRA switched off, DA3 is bit-identical to the release.

**Why DA3.** Only DA3 has a posed mode that can act as the teacher: VGGT and π³ take no poses. The closest published work, Free Geometry, uses a different privileged signal (more views) and adapts per scene at test time.

**Domain.** KITTI is the suggested first domain, because Tab. 1 has no real driving data (only vKITTI2). OXTS gives the poses, and LiDAR gives depth for evaluation only.
- **Split:** train on a few drives, test on held-out drives.
- **Aspect ratio:** KITTI images are 3.3:1, while DA3 trained up to ~1.8:1 (896×504), so crop or tile them.
- **Calibration:** OXTS → camera.
- **Licence:** KITTI is non-commercial (CC BY-NC-SA 3.0), which is fine for the course.

**Expected results** (Phase 2b):
- **Held-out drives:** ATE/RPE −5 / −20–35 / −50%; AbsRel vs LiDAR 0 / −3–8 / −15%.
- **In-domain gap** between unposed and posed DA3: 5 / 20–40 / 60% closed.
- **Sec. 6 with the LoRA on:** −0 to −2 AUC@3 (forgetting check).

**First experiment (go/no-go).** Training-free, ~1 GPU-h. On 2 KITTI drives, compare AbsRel vs LiDAR for unposed and pose-conditioned DA3-Small, and measure the unposed ATE.
- **Go** only if conditioned depth is ≥ 5% better.
- If only the poses have headroom, E3 collapses into ordinary pose-supervised fine-tuning, which is not new.

**Controls** (same data, steps and seeds):
1. LoRA with the pose losses only, no teacher depth. This isolates the novel term.
2. LoRA trained on its own unposed outputs (generic fine-tuning effect).
3. Free Geometry with its public code.
4. LoRA off = bit-exact; Sec. 6 no-regression with the LoRA on.

**Cost.**
- **GPU:** 10–15 GPU-h in total:
  - teacher caching ~0.5 h;
  - training 4–6 h (1–2 Kaggle sessions);
  - evaluation 2–3 h;
  - the Free Geometry subset 2–4 h.
- **Development:** 5–8 days, including the LoRA platform (already a TODO item).
- **Data:** 10–15 GB of KITTI raw drives. KITTI requires registration on its website; upload the drives as a private Kaggle dataset.

**Risks.**
- **Out of domain, conditioned depth may be no better than unposed depth.** In Tab. 7, conditioning gave +2.9 to +8 F1 on three datasets but −12.4 on 7Scenes. The go/no-go checks this first.
- **Gauge mismatch between teacher and student.** The posed path skips the reference-view selection; the unposed path picks the reference by cls statistics. This silently caps gains.
  - Mitigation: fix the reference view and normalise as in Sec. 3.3.
- **Forgetting outside the domain.** Measure it on HiRoom.
- **Scale.** Metric vs normalised scale must be handled consistently.
- **E_c may simply copy its input poses** (to be shown by the noise probe). Then the teacher only adds depth made consistent with the known poses. That is still useful, but must be reported as such.

**Why #3:**
- the only surviving fine-tuning idea, so it covers the planned one or two training sessions;
- the highest expected impact among the remaining families (tied with F4);
- a clear, testable novelty claim: the pose-conditioned teacher.

It ranks third because its gains lie outside the paper's benchmark and its novelty depends on the depth term.

### 4.5 Runners-up and why they were not chosen
- **E_c camera-input optimisation (F1(b), with the iterated A1/G3 loop).** Gradient search over the cameras given to E_c, so that DA3's conditioned outputs agree across views. It is DA3-specific and not found in the literature.
  - Not chosen, for three reasons:
    1. Backpropagating through DA3 on 16 GB limits it to ~24 views or 336 px, at ~15 min per scene.
    2. SpatialBench's finding that DA3 adheres to injected poses suggests the objective may be flat: depth bends to fit any pose.
    3. It must beat published baselines: BA plus one conditioned pass, and test-time LoRA (Free Geometry, Self-Geometry).
  - Promote it if both conditions hold:
    - the E_c noise probe shows a slope below 0.7, i.e. DA3 partly denoises its input;
    - along GT-perturbation directions, the loss minimum is closer to GT than the start in at least 4 of 5 scenes.
- **Self-calibrating undistortion (F4).** Feasible and useful (fisheye and action cameras), but weakly novel in a crowded area: Fisheye3R, Any3R, Wid3R, AnyCalib, CalibAnyView, 2512.15608.
  - The decisive baseline, single-image calibration → undistort → DA3, may simply win.
  - It does not move Tab. 2/3.
- **Haze / underwater loop (C4).**
  - DA3 may already be robust to haze.
  - Synthetic fog rendered with GT depth flatters the loop (inverse crime).
  - The novelty margin is thin.

---

## 5. How the three ideas interact

### 5.1 Shared building blocks (build once)
| Block | Used by | Notes |
|---|---|---|
| [official] baseline run with one fixed pose solver | all three | Step 0, notebook 03_eval; DLT and RANSAC differ by 0.3–0.6° |
| Prediction dump (head poses, ray maps + confidence, depth + confidence, poses from both solvers) | #1 (all its tests), #2 (uniform-resolution curve), #3 (unposed baseline) | Written in GPU sessions, scored in CPU-only sessions |
| Posed re-pass that reads D_C and the ray map directly | #1 (E_c residual), #2 (zoom views, R_v check), #3 (teacher) | In posed mode the API and `bench_adapter.py` return the input extrinsics |
| E_c noise probe | gates #1's E_c residual, #2's zoom cameras, #3's teacher, and the runner-up | The protocol is published (Pow3R, WorldMirror); the DA3 curve is new data |
| LoRA + backpropagation through frozen DA3 (gradient checkpointing) | #3 (and runner-up F1(b)) | LoRA-off bit-exact test; already in TODO.md |
| Mixed-resolution token routing | #2 | Equal resolutions = bit-exact; reusable for chunked and long-sequence work |

### 5.2 Synergies
- **#1 → #2: the audit tells #2 where to spend resolution.**
  - Upgrade the views it flags to focus views, and zoom on regions with low confidence or high reprojection error.
  - On zoom views, the head–ray gap and the R_v check detect when DA3 does not follow the virtual camera.
- **#1 → #3: the audit screens the teacher's pseudo-labels.**
  - Drop views where the posed teacher disagrees with the known pose, or where the E_c residual is high.
  - The triple-collocation σ can replace the raw confidence as the distillation weight.
- **#3 → #1: KITTI is a natural distribution-shift test for the audit.**
  - Run the audit on KITTI with the LoRA off and on; its calibrator was fitted on DA3-BENCH.
  - F2 gives a GT-free check of whether adaptation improves calibration in the new domain.
- **#2 → #1: a stress test.** The uniform-resolution curve checks whether the audit flags views degraded by low resolution.
- **#2 ↔ #3 (stretch).** KITTI's wide frames could use foveation instead of cropping: full-resolution focus crops plus one low-resolution context view of the whole frame.

### 5.3 Conflicts and how to avoid them
- **The LoRA (#3) changes DA3's statistics.** Confidences and head–ray gaps shift, so an audit calibrator fitted on frozen DA3 becomes stale.
  - Refit it, or rely on the GT-free per-scene F2.
  - Always report LoRA off and on.
- **Resolution vs LoRA.** A LoRA trained at one resolution per batch may change cross-scale matching (#2).
  - Evaluate #2 with the LoRA off; stacking is an extra row only.
- **Pose source.** #1's per-view pose choice changes the poses that evo alignment and TSDF see.
  - Keep the pose source fixed while evaluating #2 and #3 (one change at a time).
  - Report head / official ray / audited poses side by side, with the harness unchanged.
- **Reference view (gauge).** #2 must fix the reference to a focus view, and #3's teacher (posed) and student (unposed) choose the reference differently. Align them explicitly.
- **Protocols.** #2's region and large-view numbers and #3's KITTI numbers come from new protocols.
  - Never mix them into Sec. 6 tables.
  - Report the Sec. 6 no-regression separately.
- **Compute.** #3's training sessions and #2's long sequences compete for the ~30 GPU-h weekly quota. Run all scoring in CPU-only sessions.

### 5.4 Dependencies
```
Step 0  [official] 03_eval baseline, Small/Base, one pose solver fixed ── prediction dump
          │
          ├── E_c noise probe (reads D_C / ray outputs) ──┬──> #1 E_c residual feature
          │                                               ├──> #2 are zoom cameras followed?
          │                                               └──> #3 teacher quality
          ├── #1 go/no-go tests on the dump (CPU) ──> #1 detector, pose-source choice, F2
          ├── uniform-resolution curve + region oracle ──> routing ──> cross-scale probe ──> #2
          └── KITTI go/no-go (unposed vs posed DA3) ──> LoRA platform ──> #3 training (1-2 sessions)

#1 signals ──> #2 focus/zoom selection      #1 signals + F2 σ ──> #3 label screening / weights
```

### 5.5 Suggested implementation order
| Step | What | GPU-h | Output |
|---|---|---|---|
| 0 | [official] baseline (03_eval: HiRoom, 7Scenes, DTU-64 first) with one pose solver; dump predictions; TSDF in CPU sessions; register runs in `experiments/` | 3–6 | Baseline vs paper Tab. 2/3 |
| 1 | All cheap gates: #1 tests on the dump (CPU), E_c noise probe, uniform-resolution curve, region oracle, KITTI unposed-vs-posed check | 4–6 | Go/no-go for each idea |
| 2 | #1 in full: detector (fit on one dataset, test on four), per-view pose source, F2; E_c residual if the probe allows | 5–8 (+3–5) | First result; signals for #2 and #3 |
| 3a | #2: routing (bit-exact check), cross-scale probe, large-view protocol vs Spark3R at equal FLOPs, zoom on ETH3D/DTU | 6–12 | |
| 3b | #3: LoRA platform (bit-exact when off, overfit 50 clips), teacher cache, 1–2 training sessions, controls, KITTI + Sec. 6 evaluation | 10–15 | |
| 4 | Stacking: audit-chosen focus views (#1 → #2), audit-screened teacher (#1 → #3), recalibration with the LoRA off/on | 2–4 | Final tables |

Total ≈ 30–50 GPU-h, i.e. about 1–2 weeks of Kaggle quota, provided scoring runs in CPU-only sessions.

Why this order:
- **#1 first.** It is the cheapest, its tests run on CPU, and the dump and signals it produces feed #2 and #3.
- **All gates in step 1.** No implementation work starts before its probe says go.
- **Steps 3a and 3b are independent.**
  - With two people: one takes #2, the other #3.
  - Alone: start the LoRA platform (#3) early, because it has the longest lead time and the training sessions must fit the weekly quota.
- **If a gate fails,** promote a runner-up (§4.5) instead of forcing the idea.
