# Dynamic scenes: extending what DA3 can do

**Status: prior-work check done on 2026-11-19** (§2, §3). Several ideas from the first draft of this
document have **already been published**, some of them on DA3 itself. The enhancement has been
re-scoped to what is still open (§4). Re-run the search right before the exam: this area moves
fast.

**Update 2026-11-19 (enhancement study, [da3_enhancement_report.md](da3_enhancement_report.md) §3.5).**
- Feeding external poses into DA3 through E_c is **already published**: ScaRF-SLAM, DAVIO and VIDAR do
  it, and the official `colmap` mode accepts poses. SpatialBench also reports that DA3 adheres closely
  to injected poses.
- C1 below is therefore **no longer a contribution**. It stays only as a cited component of the
  pipeline.
- Under the strict rule (anything already implemented is discarded), C3 and C4 apply published
  mechanisms (key masking; motion-aware fine-tuning), so they count as analysis and application, not
  as new methods.
- The DA3-specific method left in this document is C2.

**Goal.** DA3's results on static scenes are already strong (paper Tab. 2-4), and we don't try to
beat them. We try to **add a capability**. DA3 assumes the world does not move while the photos are
taken, and the paper lists dynamic scenes as future work (Sec. 8). In a dynamic scene, moving
objects corrupt the camera poses and the fused 3D model.

**TL;DR of the re-scoped contribution: "motion-robust DA3".** The published DA3-based method,
Dyna3, separates moving objects from the static background but **uses DA3's raw camera poses**,
and in its own tables they are much worse than those of VGGT-based methods (§2.1). Our enhancement
targets exactly that gap, with tools that only DA3 has:
1. **Pose-conditioned second pass**: feed the motion-robust poses back into DA3 through its camera
   encoder E_c. VGGT, DUSt3R and π³ cannot take poses as input. **The mechanism is already
   published** (ScaRF-SLAM, DAVIO and VIDAR feed external poses into DA3; see the update above). It is
   kept as a cited component and not claimed.
2. **Motion-masked ray-map pose recovery**: paper Eq. 1-2 computed on static pixels only.
   **No prior work found** (a small, DA3-specific step).
3. **Where (and whether) cross-view key masking helps DA3**: a study adapting VGGT4D / Ground4D
   masking to DA3's different layer layout. **The mechanism is prior work (cite it); the DA3
   adaptation and analysis are ours.**
4. Stretch: **motion-aware LoRA fine-tuning of DA3** (the PAGE-4D / MonST3R idea applied to DA3;
   cite it).

Motion segmentation and the static + per-frame-dynamic 4D output are **not** claimed as
contributions: they are prior work (Dyna3, VGGT4D). We use them as inputs and as a demo.

---

## 1. Why DA3 is a static-scene model

### 1.1 The formulation assumes one rigid world
For every view *i*, DA3 predicts a depth map `D_i` and a ray map `M_i = (t_i, d_i)` in **one shared
world frame** (Sec. 3.1). The 3D point seen by pixel `p` of view *i* is `P_i(p) = t_i + D_i(p) · d_i(p)`.
Nothing in the output depends on time. Multi-view reasoning only works if the same physical point
has the same coordinates in every view. Ground-truth normalization and the losses (L_P, L_M) also
assume one static point cloud per scene (Sec. 3.3).

### 1.2 What each part does when something moves
| Component | What goes wrong with motion |
|---|---|
| Per-view depth `D_i` | mostly still correct: it is the geometry at the time of frame *i* |
| Cross-view attention (L_g blocks) | a moving object's tokens match wrong places, or nothing; the attention mixes inconsistent evidence |
| Ray map / camera head | object motion is partly "explained" as camera motion, so **poses get biased** |
| Fusion (point cloud / TSDF) | moving objects leave ghost copies |
| Confidence | drops on moving regions (seen in our `robot_unitree` demo) |

### 1.3 What we observed ourselves
In the `robot_unitree` demo (static camera, moving robot; `04_demo_export` on Kaggle):
- per-frame depth of the robot looked correct;
- the ray maps were identical across frames: the camera was correctly recognised as static;
- **depth confidence was lowest on the robot** (`preview.png`, confidence column; in that first run
  the colour scale was inverted: red meant low);
- fusing all frames into one point cloud is wrong for the robot by construction.

---

## 2. Prior work check (searched 2026-11-19)

How much of each paper was read is noted per row. Abstract-level claims should be checked in the
full paper before citing.

### 2.1 Built on DA3
| Work | What it does | Overlap with our first draft | Read |
|---|---|---|---|
| **Dyna3** (Xiang et al., UC Davis; arXiv 2610.01286, 1 Oct 2026) | Training-free, **frozen DA3-BASE**. Motion prior: cosine best-match of DA3's DPT features across neighbouring frames. A VLM (Qwen2-VL) names the moving subject, SAM 3 segments it, then motion-aware instance selection. Output: **static background** (static pixels back-projected with DA3 poses) **+ per-frame dynamic point clouds**. Camera poses come from **DA3's ray map (RANSAC)**, with **no masking, attention change or pose refinement**. Does not use DA3's depth confidence. | **Covers our L1 (motion masks on DA3) and L3 (static + per-frame dynamic 4D output).** | full text, partly |

Dyna3's reported numbers (from its tables) show the gap we target:
- **DAVIS-16 masks:** J-Mean 65.0 vs VGGT4D 59.5. Dyna3 is better at segmentation.
- **TUM-dynamics pose ATE:** 0.121 vs VGGT4D 0.016. **Dyna3's DA3 poses are much worse.**
- **Sintel pose ATE:** 0.197 vs VGGT4D 0.076. Same story.
- **Speed and memory:** Dyna3 is faster and lighter (e.g. 15.6 s vs 45.5 s, 24.7 GB vs 130 GB on TUM).
- **Limitations it states:** coarse mask boundaries, small objects, and per-frame dynamic clouds
  without temporal coherence.
- A search snippet also suggests Dyna3 argues that DA3's attention is a weaker correspondence cue than
  that of correspondence-trained backbones (VGGT, DUSt3R). That makes "does attention masking help DA3?" a
  genuine open question (§4, C3).
- No code link found.

Also checked:
- **"Depth Anything V4"** (arXiv 2608.18388): **withdrawn** two days after posting (v2,
  20 Aug 2026, comment "Major errors in research"). Don't cite it.
- The DA3 paper itself handles motion only in evaluation (COLMAP ground truth computed with dynamic
  objects masked, Fig. 5).
- No DA3 LoRA / dynamic fine-tuning recipe was found.

### 2.2 Training-free dynamic extensions of other feed-forward models
| Work | Backbone | Mechanism | Overlap | Read |
|---|---|---|---|---|
| **Easi3R** (Chen et al., ICCV 2025; arXiv 2503.24391) | DUSt3R / MonST3R | motion from attention maps; **second pass re-weighting cross-attention** | our attention cue and two-pass idea | abstract |
| **VGGT4D** (Hu et al., arXiv 2511.19971, CVPR 2026) | VGGT | Gram similarities (QQᵀ, KKᵀ) over a temporal window → dynamic saliency; **mask refinement with a depth-reprojection + photometric residual**; **keys of dynamic tokens masked in shallow global layers (1-5)**; masking all layers hurts | **our cue B (reprojection consistency) and L2 (key masking)** | full text |
| **4DVGGT-D** (Zang et al., arXiv 2605.12027) | VGGT | VGGT4D-style mask; **two forward passes**, pass 2 zeroes keys of dynamic tokens in layers ≤ 5; confidence-weighted fusion of the two depths | two-pass masking, confidence fusion | full text |
| **Ground4D** (Zhao et al., arXiv 2606.28828; not the off-road paper of the same name, 2605.04435) | VGGT | SAM masks; **K/V of dynamic tokens removed in the first 5 global layers** ("fully masking … would cause a severe distribution shift"); then per-scene dynamic Gaussian optimization | key/value masking with external masks | full text |
| **Stream4D** (arXiv 2610.31320) | streaming (StreamVGGT) | training-free motion decoupling from KV-cache attention statistics | attention cue, streaming | abstract/snippet |
| **RayMap3R** (Wang et al., arXiv 2603.20588) | streaming feed-forward model | **ray-map predictions show a "static-scene bias"**; contrasting the ray-map branch with the image branch flags dynamic regions | any "ray-map motion cue" idea | abstract |

### 2.3 Trained / fine-tuned dynamic models
| Work | What it does | Overlap | Read |
|---|---|---|---|
| **MonST3R** (arXiv 2410.03825) | DUSt3R fine-tuned on dynamic data | fine-tuning for dynamics | abstract |
| **PAGE-4D** (ICLR 2026; arXiv 2510.17568) | VGGT + dynamics-aware aggregator: predicts a mask that **suppresses dynamic regions for pose tokens but keeps them for geometry**; fine-tunes only the middle layers on limited dynamic data | **our L5 (motion-aware fine-tuning)** | abstract + reviews |
| **MoRe** (CVPR 2026; arXiv 2603.05078) | streaming feed-forward depth + poses + motion masks; attention forcing | learned motion masks | abstract |
| **GeoMotion** (CVPR 2026; arXiv 2602.21810) | feed-forward motion segmentation from latent 4D geometry + optical flow + pose features | learned masks | abstract |
| **V-DPM** (CVPR 2026) | VGGT adapted to dynamic point maps (3D motion of every point) with little synthetic data | **our L4 (scene flow)** | project page/notes |
| **Any4D** (CMU, CVPR 2026; arXiv 2512.10935) | metric feed-forward 4D: depth, intrinsics, extrinsics, **dense scene flow**; **optional inputs incl. poses (IMU), depth, radar**; trained | L4; poses as *input* to a 4D model (trained, not a self-conditioning loop) | project page |
| **4RC** (arXiv 2602.10094), **OmniX** (arXiv 2607.10840), **DynamicVGGT** (CVPR 2026) | feed-forward 4D / trajectory fields / dynamic Gaussian heads | L4/L5 | abstract/snippet |
| **FlowCam** (arXiv 2306.00180), **GFlow** (arXiv 2405.18426) | lifting optical flow to 3D scene flow with pixel-aligned depth (trained / per-video optimization) | **our cue E / L4 flow lifting** (a classic technique) | abstract |

### 2.4 Language / interaction (from our earlier search, outside this document's scope)
DA3 is already being used as the geometry encoder or teacher in spatial multimodal LLMs and world models (e.g.
*Thinking with Spatial Code*, SpatialSV, OneCanvas, PAIWorld; several of these only from search
snippets). Open-vocabulary "lift CLIP features onto a point cloud" is well-established (e.g. LERF,
OpenScene, ConceptFusion). **Not a novel direction for us.** At most, a demo.

---

## 3. What is already done vs still open

| Our first-draft idea | Status | Consequence |
|---|---|---|
| L1 motion masks from DA3 signals (features, reprojection consistency, attention, semantics) | **Done**: Dyna3 (DA3 features + VLM + SAM 3), VGGT4D (reprojection refinement), Easi3R / VGGT4D (attention) | use as an **input**, cite; not a contribution |
| L1 cue A: DA3 depth confidence as motion cue | not found as the main cue (4DVGGT-D uses confidence for depth fusion) | weak alone; at most an auxiliary cue, small novelty |
| L2 key/value masking in cross-view attention, two passes | **Done for VGGT/DUSt3R** (VGGT4D, Ground4D, 4DVGGT-D, Easi3R); **not for DA3** | adaptation + **layer study on DA3** (C3), cite the mechanism |
| L2 static-only ray-map pose recovery (Eq. 1-2 on static pixels) | **not found** (Dyna3 uses unmasked ray-map RANSAC; RayMap3R uses ray maps as a *cue*, not masked pose recovery) | **C2, ours** |
| (new) pose-conditioned second pass through E_c | **Published for static scenes and SLAM** (found 2026-11-19): ScaRF-SLAM, DAVIO and VIDAR feed external poses into DA3 through E_c, and the official `colmap` mode accepts poses. Not tested on dynamic scenes, but the mechanism is the same | **C1: cited component, not a contribution** |
| L3 static map + per-frame dynamic 4D output | **Done** (Dyna3 on DA3; VGGT4D/Ground4D on VGGT) | demo only (Blender playback), cite Dyna3 |
| L4 3D scene flow (flow lifting / learned) | **Done** (V-DPM, Any4D, 4RC; FlowCam / GFlow for lifting) | optional demo, cite; addresses Dyna3's "no temporal coherence" limitation only as engineering |
| L5 motion-aware fine-tuning | **Done for VGGT/DUSt3R** (PAGE-4D, MonST3R); **not for DA3** | stretch C4, clearly a transfer, cite |

---

## 4. Re-scoped enhancement: motion-robust DA3

Pipeline, all training-free except C4, on the released DA3 weights (`da3-*` presets):

```
pass 1: DA3(images)                          -> depth, rays, conf, head poses
motion mask (prior work, cited)              -> M_i   (C0)
robust poses:  key-masked pass (C3) and/or masked ray-map Eq. 1-2 (C2)  -> poses*
pass 2: DA3(images, poses*, K*) via E_c (C1) -> depth/rays for ALL pixels, consistent with poses*
outputs: poses*, per-frame depth (dynamic objects included), static map + dynamic layer (demo)
```

### C0. Motion masks (input; not claimed)
Use one of these, cited, and report which:
- (a) **depth-reprojection consistency**: project `P_i(p)` into the other views and flag
  systematic depth disagreement, with an occlusion-aware sign test. This is the residual VGGT4D uses
  for refinement, and classic MVS geometric consistency.
- (b) the DA3 feature best-match cue of Dyna3 (without the VLM).
- (c) an off-the-shelf segmenter (SAM-family), as Ground4D does.

**Also report results with oracle masks** wherever ground truth allows it (synthetic data: masks
derivable from ground-truth depth + camera + optical flow, e.g. Sintel; PointOdyssey and Kubric ship
motion annotations; check each dataset). The C1-C3 gains are then separated from mask quality,
which is not our contribution. This also gives the cleanest scientific statement.

### C1. Pose-conditioned second pass (cited component; the mechanism is already published)
DA3 accepts known cameras: the encoder `E_c(f, q, t)` turns them into camera tokens that replace the
learnable token (Sec. 3.2). The paper shows that pose conditioning improves geometry on static
scenes (Tab. 7, rows f/g). Our idea:
1. estimate motion-robust poses `poses*` (C2/C3);
2. **re-run DA3 conditioned on `poses*`**. Every pixel (moving objects included) gets depth and rays
   predicted *in the frame of the robust cameras*, instead of a frame biased by the motion.

Questions it answers:
- Does conditioning on robust poses improve **dynamic-object depth** and the **4D reconstruction**?
- **Oracle upper bound:** condition on the *ground-truth* poses (TUM, Sintel). How much could a perfect
  pose estimator give through E_c?
- Failure mode: pose conditioning was trained on (mostly) static data, so with large movers it may
  behave unexpectedly. Measure it; don't assume.

Prior work (found 2026-11-19): feeding external poses into DA3 through E_c is already published for
static scenes and SLAM:
- ScaRF-SLAM (arXiv 2606.00307);
- DAVIO (2609.27702);
- VIDAR (2607.17171), where injection did *worse* than decoupled alignment;
- the official `colmap` mode.

SpatialBench (2605.27367) reports that DA3 adheres closely to injected poses. C1 is therefore not a
contribution: we only measure what it does in dynamic scenes. In posed mode the API returns the input
poses, so read D_C and the ray map directly when testing it.

### C2. Motion-masked ray-map pose recovery (DA3-specific)
Paper Eq. 1: camera centre = mean of the ray origins. Eq. 2: R and K from a DLT on the ray directions.
Both use **all** pixels. With the mask, use static pixels only (weight 0 on `M_i`). It is cheap and
complementary to C3. Compare it against:
- unmasked plain DLT (our `geometry/rays.py`);
- the official confidence-weighted RANSAC (the pose source of Dyna3);
- the camera head D_C (masked and unmasked token inputs, via C3).

This also gives the first systematic **camera head vs ray map under motion** comparison for DA3.
Dyna3 simply chose the ray-map RANSAC as "more reliable".

### C3. Where does cross-view key masking help DA3? (adaptation study)
Mechanism from VGGT4D / Ground4D / 4DVGGT-D: in the cross-view (global) attention blocks, other views
cannot attend to the tokens of moving regions. Implementation: a boolean key mask
`(B, 1, 1, S·N)` passed to `scaled_dot_product_attention`, at `O(S·N)` memory. Within-view blocks
stay untouched.

DA3 is not VGGT:
- DA3-S/B have only **4 cross-view blocks** (indices 5, 7, 9, 11), and DA3-L has 8.
- VGGT has 24 global layers, and the VGGT papers mask its first 5.
- Dyna3 suggests DA3's attention is a weaker correspondence cue.

Study: mask {none; block 5; blocks 5-7; all cross-view blocks}, with oracle masks and C0
masks, on poses and depth. The result is informative either way:
- **if masking helps:** we show the mechanism transfers to DA3, and in which blocks;
- **if it doesn't:** we explain why with DA3's architecture, and C1/C2 become the way to fix the poses.

### C4. Stretch: motion-aware LoRA fine-tuning of DA3 (transfer of PAGE-4D / MonST3R)
Fine-tune the released weights with LoRA, **with vs without** a motion-aware change under the same
budget (2-3 seeds). The change is loss-level: the camera/ray terms (L_M, L_C and cross-view L_P) are
masked on dynamic pixels, while the depth terms (L_D, L_grad) stay supervised everywhere. This is
the same principle PAGE-4D implements in its architecture (suppress dynamics for pose, keep them for
geometry), so cite it. Data: synthetic sets with motion annotations, e.g. PointOdyssey (in DA3's
Tab. 1), Dynamic Replica or Kubric. Check licenses and sizes. Never fine-tune on evaluation sequences.

### Demo (not claimed): 4D export to Blender
Static map (static pixels of all frames) + per-frame dynamic point clouds with keyframed visibility,
following Dyna3's decomposition (cite). Optional scene-flow arrows from lifted optical flow (cite
FlowCam/GFlow-style lifting; trained alternatives: V-DPM, Any4D).

---

## 5. Evaluation

Use the **same benchmarks and metrics as Dyna3 / VGGT4D** to stay comparable:

| What | Data | Metrics |
|---|---|---|
| Camera poses in dynamic scenes (C1-C3) | **TUM-dynamics**, **Sintel** (+ Bonn RGB-D Dynamic optional) | ATE, RTE, RRE (as in Dyna3/VGGT4D); paper's pose AUC@3/30 as an extra |
| Depth of dynamic objects (C1) | Sintel (GT depth), Bonn | AbsRel, δ<1.25, **split into dynamic and static pixels** |
| 4D reconstruction (C1, demo) | DyCheck (as in Dyna3/VGGT4D) | Accuracy / Completeness / Distance |
| Masks (only if we report C0) | DAVIS-16/17 | J-Mean, F-Mean |
| No regression | HiRoom (`03_eval`) | pose AUC and F1 unchanged with the extension on |

Rows to report:
- **DA3 raw poses** (camera head, and ray-map RANSAC = Dyna3's pose source);
- **+C2**, **+C3** (per masking choice), **+C1 (pose-conditioned)**, **C1 with GT poses (oracle)**;
- each with oracle masks (where derivable) and with C0 masks;
- Dyna3 / VGGT4D numbers **copied from their papers**, with a note on protocol differences (frame
  sampling, resolution, alignment). Dyna3 has no released code, so we can't rerun it.

---

## 6. How to state the contribution honestly
**Can claim** (as of the 2026-11-19 search):
- motion-masked ray-map pose recovery for DA3 (C2);
- the first analysis of cross-view key masking inside DA3, and of camera head vs ray map under
  motion (C3);
- a DA3 LoRA motion-aware fine-tuning recipe (C4), presented as the application of PAGE-4D/MonST3R's
  principle to DA3.

**Must not claim:**
- feeding external or motion-robust poses into DA3 through E_c as a new mechanism (**ScaRF-SLAM,
  DAVIO, VIDAR**); C1 is only a component;
- motion segmentation on DA3, or the static + per-frame dynamic 4D decomposition (**Dyna3**);
- attention/key masking for robust poses as a new mechanism (**Easi3R, VGGT4D, Ground4D, 4DVGGT-D**);
- reprojection-consistency motion masks (**VGGT4D**, classic MVS);
- scene flow from feed-forward models or flow lifting (**V-DPM, Any4D, FlowCam, GFlow**).

Using these methods as components **with citations** is normal and not plagiarism. Presenting them
as ours would be.

---

## 7. Implementation plan and status in our repo (Kaggle-only)

Status as of 2026-11-19. **None of the dynamic-scene pieces exist yet.** Some building blocks from
the baseline work can be reused.

| Piece | Where | Status in our repo | Notes |
|---|---|---|---|
| Pose-conditioned second pass (C1) | `src/da3/dynamic/pipeline.py` | **building block exists**: `DA3.forward(images, w2c=..., K=...)` already feeds known cameras through E_c (released DA3 weights; matched the official code in the "multiview_posed" parity mode, local CPU). The two-pass pipeline itself: **to do** | pass 1 → mask → poses* → pass 2 with poses* |
| Masked ray-map pose recovery (C2) | `src/da3/geometry/rays.py` | **partly**: plain Eq. 1-2 (DLT + RQ) implemented and unit-tested; **pixel weights / mask: to do** | weight 0 on dynamic pixels in Eq. 1 and in the DLT |
| Key-masked cross-view attention (C3) | `src/da3/backbone/layers.py`, `vit.py` | **to do**: attention currently has no mask argument | optional `attn_mask`, cross-view blocks only, block selection configurable; off = bit-exact |
| Motion masks (C0) | `src/da3/dynamic/masks.py` | **to do**. Reusable: unprojection / projection utilities, `scripts/check_dataset.py` (reprojection check) | reprojection consistency (+ optional confidence); hook for external / oracle masks |
| Evaluation | `scripts/eval_dynamic.py` + loaders (TUM-dynamics, Sintel, DyCheck, Bonn, DAVIS) | **to do**. Reusable: Sim(3) Umeyama + RANSAC alignment and pose AUC in `src/da3/eval/metrics.py` (match the released harness to round-off) | ATE/RTE/RRE (Sim(3)-aligned), depth split, DyCheck metrics |
| Demo export | `src/da3/dynamic/fourd.py`, `scripts/blender_import.py` | **to do**. Existing: static export (PLY / GLB / transforms.json) + Blender import of one point cloud, verified headless | static + per-frame dynamic, keyframed visibility |
| LoRA fine-tuning (C4) | LoRA module + masked-loss option + config | **to do**. Existing: losses (L_D, L_M, L_P, L_C, L_grad) and train loop with resume (built for from-scratch training) | released weights as start; with vs without |
| Notebook | `notebooks/05_dynamic.ipynb` | **to do** | data download, ablation table, Blender package |
| Tests | `tests/test_dynamic.py` | **to do** | empty mask = bit-exact; masked DLT on a synthetic moving object; pass 2 with GT poses reproduces conditioning |

## 8. Risks
- **C1 may not help**, if pose conditioning ignores or mishandles moving content. The oracle (GT poses)
  tells us quickly whether there is any headroom. Run it first.
- **C3 may hurt** (distribution shift; DA3 has few cross-view blocks). That is a valid, reportable
  finding.
- **Mask quality** dominates if we only use C0 masks, so always report GT-mask results too.
- **Scooping:** a DA3 pose-robustness paper could appear before the exam. Re-search the week before.

## 9. What NOT to do
- ❌ Don't present motion segmentation, the 4D decomposition, attention masking, or scene flow as our
  inventions (§6).
- ❌ Don't cite "Depth Anything V4" (withdrawn).
- ❌ Don't claim better scores on the paper's static benchmark. The claim is robustness in dynamic
  scenes, plus no regression on static ones.
- ❌ Don't tune thresholds or masking choices on test sequences. Use a validation sequence.
- ❌ Don't compare with published numbers under a different protocol without saying so.
- ❌ Don't let the extension change the static path. With the extension switched off, outputs must be bit-identical.

## 10. References (links checked on 2026-11-19)
- DA3: arXiv [2511.10647](https://arxiv.org/abs/2511.10647)
- **Dyna3**: arXiv [2610.01286](https://arxiv.org/abs/2610.01286)
- Easi3R: arXiv [2503.24391](https://arxiv.org/abs/2503.24391), ICCV 2025
- VGGT4D: arXiv [2511.19971](https://arxiv.org/abs/2511.19971)
- 4DVGGT-D: arXiv [2605.12027](https://arxiv.org/abs/2605.12027)
- Ground4D (monocular): arXiv [2606.28828](https://arxiv.org/abs/2606.28828)
- Stream4D: arXiv [2610.31320](https://arxiv.org/abs/2610.31320)
- RayMap3R: arXiv [2603.20588](https://arxiv.org/abs/2603.20588)
- MonST3R: arXiv [2410.03825](https://arxiv.org/abs/2410.03825)
- PAGE-4D: arXiv [2510.17568](https://arxiv.org/abs/2510.17568), [ICLR 2026](https://iclr.cc/virtual/2026/poster/10009841)
- MoRe: arXiv [2603.05078](https://arxiv.org/abs/2603.05078)
- GeoMotion: arXiv [2602.21810](https://arxiv.org/abs/2602.21810)
- Any4D: arXiv [2512.10935](https://arxiv.org/abs/2512.10935), [project page](https://any-4d.github.io/)
- 4RC: arXiv [2602.10094](https://arxiv.org/abs/2602.10094)
- OmniX: arXiv [2607.10840](https://arxiv.org/abs/2607.10840)
- V-DPM: [project page](https://www.robots.ox.ac.uk/~vgg/research/vdpm/)
- FlowCam: arXiv [2306.00180](https://arxiv.org/abs/2306.00180)
- GFlow: arXiv [2405.18426](https://arxiv.org/abs/2405.18426)
- Poses fed into DA3 (prior work for C1): ScaRF-SLAM arXiv [2606.00307](https://arxiv.org/abs/2606.00307), DAVIO arXiv [2609.27702](https://arxiv.org/abs/2609.27702), VIDAR arXiv [2607.17171](https://arxiv.org/abs/2607.17171), SpatialBench arXiv [2605.27367](https://arxiv.org/abs/2605.27367)
- Withdrawn, do not cite: "Depth Anything V4", arXiv [2608.18388](https://arxiv.org/abs/2608.18388)
