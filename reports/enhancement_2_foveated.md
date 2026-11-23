# Enhancement 2: Foveated DA3, mixed per-view resolutions in one pass

**Status (2026-11-23): first Kaggle run done, a limited end-to-end check. Not yet the confirmatory run.**
- Method and setup (§1–3) are unchanged.
- §4 holds the numbers of the **limited run**: 2 scenes per dataset (4 scenes), Sec. 6 pose only, and
  100 / 250 views in the large-view protocol. Setting **[official]**: the released DA3-Small / DA3-Base
  weights, run by us.
- Gate decisions on 4 scenes are a smoke test, not the pre-registered result (§4.8). The full run
  (`MAX_SCENES = 0`) and the region-zoom session (gate 2) are still to do.
- After this run: the inspection is logged in the plan's deviations log, and exploratory
  diagnostics plus a two-pass control were added for the full run (§7). The plan's constants and
  gates are unchanged.
- **Preliminary conclusion.** The routing is exact. With the released weights, however, mixing
  resolutions in one pass does not give one consistent reconstruction:
  - pose AUC between focus and context views collapses;
  - the two resolution groups come out at different depth scales;
  - both groups are degraded internally.

  Every alternative with the same token count is better. The effects are much larger than the
  uncertainty of 4 scenes (§5).

Origin of the idea: rank 2 of the multi-agent study,
[docs/da3_enhancement_report.md](../docs/da3_enhancement_report.md) §4.3 (idea B2, with H4 as its
region variant). It is the only idea the novelty check judged fully novel (§3.4 there): native mixed
per-view resolutions in one training-free pass.

## 1. Idea in one paragraph

DA3 runs every view at one resolution, so the token count grows with views × resolution and the
cross-view attention cost grows quadratically with it. Foveated DA3 gives a few **focus views** the
full resolution (504 px) and many **context views** a quarter of the tokens (252 px), in **one**
forward pass. This works without retraining because the released backbone gives every patch token
the same constant RoPE position in its cross-view blocks: cross-view attention does not depend on
where a token sits in its image or on how many tokens a view has, while patch embedding, position
embedding, within-view blocks, Dual-DPT and the camera head all work per view. The region-level use
(H4) turns a region of a full-resolution photo into a **virtual zoom camera**, an exact rotation of
the source camera with a narrow field of view, processed together with the scene views and given its
camera through E_c.

## 2. What was implemented

| Part | Code | What it does |
|---|---|---|
| Routing | `src/da3/enhancements/foveated/routing.py` | `foveated_forward`: one pass over views of any mix of resolutions. Patch embedding, interpolated position embedding, within-view blocks (2D RoPE per grid), Dual-DPT and camera decoding run per resolution group; cross-view blocks run on the concatenation of all tokens. Mirrors `DA3.forward` operation by operation (released reference-view selection, camera tokens, E_c with per-view FOV, concatenated head features, chunked DPT), so with one resolution the outputs are **bit-identical** to the original (unit tests on CPU, `parity` on the GPU) |
| Attention-mass balancing | same file, `balance=True` | Optional, training-free. Adds log(N_focus / N_view) to the logits of a context view's patch keys in the cross-view blocks. This equals replicating each of its tokens N_focus / N_view times (nearest-neighbour upsampling of its token grid as seen by the softmax), so every view weighs as much as in training, where all views had the same token count. Implemented with one extra q/k channel, no N_q × N_k mask; a unit test checks the equivalence with explicit replication |
| Plans and budgets | `plan.py` | Focus-view selection (evenly spaced by default, view 0 = reference always a focus view), per-view resolutions, patch tokens and backbone GFLOPs, the uniform resolution with the same token budget, the equal-budget subset of full-resolution views |
| Benchmark adapter | `adapter.py` + `scripts/eval_bench.py --foveated / --process-res / --meter` | The released Sec. 6 harness with mixed resolutions: context depth is upsampled to the focus size and its intrinsics rescaled per axis for the export. Every run writes per-scene efficiency meta (tokens, GFLOPs, time, peak memory). With all views in focus it is bit-identical to the original adapter (unit test) |
| Region zoom | `zoom.py` | Virtual zoom camera for a box of the full-resolution photo (minimal roll, focal so the box fits), exact warp with an area pre-filter, naive crop baseline, depth of each method mapped back to source-camera depth at the same points, oracle region choice (geometric detail in the GT) |
| Metrics | `metrics.py`, `src/da3/eval/gt_depth.py` | AbsRel / δ (median-aligned per view or region; per scene for consistency), pose AUC on view subsets (focus–focus, context–context, focus–context), region point metrics (F1 at 0.5 / 1 / 2 % of the depth, Chamfer), GT depth decoders of HiRoom, 7Scenes, ETH3D |
| Statistics | `stats.py`, `analysis.py`, `configs/foveated_analysis.yaml` | Pre-registered analysis: scene-bootstrap CIs, three-way decisions, critical margins, A/A noise floor (capped), Bonferroni CIs for secondary configurations, stratified dev/test split for the rescue analysis; unit tests on synthetic data with known answers (`tests/test_foveated_stats.py`) |
| Experiments | `scripts/foveated_eval.py` | `parity`, `probe`, `scaling`, `sec6views`, `zoom`, `analyse` (gates, tables, decision figure, standard result records, `summary.md`) |

## 3. Experimental setup

- **Weights:** DA3-Small and DA3-Base, released (Apache-2.0); network unchanged; fp16 autocast on
  T4 / P100 as in the baseline; camera-head poses (API default) everywhere, so the pose source is
  fixed (one change at a time).
- **Reference view:** "first" (the paper protocol); view 0 is always a focus view.

| Protocol | Data | Configurations | Metrics |
|---|---|---|---|
| **Probe** (gates 1, 3) | HiRoom (30 scenes), 7Scenes (7); 3 interleaved subsets of 16 evenly spaced views per scene, every other view a focus view | uniform 504 / 378 / 336 / 280 / 252 px; A/A (504, reference view moved); focus views alone; focus 504 + context 252 (primary), 280, 252 balanced; primary with 3 random focus choices | AbsRel / δ1 of focus and context views; AUC@3 / AUC@30 of focus, context and all pairs; tokens, time, memory |
| **Large views** | 7Scenes, all 7 sequences (1000 frames; stairs 500); 100, 250, 500, 1000 views per pass; 10% focus views | all 504; foveated 504/252 (± balanced); uniform at equal tokens; fewer 504-px views at equal tokens (focus views included); all 252 | time (median of 3 on 2 scenes), peak memory, out-of-memory, tokens, GFLOPs; AUC of focus pairs; AbsRel of focus views |
| **Sec. 6** | HiRoom, 7Scenes; released harness (≤ 100 frames, seed 42, evo + RANSAC, TSDF) | uniform 504 (original DA3) / 336 / 252; foveated 25% and 10% at 504/252 (± balanced) | AUC@3 / AUC@30, F1 without poses; per-view AbsRel and pose AUC of the focus / context views; paired CIs vs the original and vs the uniform curve at equal tokens |
| **Region zoom** (gate 2) | ETH3D, 11 scenes, 2 source views each, one region (1/4 of the image side) per view and selector: oracle (GT detail), random, texture | global 504 / 896 px; naive crop (scene pass, alone); zoom view (scene pass unposed, posed with DA3's own cameras, posed with GT cameras, alone) | region AbsRel / δ1.05 / F1 at fine thresholds / Chamfer; AbsRel with the source view's scale (consistency); R_v error from D_C and from the ray map; zoom focal error |

On 640 × 480 frames, 504 px = 972 patch tokens per view and 252 px = 252, so 25% focus views at 504
plus 75% at 252 cost exactly as many tokens as all views at 336 px (432 per view); 10% focus views
match about 294 px.

**Statistics: pre-registered** in [docs/FOVEATED_ANALYSIS_PLAN.md](../docs/FOVEATED_ANALYSIS_PLAN.md)
(constants in `configs/foveated_analysis.yaml`, SHA-256 stored with every result), fixed before the
first run.
- The scene is the unit of analysis; every comparison is paired, with a 95% scene-bootstrap CI.
- Every gate is **PASS / FAIL / INCONCLUSIVE**, read off its CI, together with the margins at which
  the decision would change.
- Margin: the report's 3%, raised to the noise floor of a treatment-blind A/A run (the original DA3
  with only the reference view moved) and capped at 6%.
- One primary configuration per gate. Mitigations are secondary, with simultaneous CIs. The only
  selection step (rescue) chooses on a dev half of the scenes and decides on the held-out half.

**Go/no-go gates** (report §4.3, made precise in the plan):
- **G0 routing:** equal resolutions bit-identical to `DA3.forward`; mixed-resolution outputs
  permute with the views.
- **G1 uniform-resolution curve:** G1a and G1b must both pass.
  - G1a, headroom: all-252 has a higher AbsRel than all-504, by more than the margin. If it does
    not, plain downscaling would win.
  - G1b, context informative: the focus views are more accurate with the context views than alone,
    by more than the margin. If they are not, context views carry nothing.
- **G2 region zoom (oracle regions):** region AbsRel ratio zoom / global 896 px ≤ 0.90 and median
  R_v error ≤ 1°.
- **G3 cross-scale probe (8 at 504 + 8 at 252):** C1 and C2 must both pass.
  - C1: the focus-view AbsRel ratio to all-504 is ≤ 1 + margin.
  - C2: the context-pair AUC@30 ratio to all-252 is ≥ 1 − margin.

## 4. Results [official], limited run

### 4.0 The run
- **Session.** Notebook `06_foveated` on a Kaggle T4, with:
  - `MAX_SCENES = 2`: 2 HiRoom + 2 7Scenes scenes, i.e. **4 scenes** in the probe and in Sec. 6;
  - `SCALING_VIEWS = [100, 250]`;
  - `SEC6_MODES = ["pose"]`: no TSDF fusion, so no F1 (the F1 panels of the figures are empty).
- **Protocols run:**
  - parity, probe and Sec. 6 on both models;
  - large-view protocol on DA3-Small, with all 7 7Scenes sequences as planned.
- **Not run yet:** region zoom (gate 2), large-view protocol at 500 / 1000 views, and the full scene
  sets.
- **Plan.** The plan constants were not changed after this run. Every quantity in §4.7 is derived
  after seeing the results and is **exploratory** (it is not in the pre-registered plan).
- **Result records.** The records of this run are tagged `_limited2`. They should not be committed
  to `experiments/results/` as final numbers.

### 4.1 Routing check (G0): PASS on both models
| Check | DA3-Small | DA3-Base |
|---|---|---|
| equal resolution, fp32 / fp16 / reference selection (saddle_balanced) / posed through E_c / balance no-op: bit-identical | yes (max diff 0) | yes (max diff 0) |
| equal resolution split into two groups (round-off only) | 3.8e-6 rel. | 6.3e-6 rel. |
| mixed resolution, view-order equivariance | 1.0e-6 rel. | 1.6e-6 rel. |

Time per patch token is the same for the foveated pass, uniform 294 px and fewer 504-px views (about
5.7e-5 s at 100 views): the routing adds no overhead. The cross-view RoPE position is a true constant
(special tokens 0, patches 1) independent of the grid size (`src/da3/backbone/vit.py:178-186`).

### 4.2 Probe (16 views per subset, 8 focus, 3 subsets per scene, 4 scenes; means over scenes)
AbsRel is median-aligned per view. "Scene-scale" AbsRel uses one scale per scene (cross-view
consistency). The cross-pair AUC is derived exactly from the all / focus / context columns (§4.7).

| Model | Configuration | AbsRel focus | AbsRel context | AbsRel scene-scale | AUC@30 focus pairs | AUC@30 context pairs | AUC@30 cross pairs | AUC@3 all | tokens | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| DA3-Small | all 504 | 0.0551 | 0.0509 | 0.0571 | 80.0 | 78.5 | 79.5 | 15.7 | 16 850 | 0.79 |
| DA3-Small | all 378 | 0.0574 | 0.0534 | 0.0581 | 79.2 | 75.9 | – | 14.3 | 9 423 | 0.40 |
| DA3-Small | all 336 | 0.0572 | 0.0548 | 0.0588 | 77.6 | 75.1 | – | 11.8 | 7 488 | 0.32 |
| DA3-Small | all 252 | 0.0674 | 0.0668 | 0.0725 | 65.8 | 65.9 | 67.3 | 5.2 | 4 284 | 0.18 |
| DA3-Small | focus views alone | 0.0558 | – | 0.0578 | 78.9 | – | – | 16.0 | 8 424 | 0.36 |
| DA3-Small | 504 / 252 | 0.0574 | 0.0733 | **0.237** | 79.6 | 59.6 | **21.0** | 3.2 | 10 570 | 0.47 |
| DA3-Small | 504 / 280 | 0.0568 | 0.0668 | 0.206 | 78.1 | 62.5 | 25.1 | 3.7 | 11 020 | 0.49 |
| DA3-Small | 504 / 252 balanced | 0.0599 | 0.0725 | 0.237 | 77.0 | 61.0 | 21.4 | 2.2 | 10 570 | 0.51 |
| DA3-Small | A/A (all 504, reference moved) | 0.0537 | 0.0499 | 0.0515 | 81.1 | 77.6 | – | 15.7 | 16 850 | 0.78 |
| DA3-Base | all 504 | 0.0462 | 0.0446 | 0.0480 | 88.5 | 81.6 | 86.4 | 25.6 | 16 850 | 2.02 |
| DA3-Base | all 378 | 0.0483 | 0.0463 | 0.0478 | 88.2 | 84.3 | – | 26.4 | 9 423 | 1.09 |
| DA3-Base | all 336 | 0.0464 | 0.0443 | 0.0471 | 89.2 | 86.3 | – | 24.6 | 7 488 | 0.86 |
| DA3-Base | all 252 | 0.0529 | 0.0551 | 0.0563 | 83.6 | 77.9 | 82.3 | 13.8 | 4 284 | 0.46 |
| DA3-Base | focus views alone | 0.0463 | – | 0.0489 | 86.4 | – | – | 25.3 | 8 424 | 0.94 |
| DA3-Base | 504 / 252 | 0.0502 | 0.0647 | **0.250** | 79.9 | 57.8 | **22.5** | 4.3 | 10 570 | 1.21 |
| DA3-Base | 504 / 280 | 0.0494 | 0.0591 | 0.216 | 79.8 | 62.2 | 26.6 | 4.1 | 11 020 | 1.26 |
| DA3-Base | 504 / 252 balanced | 0.0536 | 0.0614 | 0.239 | 71.4 | 56.2 | 23.0 | 2.2 | 10 570 | 1.29 |
| DA3-Base | A/A (all 504, reference moved) | 0.0450 | 0.0440 | 0.0443 | 86.3 | 83.9 | – | 23.6 | 16 850 | 2.01 |

**Random focus choices** (primary configuration with 3 random focus sets) give results similar to
the evenly spaced choice:

| Model | Focus AbsRel | AUC@30 focus | AUC@30 context |
|---|---|---|---|
| DA3-Small | 0.0553–0.0557 | 75.6–77.1 | 59.5–62.7 |
| DA3-Base | 0.0486–0.0499 | 69.4–78.1 | 62.3–65.7 |

### 4.3 Large-view protocol (DA3-Small, T4, 7Scenes, all 7 sequences, 10% focus views)
| Views | Configuration | tokens | time (s) | peak memory (GB) | speed-up | AUC@30 focus pairs | AbsRel focus |
|---|---|---|---|---|---|---|---|
| 100 | all 504 | 9.72e4 | 9.54 | 3.73 | 1× | 81.7 | 0.0705 |
| 100 | foveated 504/252 | 3.24e4 | 1.87 | 1.63 | 5.1× | 63.1 | 0.0812 |
| 100 | foveated 504/252 balanced | 3.24e4 | 2.22 | 1.63 | 4.3× | 58.3 | 0.0873 |
| 100 | uniform 294, equal tokens | 3.36e4 | 1.95 | 1.42 | 4.9× | 76.5 | 0.0746 |
| 100 | 33 views at 504, equal tokens | 3.21e4 | 1.80 | 1.93 | 5.3× | **81.9** | **0.0703** |
| 100 | all 252 | 2.52e4 | 1.35 | 1.12 | 7.1× | 72.9 | 0.0815 |
| 250 | all 504 | 2.43e5 | 52.6 | 8.29 | 1× | 78.4 | 0.0717 |
| 250 | foveated 504/252 | 8.10e4 | 6.93 | 2.54 | 7.6× | 62.3 | 0.0818 |
| 250 | foveated 504/252 balanced | 8.10e4 | 9.08 | 2.54 | 5.8× | 55.6 | 0.0887 |
| 250 | uniform 294, equal tokens | 8.40e4 | 7.40 | 3.00 | 7.1× | 74.8 | 0.0750 |
| 250 | 83 views at 504, equal tokens | 8.07e4 | 6.88 | 3.27 | 7.6× | **79.1** | **0.0709** |
| 250 | all 252 | 6.30e4 | 4.70 | 2.30 | 11.2× | 70.9 | 0.0812 |

Foveated vs each alternative, paired over 7 scenes (95% CIs):

| Views | Foveated vs | Δ AUC@30 focus pairs (pts) | AbsRel focus ratio |
|---|---|---|---|
| 100 | all 504 | −18.6 [−22.4, −14.6] | 1.151 [1.130, 1.168] |
| 100 | uniform 294 | −13.4 [−17.2, −9.9] | 1.088 [1.044, 1.141] |
| 100 | 33 views at 504 | −18.8 [−22.3, −15.1] | 1.155 [1.134, 1.175] |
| 100 | all 252 | −9.8 [−14.1, −5.4] | 0.996 [0.923, 1.082] |
| 100 | balanced | +4.7 [+3.7, +5.7] | 0.930 [0.898, 0.956] |
| 250 | all 504 | −16.1 [−18.8, −13.3] | 1.141 [1.107, 1.174] |
| 250 | uniform 294 | −12.5 [−17.2, −7.9] | 1.090 [1.057, 1.135] |
| 250 | 83 views at 504 | −16.8 [−19.6, −13.8] | 1.153 [1.132, 1.177] |
| 250 | all 252 | −8.6 [−12.8, −4.2] | 1.007 [0.941, 1.091] |
| 250 | balanced | +6.7 [+5.6, +7.7] | 0.922 [0.886, 0.950] |

### 4.4 Sec. 6 protocol (released harness, ≤ 100 frames, seed 42, 4 scenes, pose only)
Camera-head poses. The differences in brackets are relative to the original DA3 (504 px). F1 was not
run.

| Model | Configuration | HiRoom AUC@3 / AUC@30 | 7Scenes AUC@3 / AUC@30 |
|---|---|---|---|
| DA3-Small | all 504 (original) | 3.9 / 70.3 | 29.3 / 88.3 |
| DA3-Small | all 336 | 3.5 / 70.3 (+0.0) | 19.8 / 85.8 (−2.5) |
| DA3-Small | all 252 | 1.0 / 59.0 (−11.3) | 11.9 / 81.5 (−6.8) |
| DA3-Small | foveated 10% | 0.9 / 55.0 (−15.3) | 6.2 / 66.3 (−22.0) |
| DA3-Small | foveated 25% | 0.0 / 39.0 (−31.3) | 4.5 / 56.0 (−32.3) |
| DA3-Small | foveated 25% balanced | 0.1 / 39.5 (−30.9) | 3.5 / 54.9 (−33.4) |
| DA3-Base | all 504 (original) | 15.3 / 81.6 | 32.5 / 90.0 |
| DA3-Base | all 336 | 17.3 / 85.5 (+3.9) | 24.8 / 88.5 (−1.5) |
| DA3-Base | all 252 | 5.7 / 78.1 (−3.5) | 19.2 / 86.7 (−3.3) |
| DA3-Base | foveated 10% | 0.8 / 49.4 (−32.2) | 9.7 / 72.8 (−17.2) |
| DA3-Base | foveated 25% | 0.0 / 39.9 (−41.7) | 5.0 / 62.1 (−27.9) |
| DA3-Base | foveated 25% balanced | 0.5 / 35.9 (−45.7) | 6.2 / 64.1 (−25.9) |

Per-view analysis of the same runs, with paired 95% scene-bootstrap CIs (4 scenes). "Equal tokens"
means the uniform curve interpolated per scene in log tokens. Each run is compared on the focus set
of its own fraction.

| Model | Configuration | Focus AbsRel vs original | Focus AbsRel vs equal tokens | Focus-pair AUC@30 vs original (pts) | Focus-pair AUC@30 vs equal tokens (pts) |
|---|---|---|---|---|---|
| DA3-Small | foveated 10% | 1.089 [0.949, 1.149] | 0.933 [0.729, 1.036] | −18.6 [−22.9, −12.2] | −12.8 [−20.0, −1.2] |
| DA3-Small | foveated 25% | 1.100 [1.047, 1.187] | 1.036 [0.995, 1.055] | −31.8 [−35.7, −28.0] | −30.5 [−33.3, −27.1] |
| DA3-Small | foveated 25% balanced | 1.175 [1.087, 1.276] | 1.107 [1.044, 1.148] | −32.1 [−35.4, −29.0] | −30.8 [−33.6, −27.9] |
| DA3-Base | foveated 10% | 1.210 [1.175, 1.287] | 1.140 [1.027, 1.216] | −24.7 [−32.2, −16.4] | −23.2 [−31.6, −14.7] |
| DA3-Base | foveated 25% | 1.169 [1.114, 1.427] | 1.173 [1.107, 1.397] | −34.8 [−41.7, −25.4] | −35.9 [−45.5, −25.7] |
| DA3-Base | foveated 25% balanced | 1.269 [1.169, 1.664] | 1.273 [1.160, 1.630] | −35.8 [−45.7, −24.0] | −36.9 [−49.5, −24.3] |

Focus-view AbsRel on the 10% focus set:

| Model | all 504 | all 336 | all 252 | foveated 10% |
|---|---|---|---|---|
| DA3-Small | 0.0667 | 0.0693 | 0.0791 | 0.0757 |
| DA3-Base | 0.0619 | 0.0608 | 0.0653 | **0.0739** (worse than all 252) |

### 4.5 Region zoom (ETH3D)
_Not run yet (own session)._

### 4.6 Gates (4 scenes: smoke test, not the pre-registered decision)
Decision, estimate [95% CI], and the margins at which the decision changes.

| Gate | DA3-Small | DA3-Base |
|---|---|---|
| A/A ratio (depth / pose) → noise floor | 0.978 [0.929, 1.000] / 0.999 [0.900, 1.102] → 7.1% / 10.2% | 0.979 [0.914, 1.002] / 1.002 [0.945, 1.061] → 8.6% / 6.1% |
| Margin used (depth / pose) | 6% / 6% (noise floor above the cap: the protocol cannot resolve 3% with 4 scenes) | 6% / 6% (same) |
| G0 routing parity | PASS (7/7) | PASS (7/7) |
| G1a headroom (u252 / u504 AbsRel) | PASS, 1.266 [1.191, 1.436] | PASS, 1.189 [1.082, 1.430] |
| G1b context informative (with / without context) | FAIL, 1.028 [0.983, 1.040] | FAIL, 1.085 [1.058, 1.161] (context makes the focus views worse) |
| **G1** | **FAIL** | **FAIL** |
| G3 C1 focus AbsRel (504/252 / u504) | PASS, 1.041 [1.010, 1.057] | INCONCLUSIVE, 1.086 [1.052, 1.147] |
| G3 C2 context AUC@30 (504/252 / u252) | INCONCLUSIVE, 0.904 [0.809, 0.959] | FAIL, 0.742 [0.559, 0.880] |
| **G3 primary** | **INCONCLUSIVE** | **FAIL** |
| G3 secondary 504/280 (97.5% CIs) | FAIL (C1 1.029 PASS; C2 0.862 [0.729, 0.919] FAIL) | FAIL (C1 1.068 INC; C2 0.758 FAIL) |
| G3 secondary balanced (97.5% CIs) | INCONCLUSIVE (C1 1.087, C2 0.925) | FAIL (C1 1.160 FAIL; C2 0.721 FAIL) |
| G3 robustness: 3 random focus choices | INCONCLUSIVE ×3 | FAIL ×3 |
| G3 rescue (dev 2 scenes / test 2 scenes) | FAIL: no mitigation keeps context AUC@30 within the margin on dev | FAIL (same) |
| G2 region zoom | not run | not run |

### 4.7 Derived quantities (exploratory, computed after seeing the results)
These are not in the pre-registered plan and are reported as descriptive.

**Cross-pair AUC.** Pose AUC is the mean over the error bins of the cumulative histogram, so it is a
mean of per-pair terms and is linear in the pairs. With 8 focus + 8 context views there are
- 28 focus–focus pairs,
- 28 context–context pairs,
- 64 focus–context (cross) pairs,

so AUC_cross = (120 · AUC_all − 28 · AUC_focus − 28 · AUC_context) / 64. This is exact for every
subset, and so for the scene means. The values are in the table of §4.2:
- uniform passes: cross pairs reach 67–86 AUC@30;
- mixed passes: **21–27**.

**Share of the resolution gain kept by the focus views.** The share is
(AbsRel_all252 − AbsRel_foveated) / (AbsRel_all252 − AbsRel_all504), on the focus views:

| Focus share | Protocol | DA3-Small | DA3-Base |
|---|---|---|---|
| 50% | probe | 82% | 40% |
| 25% | Sec. 6 | 49% | −37% (worse than all 252) |
| 10% | Sec. 6 | 27% | worse than all 252 |
| 10% | large views, 100 | 3% | – |
| 10% | large views, 250 | −6% | – |

**Full-resolution context ceiling.** This asks whether context helps even at 504 px. Configurations
that already ran answer it:

| Comparison | DA3-Small | DA3-Base |
|---|---|---|
| Probe: focus-view AbsRel, 8 extra 504-px views vs focus views alone | 0.0551 vs 0.0558 (−1.3%) | 0.0462 vs 0.0463 (−0.2%) |
| Probe: focus-pair AUC@30, same comparison | 80.0 vs 78.9 | 88.5 vs 86.4 |
| Large views, 100: 33 views at 504 vs all 100 at 504 (focus AUC@30 / AbsRel) | 81.9 vs 81.7 / 0.0703 vs 0.0705 | – |
| Large views, 250: 83 views at 504 vs all 250 at 504 (focus AUC@30 / AbsRel) | 79.1 vs 78.4 / 0.0709 vs 0.0717 | – |

**Scale gap between the groups.** The per-view scale was not saved in this run (`depth_rows` kept only
the errors). Inverting the scene-scale AbsRel of about 0.24 roughly puts the scale ratio between the
two groups at 1.4–1.7×. This is a back-of-envelope estimate; the full run measures it (§7).

### 4.8 How far the numbers can be trusted
- **The gate decisions are a smoke test, not a result.** With 4 scenes:
  - the percentile scene bootstrap has only 35 distinct resamples, and its CIs are too narrow;
  - the Wilcoxon signed-rank p-value cannot go below 0.125;
  - the "noise floor above the cap" status is mostly a consequence of n = 4.
- **The main effects are far outside that uncertainty:**
  - cross-pair AUC@30 falls by about 60 points;
  - Sec. 6 AUC@30 falls by 15–46 points;
  - scene-scale AbsRel grows 4–5×;
  - the large-view protocol already uses its planned 7 scenes, with CIs that exclude 0 against
    every alternative at equal tokens.
- **The full run is unlikely to change the conclusion.** It may change individual labels: Small's G3
  C1 PASS could become INCONCLUSIVE, and its C2 is likely to become FAIL.

## 5. Discussion

**The implementation is not the cause.**
- The routing is bit-identical at equal resolution and only round-off when same-size views are split
  into groups.
- Uniform 252 px on its own behaves normally.
- No metric code is specific to mixed passes.

What fails is the released network's behaviour on an input it never saw in training.

**Main finding: the two resolution groups do not form one reconstruction.**
- Per-view depth shape survives: median-aligned AbsRel stays at 0.05–0.07.
- With one scale per scene, AbsRel grows from about 0.05 to 0.21–0.25 in both models: focus and
  context views come out at different depth scales.
- Pose AUC@30 between a focus and a context view drops from 67–86 (uniform) to 21–27, and AUC@3 over
  all pairs from 5–26 to 2–4.
- The pre-registered gates do not look at either of these quantities: G3 C2 uses only
  context–context pairs. The largest effect of the study is therefore descriptive (§6).

**Likely mechanism (hypothesis, to be tested):**
- With constant RoPE positions, cross-view attention treats each view as an unordered set of
  tokens, so views are linked by feature similarity alone.
- A 252-px patch covers four times the image area of a 504-px patch, and DINOv2 features are not
  scale-invariant. The same surface therefore gives different tokens in the two groups.
- DA3 was trained with one resolution per batch, so it never learned to match tokens across scales.
- The groups behave like two loosely coupled reconstructions.
- Attention-mass balancing does not fix the cross pairs. This fits a feature-scale problem, not an
  attention-mass problem.

**Mixing also degrades each group internally, and the damage grows as the focus share drops.**
- Context-pair AUC@30 vs all 252:
  - Base: 57.8 vs 77.9;
  - Small: 59.6 vs 65.9.
- Base's focus pairs: 79.9 mixed vs 86.4 for the focus views alone.
- The focus views keep 82% of the 252 → 504 px depth gain at 50% focus views (Small), 49% at 25%,
  27% at 10%, and about none in the 100–250-view protocol (§4.7).
- At 10% focus views, the share the method was designed for, the foveated focus views are no better
  than all views at 252 px: AbsRel ratio 0.996 [0.92, 1.08].
- DA3-Base is more sensitive than DA3-Small throughout. At 25% and 10% focus views its focus views
  are worse than with every view at 252 px.

**Context views barely help the focus views, even at 504 px.**
- Adding 8 views at 504 px changes focus-view AbsRel by 0.2–1.3%.
- In the large-view protocol, dropping two thirds of the 504-px views leaves the focus views
  unchanged (§4.7).
- DA3's geometry for the focus views saturates after a few views. G1b could not reach its −6% bar in
  this protocol even with perfect context. In hindsight the gate should have been reported next to
  this ceiling.
- The only remaining case for foveation is putting all views in one frame cheaply, and that is
  exactly what fails.

**Alternatives at the same token count win.**
- At 100 views, foveated vs uniform 294 px vs 33 views at 504 px:

  | | foveated | uniform 294 | 33 views at 504 |
  |---|---|---|---|
  | focus AUC@30 | 63.1 | 76.5 | 81.9 |
  | focus AbsRel | 0.081 | 0.075 | 0.070 |
  | time | 1.87 s | 1.95 s | 1.80 s |

- The 5–7.6× speed-up over all-504 is real, but the alternatives get the same speed-up.
- In the probe, 504/252 costs more tokens than uniform 378 px and is worse on every metric.
- The report's risk "uniform 336 px may tie" came true. For DA3-Base, 336 px matches 504 px with
  2.25× fewer tokens:
  - probe focus AbsRel 0.0464 vs 0.0462;
  - Sec. 6 HiRoom AUC@30 85.5 vs 81.6.

**The mitigations do not close the gap.**
- 280-px context views improve the cross pairs slightly: 25–27 vs 21–23.
- Attention-mass balancing never helps the focus views and costs 18–31% more time.
- The rescue analysis selects nothing.

**Side observation: the A/A comparison is not a pure null.** Moving the reference to a middle focus
view lowers AbsRel by about 2% in both models (ratios 0.978 / 0.979) and scene-scale AbsRel by 8–10%.
The choice of reference view matters, so the "treatment-blind" A/A comparison includes a small
systematic effect as well as noise.

## 6. Limitations and honest claims

- **Cross-scale token matching was never trained.** DA3 saw one resolution per batch with the
  short side ≥ 280 px; a 252-px context view is outside that range. Gate 3 measures the effect;
  280-px context views and attention-mass balancing are the training-free mitigations. *Limited run:
  this risk materialised (§5), and neither mitigation removes it.*
- **Uniform downscaling may tie** at the same token budget. That is a reportable finding: the
  claim is detail where it is needed (the focus views), not raw speed. *Limited run: uniform
  downscaling and fewer full-resolution views do better than foveation.*
- **The efficiency bar is Spark3R** (training-free token reduction on DA3, ~17× at 1000 frames,
  accuracy unchanged). It is not re-run here; a comparison at equal FLOPs and stacking the two
  are future work.
- **Sec. 6 is not token-bound at ≤ 100 views** on a T4, so the paper's tables do not show the
  efficiency gain; the large-view protocol and the region protocol are new and never mixed into the
  Sec. 6 tables.
- **Region zoom:** zoom views have a narrow field of view and no baseline of their own, so their
  detail is mostly monocular; regions are chosen with GT (oracle); ETH3D photos are distorted
  (thin-prism fisheye) and treated as pinhole, like the released harness does. Two points to read
  when gate 2 runs:
  - **R_v.** The gate reads the R_v error from the pass where the zoom camera is given through E_c.
    If D_C copies its E_c input, C2 passes trivially. The informative number is the R_v error of
    `zoom_unposed`, which is computed too.
  - **Multi-view contribution.** Compare `zoom_posed_da3cams` with `zoom_mono`. If they match, the
    multi-view part adds nothing and the region variant reduces to monocular tiling, which is prior
    art.
  - **Scale gap.** The zoom view has 4× the pixel density of the scene views, a larger gap than the
    2× that broke G3. It does keep the same token grid as the scene views.
- **7Scenes GT depth** is Kinect depth not registered to the RGB camera; compare methods with each
  other, not with HiRoom.
- **Small number of scenes in places.** The large-view protocol has 7 scenes and the zoom 11, so
  their CIs are wide and INCONCLUSIVE is a likely, honest outcome. The view subsets of a scene are
  not independent samples and are not counted as such. *The limited run has 4 scenes in the probe and
  in Sec. 6 (§4.8).*
- **Gap in the pre-registered gates.** No gate covers cross-view consistency between the groups
  (focus–context pose pairs, scene-scale depth). The data show this is the largest effect. It is
  reported descriptively and not added as a gate after the fact.
- **The A/A comparison is not a pure null.** The reference view has a small systematic effect
  (§5), so the noise floor measured from it includes that shift.
- **May claim** (not found in the literature, report §3.4):
  - native mixed per-view resolutions in one training-free pass of a feed-forward multi-view model,
    bit-identical to the original at equal resolution (*the routing claim holds; that the outputs
    are useful does not, with the released weights*);
  - attention-mass balancing of the cross-view softmax as the training-free correction for unequal
    token counts (*exact as a mechanism, but harmful in every protocol*);
  - virtual zoom cameras through E_c as the region-level use (weak on its own).
- **Must cite, not claim:** token merging and pruning (Spark3R, PrePARE, TurboVGGT); trained
  high-resolution branches (HD-VGGT); monocular tiling (PatchFusion, BoostingDepth); perspective
  crops (Perspective Crop Layers).

## 7. Next steps

**Done after the limited run (2026-11-23):**
1. **Deviations log.** Entry 1 of [docs/FOVEATED_ANALYSIS_PLAN.md](../docs/FOVEATED_ANALYSIS_PLAN.md):
   the limited run was inspected, the plan's constants and gates are unchanged, everything added
   afterwards is exploratory, and Sec. 6 runs pose only.
2. **Exploratory diagnostics.** They are descriptive only: they appear in a separate block of
   `summary.md` and in columns marked `*`, and never change a gate.

   | Diagnostic | Where | Question it answers |
   |---|---|---|
   | Per-view depth scale; scale ratio context / focus; AbsRel with one scale per resolution group | probe, large views, Sec. 6 views | Do the two groups differ by a depth scale? If group-scale AbsRel drops back to about 0.05, the scale is the whole depth problem |
   | Cross-pair (focus–context) AUC@30, also with rotation-only and translation-only errors | probe (large views: max of the two only) | Are the cross pairs broken in rotation, or only in translation direction (which a scale offset would explain)? |
   | Full-resolution context ceiling next to G1b | probe | Could G1b pass at all with perfect context? |
   | `diag_up252`: context views reduced to 252-px detail but fed on the 504-px grid | probe (`--diagnostics`) | Same tokens as all-504. If the cross pairs recover, the cause is the patch-scale mismatch, not the lost detail |
   | Context sweep at 336 / 378 px | probe | How small a resolution gap still breaks the coupling? |
   | **Two-pass control**: all views at 252 px + the focus views alone at 504 px, mapped into one frame by a similarity fitted on the focus cameras | probe (`--diagnostics`, reuses passes already run), large views (`--two-pass`, one extra pass) | Does a simple non-novel alternative beat the one-pass method? Attention cost grows with the square of the token count, so it costs about as much as the foveated pass |

   Code: `scripts/foveated_eval.py` (`--diagnostics`, `--two-pass`, `depth_rows`, `two_pass_row`,
   `scaling_two_pass_row`), `src/da3/enhancements/foveated/metrics.py` (`pairwise_rot_trans_errors`,
   `align_cameras`, `transform_w2c`), `analysis.py` (`probe_exploratory`). Unit tests are in
   `tests/test_foveated_stats.py`. Notebook `06_foveated` turns all of this on, and sets Sec. 6 to
   pose only.

**Still to do:**
3. **Full confirmatory run** (`MAX_SCENES = 0`), run as a saved version so the outputs survive:
   - probe on both models, with the diagnostics;
   - Sec. 6 pose only on all scenes;
   - large-view protocol with the two-pass control, at 500 / 1000 views if time allows.

   If one session is not enough, split the parts with `RUN` and join them with `RESUME_FROM`.
4. **Region zoom session** (gate 2), as planned, read as described in §6.
5. **Decision.** If the full run confirms the limited one, close the training-free one-pass claim as
   a documented negative result, and follow report §5: promote a runner-up rather than force the
   idea.
   - Post-hoc fixes (bridge views, aligning the groups) cannot beat the two-pass control, because
     each group is also damaged internally.
   - The principled fix is training: a LoRA on the cross-view blocks, self-distilled from the
     uniform-504 pass on unlabelled multi-view video. It needs the LoRA setup that #3 needs, so it
     is a stretch goal after #3.

## 8. Reproduce

Notebook `06_foveated` on Kaggle (GPU); see [docs/ENHANCEMENTS_RUNBOOK.md](../docs/ENHANCEMENTS_RUNBOOK.md) §3.
The limited run used `MAX_SCENES = 2`, `SCALING_VIEWS = [100, 250]`, `SEC6_MODES = ["pose"]`, `RUN["zoom"] = False`,
without the diagnostics. From 2026-11-23 the notebook adds `--diagnostics` to the probe, `--two-pass` to the large-view
protocol, and the 336 / 378-px context sweep, all exploratory.
The main commands it runs per model:

```
python scripts/foveated_eval.py parity --weights official:da3-small --images <frames> --out <out>/parity_da3-small.json
python scripts/foveated_eval.py probe --official-src <official>/src --bench-root /tmp/da3bench --work-root <work> \
    --weights official:da3-small --datasets hiroom 7scenes --out <out>/probe_da3-small.json
python scripts/foveated_eval.py scaling --official-src <official>/src --bench-root /tmp/da3bench --work-root <work> \
    --weights official:da3-small --scenes chess office --n-views 100 250 500 1000 --balance --out <out>/scaling_da3-small.json
python scripts/eval_bench.py --official-src <official>/src --bench-root /tmp/da3bench --work-root <work> \
    --weights official:da3-small --datasets hiroom 7scenes --modes pose recon_unposed \
    --tag fov_da3-small_hiroom-7scenes__fov25_504-252 --enhancement foveated --variant fov25_504-252 \
    --foveated "fraction=0.25,context_res=252" --results-root <out>/results
python scripts/foveated_eval.py sec6views --official-src <official>/src --bench-root /tmp/da3bench --work-root <work> \
    --tag fov_da3-small_hiroom-7scenes__fov25_504-252 --config fov25_504-252 --datasets hiroom 7scenes \
    --out <out>/sec6views_fov_da3-small_hiroom-7scenes__fov25_504-252.json
python scripts/foveated_eval.py zoom --official-src <official>/src --bench-root /tmp/da3bench --work-root <work> \
    --weights official:da3-small --eth3d-zip /tmp/zips/eth3d.zip --fig-dir <out>/zoom_da3-small --out <out>/zoom_da3-small.json
python scripts/foveated_eval.py analyse --out-dir <out> --results-root <out>/results --models da3-small da3-base
```
