# Enhancement 2: Foveated DA3, mixed per-view resolutions in one pass

**Status (2026-11-23): full confirmatory run done. Decision: no-go on both models (documented
negative result).**
- Method and setup (§1–3) are unchanged.
- §4 holds the numbers of the **full run**:
  - all 37 probe scenes (30 HiRoom + 7 7Scenes);
  - Sec. 6 on all scenes, pose only;
  - large-view protocol at 100 / 250 / 500 / 1000 views (DA3-Small).

  Setting **[official]**: the released DA3-Small / DA3-Base weights, run by us. Plan SHA-256
  `3c7e751ad4c0`, the same as the registered `configs/foveated_analysis.yaml`.
- An earlier limited run (4 scenes, 2026-11-23) was only a smoke test. It is logged in the plan's
  deviations log, and its conclusions are confirmed here. Its numbers are replaced by the full run;
  they remain in the git history.
- **Pre-registered decision** (§4.6):
  - DA3-Base: G3 FAILs and G1 is INCONCLUSIVE;
  - DA3-Small: G1 FAILs and G3 is INCONCLUSIVE.

  A FAIL on a required gate means no-go. G2 (region zoom) was not run.
- **Main finding** (§5). DA3's depth scale depends on each view's token-grid size. In one mixed
  pass the two resolution groups come out at different depth scales: context / focus scale ratio ≈
  (context res / focus res)^0.7, about 0.6 at 252 px.
  - This splits the reconstruction into two inconsistently scaled pieces. Pose AUC@30 between a
    focus and a context view falls from 76–83 to about 22, almost entirely in translation.
  - Feeding the same low-detail context images on the full-resolution grid removes the effect
    completely.
  - Even a perfect fix has little to gain: full-resolution context improves the focus views by only
    3–5% AbsRel.

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
| Metrics | `metrics.py`, `src/da3/eval/gt_depth.py` | AbsRel / δ (median-aligned per view or region; per scene for consistency; per resolution group as a diagnostic), per-view alignment scale, pose AUC on view subsets (focus–focus, context–context, focus–context), rotation-only and translation-only pair errors, similarity alignment of camera sets (two-pass control), region point metrics (F1 at 0.5 / 1 / 2 % of the depth, Chamfer), GT depth decoders of HiRoom, 7Scenes, ETH3D |
| Statistics | `stats.py`, `analysis.py`, `configs/foveated_analysis.yaml` | Pre-registered analysis: scene-bootstrap CIs, three-way decisions, critical margins, A/A noise floor (capped), Bonferroni CIs for secondary configurations, stratified dev/test split for the rescue analysis, plus a separate block of exploratory comparisons that never affects a gate; unit tests on synthetic data with known answers (`tests/test_foveated_stats.py`) |
| Experiments | `scripts/foveated_eval.py` | `parity`, `probe` (`--diagnostics`), `scaling` (`--two-pass`), `sec6views`, `zoom`, `analyse` (gates, tables, decision figure, standard result records, `summary.md`) |

## 3. Experimental setup

- **Weights:** DA3-Small and DA3-Base, released (Apache-2.0); network unchanged; fp16 autocast on
  a Kaggle T4 as in the baseline; camera-head poses (API default) everywhere, so the pose source is
  fixed (one change at a time).
- **Reference view:** "first" (the paper protocol); view 0 is always a focus view.

| Protocol | Data | Configurations | Metrics |
|---|---|---|---|
| **Probe** (gates 1, 3) | HiRoom (30 scenes), 7Scenes (7); interleaved subsets of 16 evenly spaced views, every other view a focus view. 3 subsets per scene where the sequence is long enough (all 7Scenes scenes, 6 HiRoom scenes); the other 24 HiRoom scenes give 1 subset | uniform 504 / 378 / 336 / 280 / 252 px; A/A (504, reference view moved); focus views alone; focus 504 + context 252 (primary), 280, 252 balanced; primary with 3 random focus choices. Exploratory: context 336 / 378, `diag_up252`, two-pass control | AbsRel / δ1 of focus and context views, scene-scale and group-scale AbsRel, per-view depth scale; AUC@3 / AUC@30 of focus, context, cross and all pairs (cross also rotation-only / translation-only); tokens, time, memory |
| **Large views** | 7Scenes, all 7 sequences (1000 frames; stairs 500, so 6 scenes at 1000 views); 100, 250, 500, 1000 views per pass; 10% focus views; DA3-Small | all 504; foveated 504/252 (± balanced); uniform at equal tokens (294 px); fewer 504-px views at equal tokens (focus views included); all 252; focus views alone; two-pass control | time (median of 3 on 2 scenes), peak memory, out-of-memory, tokens, GFLOPs; AUC of focus, cross and all pairs; AbsRel of focus and context views; scale ratio |
| **Sec. 6** | HiRoom, 7Scenes; released harness (≤ 100 frames, seed 42, evo + RANSAC); pose only | uniform 504 (original DA3) / 336 / 252; foveated 25% and 10% at 504/252 (± balanced) | AUC@3 / AUC@30; per-view AbsRel and pose AUC of the focus / context views; paired CIs vs the original and vs the uniform curve at equal tokens |
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
- Quantities added after the limited run (cross pairs, scales, `diag_up252`, 336 / 378 px, two-pass)
  are **exploratory**: descriptive, with CIs, never gated (plan deviations log, entry 1).

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

## 4. Results [official], full run

### 4.0 The run
- **Session.** Notebook `06_foveated` on a Kaggle T4, with:
  - `MAX_SCENES = 0`;
  - `SCALING_VIEWS = [100, 250, 500, 1000]`;
  - `SEC6_MODES = ["pose"]`;
  - probe diagnostics and two-pass control on;
  - `RUN["zoom"] = False`.
- **Protocols run:**
  - parity, probe and Sec. 6 on both models;
  - large-view protocol on DA3-Small.
- **Not run:**
  - region zoom (gate 2);
  - large-view protocol on DA3-Base;
  - F1 / TSDF (deviation 1).
- **Outputs.** `summary.md`, the gate figure and the JSONs are in `configs/results/foveated/`. The
  standard result records are in `configs/results/foveated/results/{baseline,foveated}/`.

### 4.1 Routing check (G0): PASS on both models
| Check | DA3-Small | DA3-Base |
|---|---|---|
| equal resolution, fp32 / fp16 / reference selection (saddle_balanced) / posed through E_c / balance no-op: bit-identical | yes (max diff 0) | yes (max diff 0) |
| equal resolution split into two groups (round-off only) | 3.8e-6 rel. | 6.3e-6 rel. |
| mixed resolution, view-order equivariance | 1.0e-6 rel. | 1.6e-6 rel. |

The cross-view RoPE position is a true constant (special tokens 0, patches 1), independent of the
grid size (`src/da3/backbone/vit.py:178-186`). Time per patch token is the same for the foveated
pass, uniform 294 px and fewer 504-px views, so the routing adds no overhead.

### 4.2 Probe (16 views per subset, 8 focus; 37 scenes; means over scenes)
- AbsRel is median-aligned per view; "scene-scale" uses one scale per scene (cross-view
  consistency).
- "Cross" = focus–context pairs (exploratory).
- Uniform rows are evaluated on the same evenly spaced focus / context split.

| Model | Configuration | AbsRel focus | AbsRel context | AbsRel scene-scale | AUC@30 focus pairs | AUC@30 context pairs | AUC@30 cross pairs | AUC@30 all | AUC@3 all | tokens | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DA3-Small | all 504 | 0.0564 | 0.0604 | 0.0630 | 78.0 | 75.2 | 76.5 | 76.5 | 10.8 | 16 290 | 0.82 |
| DA3-Small | all 378 | 0.0568 | 0.0611 | 0.0628 | 77.5 | 73.9 | 75.3 | 75.5 | 9.9 | 9 141 | 0.41 |
| DA3-Small | all 336 | 0.0592 | 0.0633 | 0.0653 | 75.6 | 72.3 | 73.7 | 73.8 | 8.0 | 7 239 | 0.32 |
| DA3-Small | all 280 | 0.0620 | 0.0666 | 0.0689 | 71.1 | 67.2 | 68.6 | 68.9 | 5.2 | 5 027 | 0.22 |
| DA3-Small | all 252 | 0.0693 | 0.0739 | 0.0765 | 66.2 | 61.1 | 63.1 | 63.3 | 2.8 | 4 099 | 0.19 |
| DA3-Small | A/A (all 504, reference moved) | 0.0547 | 0.0576 | 0.0585 | 77.8 | 75.8 | 76.1 | 76.4 | 8.9 | 16 290 | 0.79 |
| DA3-Small | focus views alone | 0.0582 | – | 0.0628 | 76.2 | – | – | 76.2 | 11.8 | 8 126 | 0.36 |
| DA3-Small | **504 / 252** (primary) | 0.0573 | 0.0868 | **0.261** | 76.8 | 57.5 | **22.0** | 42.5 | 3.6 | 10 180 | 0.47 |
| DA3-Small | 504 / 280 | 0.0570 | 0.0788 | 0.222 | 77.3 | 63.0 | 25.2 | 45.7 | 4.0 | 10 650 | 0.49 |
| DA3-Small | 504 / 336* | 0.0565 | 0.0715 | 0.161 | 79.2 | 70.8 | 34.3 | 52.9 | 5.0 | 11 750 | 0.55 |
| DA3-Small | 504 / 378* | 0.0555 | 0.0660 | 0.121 | 80.0 | 74.3 | 43.4 | 58.8 | 5.7 | 12 710 | 0.60 |
| DA3-Small | 504 / 252 balanced | 0.0594 | 0.0843 | 0.258 | 74.2 | 57.3 | 22.5 | 42.2 | 2.9 | 10 180 | 0.51 |
| DA3-Small | `diag_up252`* (252 detail, 504 grid) | 0.0581 | 0.0597 | 0.0632 | 79.4 | 76.2 | 77.6 | 77.7 | 11.6 | 16 290 | 0.79 |
| DA3-Small | two-pass 252 + focus* | 0.0582 | 0.0739 | 0.0781 | 76.2 | 61.9 | 65.5 | 67.1 | 5.6 | 12 230 | 0.55 |
| DA3-Base | all 504 | 0.0448 | 0.0464 | 0.0471 | 84.5 | 83.2 | 83.4 | 83.6 | 19.4 | 16 290 | 1.96 |
| DA3-Base | all 378 | 0.0450 | 0.0458 | 0.0462 | 84.6 | 82.9 | 83.4 | 83.6 | 21.6 | 9 141 | 1.04 |
| DA3-Base | all 336 | 0.0457 | 0.0470 | 0.0474 | 84.6 | 82.7 | 83.4 | 83.5 | 20.8 | 7 239 | 0.82 |
| DA3-Base | all 280 | 0.0473 | 0.0496 | 0.0494 | 81.7 | 79.3 | 79.8 | 80.2 | 13.8 | 5 027 | 0.53 |
| DA3-Base | all 252 | 0.0500 | 0.0559 | 0.0546 | 78.0 | 73.9 | 75.7 | 75.8 | 9.0 | 4 099 | 0.44 |
| DA3-Base | A/A (all 504, reference moved) | 0.0445 | 0.0440 | 0.0452 | 82.7 | 81.5 | 82.2 | 82.2 | 17.9 | 16 290 | 1.92 |
| DA3-Base | focus views alone | 0.0470 | – | 0.0484 | 82.5 | – | – | 82.5 | 21.6 | 8 126 | 0.90 |
| DA3-Base | **504 / 252** (primary) | 0.0464 | 0.0759 | **0.274** | 77.5 | 56.3 | **22.9** | 43.0 | 5.2 | 10 180 | 1.16 |
| DA3-Base | 504 / 280 | 0.0462 | 0.0682 | 0.230 | 78.6 | 62.8 | 26.1 | 46.5 | 5.4 | 10 650 | 1.21 |
| DA3-Base | 504 / 336* | 0.0445 | 0.0589 | 0.164 | 81.1 | 73.3 | 35.1 | 54.3 | 6.7 | 11 750 | 1.37 |
| DA3-Base | 504 / 378* | 0.0441 | 0.0533 | 0.123 | 83.8 | 78.5 | 44.4 | 61.2 | 8.8 | 12 710 | 1.49 |
| DA3-Base | 504 / 252 balanced | 0.0487 | 0.0732 | 0.265 | 73.3 | 56.9 | 23.5 | 42.5 | 4.0 | 10 180 | 1.23 |
| DA3-Base | `diag_up252`* (252 detail, 504 grid) | 0.0453 | 0.0453 | 0.0469 | 86.9 | 84.8 | 84.9 | 85.3 | 23.2 | 16 290 | 1.93 |
| DA3-Base | two-pass 252 + focus* | 0.0470 | 0.0559 | 0.0584 | 82.5 | 73.8 | 76.3 | 77.2 | 12.1 | 12 230 | 1.34 |

\* exploratory (added after the limited run).

**Random focus choices.** The primary configuration with 3 random focus sets behaves like the evenly
spaced choice, slightly worse:

| Model | Focus AbsRel | AUC@30 focus | AUC@30 context | AUC@30 cross |
|---|---|---|---|---|
| DA3-Small | 0.0572–0.0619 | 73.4–76.0 | 55.8–58.3 | 21.5–23.4 |
| DA3-Base | 0.0453–0.0520 | 74.3–76.7 | 54.7–56.5 | 22.1–24.1 |

### 4.3 Large-view protocol (DA3-Small, T4, 7Scenes, 10% focus views)
Means over scenes: 7 scenes, 6 at 1000 views. Speed-up is relative to all views at 504 px. All
views at 504 px run out of memory at 500 and 1000 views.

| Views | Configuration | tokens | time (s) | peak memory (GB) | speed-up | AUC@30 focus pairs | AUC@30 all | AUC@30 cross | AbsRel focus | scale ctx / focus |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | all 504 | 9.72e4 | 9.47 | 3.73 | 1× | 81.7 | 78.6 | 79.2 | 0.0705 | 1.01 |
| 100 | foveated 504/252 | 3.24e4 | 1.85 | 1.63 | 5.1× | 63.1 | 54.3 | 20.1 | 0.0812 | 0.65 |
| 100 | foveated balanced | 3.24e4 | 2.21 | 1.63 | 4.3× | 58.3 | 52.5 | 20.4 | 0.0873 | 0.65 |
| 100 | uniform 294, equal tokens | 3.36e4 | 1.94 | 1.42 | 4.9× | 76.5 | 73.3 | 74.0 | 0.0746 | 1.00 |
| 100 | 33 views at 504, equal tokens | 3.21e4 | 1.82 | 1.93 | 5.2× | **81.9** | **80.1** | 79.6 | 0.0703 | 1.01 |
| 100 | all 252 | 2.52e4 | 1.35 | 1.12 | 7.0× | 72.9 | 69.0 | 69.6 | 0.0815 | 1.01 |
| 100 | focus views alone | 9.72e3 | 0.44 | 1.30 | 21.5× | 81.1 | – | – | 0.0710 | – |
| 100 | two-pass* | 3.49e4 | 1.79 | 1.30 | 5.3× | 81.1 | 69.6 | 71.1 | 0.0710 | 0.96 |
| 250 | all 504 | 2.43e5 | 55.8 | 8.29 | 1× | 78.4 | 77.3 | 77.2 | 0.0717 | 1.00 |
| 250 | foveated 504/252 | 8.10e4 | 6.91 | 2.54 | 8.1× | 62.3 | 52.9 | 18.7 | 0.0818 | 0.65 |
| 250 | foveated balanced | 8.10e4 | 8.91 | 2.54 | 6.3× | 55.6 | 49.5 | 18.5 | 0.0887 | 0.64 |
| 250 | uniform 294, equal tokens | 8.40e4 | 7.34 | 3.00 | 7.6× | 74.8 | 72.8 | 73.0 | 0.0750 | 1.00 |
| 250 | 83 views at 504, equal tokens | 8.07e4 | 6.78 | 3.27 | 8.2× | **79.1** | **78.3** | 77.9 | 0.0709 | 1.00 |
| 250 | all 252 | 6.30e4 | 4.62 | 2.30 | 12.1× | 70.9 | 67.8 | 68.5 | 0.0812 | 1.00 |
| 250 | focus views alone | 2.43e4 | 1.27 | 1.71 | 43.9× | 78.9 | – | – | 0.0709 | – |
| 250 | two-pass* | 8.73e4 | 5.89 | 2.30 | 9.5× | 78.9 | 68.7 | 70.1 | 0.0709 | 0.96 |
| 500 | all 504 | OOM | – | – | – | – | – | – | – | – |
| 500 | foveated 504/252 | 1.62e5 | 25.3 | 4.87 | – | 61.0 | 51.5 | 17.9 | 0.0824 | 0.64 |
| 500 | foveated balanced | 1.62e5 | 30.7 | 4.87 | – | 52.5 | 47.0 | 17.6 | 0.0911 | 0.64 |
| 500 | uniform 294, equal tokens | 1.68e5 | 26.6 | 5.80 | – | 72.4 | 71.5 | 71.4 | 0.0761 | 1.00 |
| 500 | 166 views at 504, equal tokens | 1.61e5 | 24.5 | 5.57 | – | **78.5** | **78.0** | 77.7 | 0.0711 | 1.00 |
| 500 | all 252 | 1.26e5 | 15.2 | 4.41 | – | 67.9 | 67.4 | 67.2 | 0.0821 | 1.00 |
| 500 | focus views alone | 4.86e4 | 3.11 | 2.38 | – | 78.9 | – | – | 0.0705 | – |
| 500 | two-pass* | 1.75e5 | 18.3 | 4.41 | – | 78.9 | 68.5 | 70.2 | 0.0705 | 0.97 |
| 1000 | all 504 | OOM | – | – | – | – | – | – | – | – |
| 1000 | foveated 504/252 | 3.24e5 | 97.9 | 9.54 | – | 62.3 | 52.2 | 18.0 | 0.0783 | 0.64 |
| 1000 | foveated balanced | 3.24e5 | 111.4 | 9.54 | – | 53.3 | 47.6 | 17.4 | 0.0884 | 0.63 |
| 1000 | uniform 294, equal tokens | 3.36e5 | 105.5 | 11.42 | – | 76.6 | 75.9 | 75.9 | 0.0716 | 1.00 |
| 1000 | 333 views at 504, equal tokens | 3.24e5 | 99.0 | 10.99 | – | 80.2 | **79.8** | 79.6 | 0.0684 | 1.00 |
| 1000 | all 252 | 2.52e5 | 59.9 | 8.62 | – | 73.4 | 72.6 | 72.6 | 0.0739 | 1.00 |
| 1000 | focus views alone | 9.72e4 | 9.39 | 3.73 | – | 81.5 | – | – | 0.0668 | – |
| 1000 | two-pass* | 3.49e5 | 69.3 | 8.62 | – | 81.5 | 74.6 | 75.4 | 0.0668 | 0.97 |

Foveated vs each alternative, paired over scenes (95% CIs). Each cell gives Δ AUC@30 of the focus
pairs, foveated − other, in points / AbsRel ratio of the focus views, foveated / other. Two-pass
focus metrics equal those of the focus views alone by construction.

| Views | vs all 504 | vs uniform 294 | vs fewer views at 504 | vs all 252 | vs focus alone / two-pass | vs balanced |
|---|---|---|---|---|---|---|
| 100 | −18.6 [−22.4, −14.6] / 1.151 [1.130, 1.168] | −13.4 [−17.2, −9.9] / 1.088 [1.044, 1.141] | −18.8 [−22.3, −15.1] / 1.155 [1.134, 1.175] | −9.8 [−14.1, −5.4] / 0.996 [0.923, 1.082] | −18.0 [−22.0, −14.0] / 1.143 [1.105, 1.175] | +4.7 [+3.7, +5.7] / 0.930 [0.898, 0.956] |
| 250 | −16.1 [−18.8, −13.3] / 1.141 [1.107, 1.174] | −12.5 [−17.2, −7.9] / 1.090 [1.057, 1.135] | −16.8 [−19.6, −13.8] / 1.153 [1.132, 1.177] | −8.6 [−12.8, −4.2] / 1.007 [0.941, 1.091] | −16.6 [−19.3, −13.5] / 1.153 [1.128, 1.180] | +6.7 [+5.6, +7.7] / 0.922 [0.886, 0.950] |
| 500 | OOM | −11.4 [−16.3, −6.3] / 1.082 [1.031, 1.141] | −17.5 [−20.6, −14.0] / 1.159 [1.140, 1.179] | −6.9 [−12.3, −1.0] / 1.004 [0.935, 1.091] | −17.9 [−21.1, −14.3] / 1.168 [1.149, 1.191] | +8.5 [+7.0, +9.8] / 0.904 [0.855, 0.943] |
| 1000 | OOM | −14.3 [−19.5, −9.6] / 1.094 [1.037, 1.150] | −17.9 [−21.6, −14.0] / 1.145 [1.124, 1.169] | −11.0 [−15.7, −6.8] / 1.061 [0.992, 1.137] | −19.1 [−23.3, −15.2] / 1.173 [1.153, 1.198] | +9.0 [+7.2, +10.7] / 0.886 [0.827, 0.935] |

Time of the alternative / time of the foveated pass, on the timed scenes:

| Views | uniform 294 | fewer views at 504 | all 252 | two-pass |
|---|---|---|---|---|
| 100 | 1.03 | 0.95 | 0.71 | 0.94 |
| 250 | 1.05 | 0.97 | 0.67 | 0.85 |
| 500 | 1.07 | 0.97 | 0.62 | 0.74 |
| 1000 | 1.08 | 1.01 | 0.60 | 0.70 |

### 4.4 Sec. 6 protocol (released harness, ≤ 100 frames, seed 42, all scenes, pose only)
Camera-head poses. The differences in brackets are relative to the original DA3 (504 px).

| Model | Configuration | HiRoom AUC@3 / AUC@30 | 7Scenes AUC@3 / AUC@30 |
|---|---|---|---|
| DA3-Small | all 504 (original) | 9.7 / 75.2 | 14.6 / 78.2 |
| DA3-Small | all 336 | 7.1 / 72.8 (−2.4) | 10.6 / 75.3 (−2.9) |
| DA3-Small | all 252 | 2.3 / 61.7 (−13.5) | 5.1 / 68.8 (−9.4) |
| DA3-Small | foveated 10% | 1.1 / 49.6 (−25.6) | 3.2 / 55.6 (−22.6) |
| DA3-Small | foveated 25% | 1.4 / 44.1 (−31.1) | 2.4 / 46.9 (−31.3) |
| DA3-Small | foveated 25% balanced | 1.2 / 42.9 (−32.3) | 1.9 / 45.2 (−33.0) |
| DA3-Base | all 504 (original) | 18.8 / 83.3 | 21.3 / 82.4 |
| DA3-Base | all 336 | 21.5 / 83.4 (+0.1) | 16.0 / 81.0 (−1.4) |
| DA3-Base | all 252 | 8.3 / 74.6 (−8.7) | 10.5 / 77.4 (−5.1) |
| DA3-Base | foveated 10% | 1.4 / 47.3 (−36.0) | 4.2 / 61.4 (−21.0) |
| DA3-Base | foveated 25% | 1.5 / 42.7 (−40.6) | 2.5 / 52.6 (−29.8) |
| DA3-Base | foveated 25% balanced | 1.3 / 41.8 (−41.4) | 2.7 / 54.0 (−28.4) |

Per-view analysis of the same runs, with paired 95% scene-bootstrap CIs over 37 scenes. Each run is
compared on the focus set of its own fraction. "Equal tokens" = the uniform curve interpolated per
scene in log tokens. AUC@30 is over all pairs.

| Model | Configuration | Focus AbsRel vs original | Focus AbsRel vs equal tokens | AUC@30 vs original (pts) | AUC@30 vs equal tokens (pts) |
|---|---|---|---|---|---|
| DA3-Small | foveated 10% | 1.091 [0.990, 1.198] | 0.936 [0.849, 1.020] | −25.0 [−28.3, −21.8] | −16.8 [−19.7, −13.8] |
| DA3-Small | foveated 25% | 1.029 [0.942, 1.125] | 0.967 [0.909, 1.022] | −31.2 [−33.8, −28.6] | −27.9 [−30.2, −25.6] |
| DA3-Small | foveated 25% balanced | 1.086 [1.004, 1.174] | 1.021 [0.961, 1.080] | −32.4 [−35.4, −29.6] | −29.1 [−31.8, −26.6] |
| DA3-Base | foveated 10% | 1.113 [1.000, 1.243] | 1.042 [0.946, 1.151] | −33.2 [−37.1, −29.3] | −28.5 [−32.2, −24.8] |
| DA3-Base | foveated 25% | 1.098 [0.999, 1.222] | 1.055 [0.939, 1.178] | −38.6 [−41.6, −35.5] | −37.9 [−40.9, −34.8] |
| DA3-Base | foveated 25% balanced | 1.172 [1.049, 1.322] | 1.126 [1.005, 1.273] | −39.0 [−42.5, −35.4] | −38.3 [−41.8, −34.7] |

Focus-view AbsRel on the 10% focus set:

| Model | all 504 | all 336 | all 252 | foveated 10% |
|---|---|---|---|---|
| DA3-Small | 0.0614 | 0.0623 | 0.0735 | 0.0692 |
| DA3-Base | 0.0539 | 0.0545 | 0.0595 | **0.0632** (worse than all 252) |

### 4.5 Region zoom (ETH3D)
_Not run (gate 2)._

### 4.6 Gates (pre-registered decision, 37 scenes)
Decision, estimate [95% CI], and the margins at which the decision changes.

| Gate | DA3-Small | DA3-Base |
|---|---|---|
| A/A ratio (depth / pose) → noise floor | 0.962 [0.934, 0.991] / 0.998 [0.978, 1.018] → 6.6% / 2.2% | 0.970 [0.931, 1.002] / 0.983 [0.963, 1.001] → 6.9% / 3.7% |
| Margin used (depth / pose) | 6% (noise floor above the cap) / 3% | 6% (noise floor above the cap) / 3.7% |
| G0 routing parity | PASS (7/7) | PASS (7/7) |
| G1a headroom (u252 / u504 AbsRel) | PASS, 1.227 [1.066, 1.378] | INCONCLUSIVE, 1.161 [0.960, 1.338] |
| G1b context informative (with / without context) | FAIL, 0.984 [0.963, 1.004] (FAIL for any margin > 3.7%) | INCONCLUSIVE, 0.986 [0.928, 1.043] (never PASS; FAIL for margin > 7.2%) |
| **G1** | **FAIL** | **INCONCLUSIVE** |
| G3 C1 focus AbsRel (504/252 / u504) | PASS, 1.016 [0.986, 1.051] | INCONCLUSIVE, 1.034 [0.988, 1.093] |
| G3 C2 context AUC@30 (504/252 / u252) | INCONCLUSIVE, 0.940 [0.874, 1.003] | FAIL, 0.763 [0.709, 0.814] (FAIL for any margin < 18.6%) |
| **G3 primary** | **INCONCLUSIVE** | **FAIL** |
| G3 secondary 504/280 (97.5% CIs) | INCONCLUSIVE (C1 1.010 PASS; C2 0.938 [0.876, 1.004] INC) | FAIL (C1 1.029 INC; C2 0.791 [0.732, 0.845] FAIL) |
| G3 secondary balanced (97.5% CIs) | INCONCLUSIVE (C1 1.053 INC; C2 0.938 INC) | FAIL (C1 1.086 [1.012, 1.179] INC; C2 0.770 FAIL) |
| G3 robustness: 3 random focus choices | INCONCLUSIVE ×3 (C2 0.914–0.938) | FAIL ×3 (C2 0.732–0.746) |
| G3 rescue (dev 18 scenes / test 19) | FAIL: no mitigation keeps context AUC@30 within the margin on dev | FAIL (same) |
| G2 region zoom | not run | not run |
| **Overall** | **no-go** (G1 FAIL) | **no-go** (G3 FAIL) |

Secondary G1 quantities:
- DA3-Small: AUC@30 u252 / u504 0.827 [0.796, 0.857]; focus-pair AUC@30 with / without context 1.007 [0.978, 1.037].
- DA3-Base: AUC@30 u252 / u504 0.907 [0.882, 0.932]; focus-pair AUC@30 with / without context 0.939 [0.895, 0.980].

**Reading DA3-Small's G3 INCONCLUSIVE.** C2 only compares context–context pairs, and the cross
pairs are where the method breaks. Over all pairs the primary configuration has AUC@30 42.5,
against 63.3 for all-252 and 76.5 for all-504. G3 INCONCLUSIVE therefore does not mean "maybe it
works". It means the pre-registered gate does not look at the effect (§6).

### 4.7 Exploratory diagnostics (added after the limited run; descriptive, no decision)
95% scene-bootstrap CIs over 37 scenes, from `summary.md`.

**Where the pose error is: translation, not rotation.** Cross-pair AUC@30 of the primary
configuration minus all-252, in points:

| | DA3-Small | DA3-Base |
|---|---|---|
| max(rotation, translation) error | −41.1 [−45.4, −36.5] | −52.8 [−56.1, −49.6] |
| rotation only | +1.3 [+0.0, +2.8] | −3.6 [−4.7, −2.6] |
| translation only | −42.6 [−46.8, −38.0] | −54.1 [−57.3, −50.8] |

The context cameras are oriented correctly but placed wrongly relative to the focus cameras. Raising
the context resolution shrinks the gap:

| Context resolution | 252 | 280 | 336 | 378 |
|---|---|---|---|---|
| DA3-Small: cross-pair AUC@30 − all 252 | −41.1 | −37.9 | −28.8 | −19.6 |
| DA3-Base: cross-pair AUC@30 − all 252 | −52.8 | −49.6 | −40.7 | −31.4 |

**The two groups are at different depth scales, following a power law.** The scale ratio is the
median per-view alignment scale, median(gt) / median(pred), of the context views over that of the
focus views (1 = consistent).

| | 252 | 280 | 336 | 378 | all uniform runs | `diag_up252` |
|---|---|---|---|---|---|---|
| resolution ratio context / focus | 0.50 | 0.56 | 0.67 | 0.75 | 1 | 1 (grid) |
| DA3-Small: scale ratio | 0.62 | 0.67 | 0.76 | 0.82 | 1.00 | 1.00 |
| DA3-Base: scale ratio | 0.60 | 0.65 | 0.74 | 0.81 | 0.99–1.00 | 0.99 |
| implied exponent γ, ratio = (res ratio)^γ, Small / Base | 0.69 / 0.73 | 0.69 / 0.73 | 0.69 / 0.73 | 0.67 / 0.74 | – | – |

- **Direction.** The context views predict depths about 1.6× too large relative to the focus views.
- **Stability across scenes.** Per scene (offline from `probe_*.json`) the ratio at 252 px is tight:
  - 10th–90th percentile 0.60–0.66 (Small) and 0.56–0.68 (Base);
  - the per-scene γ has a standard deviation of 0.06 (Small) and 0.13 (Base).
- **Dataset dependence.** The ratio depends slightly on the dataset (HiRoom 0.61 / 0.59, 7Scenes
  0.64 / 0.65 for Small / Base).
- **Within a group.** Per-view scales vary by only about 3%, close to their spread in uniform
  passes (2–3%).

**With one scale per group, the depth is close to normal.** AbsRel of all views with one scale per
resolution group, primary configuration / all-252:

| | DA3-Small | DA3-Base |
|---|---|---|
| ratio | 1.017 [0.960, 1.086] | 1.213 [1.115, 1.345] |

For DA3-Small the scale gap is almost the whole depth problem. DA3-Base is also degraded inside the
groups: its context-pair AUC@30 is 56.3 against 73.9 at all-252.

**Grid, not detail.** `diag_up252` feeds the context views with 252-px detail on the 504-px grid
(same tokens as all-504):

| `diag_up252` vs primary 504/252 | DA3-Small | DA3-Base |
|---|---|---|
| cross-pair AUC@30 (pts) | +55.6 [+51.7, +59.3] | +62.0 [+58.8, +65.2] |
| context-pair AUC@30 (pts) | +18.8 [+15.0, +23.0] | +28.4 [+24.4, +32.9] |
| focus AbsRel / u504 | 1.029 [0.998, 1.058] | 1.011 [0.972, 1.046] |

The scale ratio returns to 1.00 and every pose and depth number returns to the all-504 level. The
loss of image detail does not break the coupling: the different token grid does. Base's
`diag_up252` even scores above all-504 on pose (AUC@30 85.3 vs 83.6, AUC@3 23.2 vs 19.4). That
difference is about the size of the A/A shift and has no CI, so it is not read as a gain.

**Full-resolution context ceiling for G1b.** Focus views with 8 extra 504-px views vs the focus
views alone:

| | DA3-Small | DA3-Base |
|---|---|---|
| focus AbsRel ratio | 0.969 [0.934, 0.998] | 0.953 [0.901, 0.997] |
| focus-pair AUC@30 (pts) | +1.8 [+0.2, +3.5] | +2.0 [+0.3, +3.7] |

Even perfect context improves the focus views by only 3–5% AbsRel and about 2 AUC points: at or
below the 6% margin G1b needed. With 252-px context the gain is about 1.5% (0.984 / 0.986).

**Two-pass control.** All views at 252 px, plus the focus views alone at 504 px, aligned on the
focus cameras. Two-pass vs the primary one-pass configuration:

| | DA3-Small | DA3-Base |
|---|---|---|
| focus AbsRel ratio | 1.016 [0.996, 1.039] | 1.014 [0.959, 1.077] |
| cross-pair AUC@30 (pts) | +43.5 [+38.3, +48.4] | +53.4 [+48.8, +57.6] |
| all-pair AUC@30 (pts) | +24.5 [+21.1, +27.7] | +34.2 [+30.9, +36.9] |
| scene-scale AbsRel ratio | 0.299 [0.260, 0.346] | 0.213 [0.174, 0.257] |

**Share of the 252 → 504 px depth gain kept by the focus views**, computed as
(AbsRel_all252 − AbsRel_foveated) / (AbsRel_all252 − AbsRel_all504):

| Focus share | Protocol | DA3-Small | DA3-Base |
|---|---|---|---|
| 50% | probe | 93% | 70% |
| 25% | Sec. 6 | 60% | 9% |
| 10% | Sec. 6 | 36% | worse than all 252 |
| 10% | large views, 100 | 3% | – |
| 10% | large views, 250 | worse than all 252 | – |
| 10% | large views, 500 / 1000 | worse than all 252 (no all-504 reference) | – |

### 4.8 How far the numbers can be trusted
- **The protocol is as registered.** The plan SHA matches and the gates use all 37 scenes.
- **Two deviations.**
  - 24 of the 30 HiRoom scenes are too short for 3 interleaved 16-view subsets and contribute 1.
    The unit is the scene, so the test is valid, but within-scene noise is higher than planned.
  - G2 and the DA3-Base large-view protocol were not run.
- **The depth margin is capped.** On both models the A/A depth noise floor (6.6–6.9%) is above the
  6% cap, so the depth criteria cannot resolve the planned 3% tolerance. This affects only
  borderline depth labels (G1, G3 C1). The decisive effects are pose effects far outside any margin.
- **The main effects are far outside the CIs:**
  - cross-pair AUC@30 −41 to −53 points;
  - Sec. 6 AUC@30 −21 to −41 points;
  - scene-scale AbsRel 4–6× higher;
  - large-view focus AUC@30 −11 to −19 points against every equal-token alternative.
- **The large-view protocol has 7 scenes (6 at 1000 views),** so its CIs are wide, but they exclude
  0 against every equal-token alternative.
- **The scale analysis covers depth only.** The probe records store per-view depth scales and pair
  errors, not predicted camera positions. Whether rescaling the context group would also repair the
  poses was not measured.

## 5. Discussion

**The implementation is not the cause.**
- The routing is bit-identical at equal resolution and only round-off when same-size views are split
  into groups.
- Every uniform resolution behaves normally (scale ratio 1.00).
- Feeding the context views on the full grid (`diag_up252`) restores everything, through the same
  code path.

What fails is the released network's behaviour on an input it never saw in training: views with
different token grids in one pass.

**Main finding: DA3's depth scale depends on the token grid, so the two groups form two
reconstructions.**
- Per-view depth shape survives: median-aligned AbsRel stays near the all-252 level.
- With one scale per scene, AbsRel grows from about 0.05–0.06 to 0.26–0.27. The context views
  predict depths about 1.6× too large relative to the focus views.
- The scale ratio follows (context res / focus res)^γ with γ ≈ 0.7 (0.69 Small, 0.73 Base). It is
  nearly constant within a scene and across the 37 scenes.
- Pose errors follow from this:
  - focus–context rotations stay near normal;
  - translation directions collapse, AUC@30 from 65–77 to about 22;
  - AUC@30 over all pairs drops to 42–43, below all-252 (63–76).
- `diag_up252` separates the two candidate causes. With the same reduced detail on the full grid,
  the groups stay consistent. The cause is the token grid (how many patches cover the image, i.e.
  the image area per patch), not the missing detail.
- Hypothesis, not tested: the network infers depth scale partly from how large structures appear
  in patch units. In a uniform pass this cancels out, because the normalisation is shared. In a
  mixed pass, objects in the coarse views span fewer patches and are placed farther away. The
  exponent below 1 suggests that cross-view attention partly, but not fully, reconciles the groups.
- Attention-mass balancing does not change the scale ratio (0.61–0.63) or the cross pairs. This
  fits a grid-scale problem, not an attention-mass problem.

**Mixing also degrades the groups internally, more for DA3-Base and more as the focus share
drops.**
- Context-pair AUC@30, mixed vs all 252: Base 56.3 vs 73.9; Small 57.5 vs 61.1.
- Group-scale AbsRel vs all 252: Base 1.21×; Small 1.02×.
- The focus views keep:
  - 70–93% of the 252 → 504 px depth gain at 50% focus views;
  - 9–60% at 25%;
  - nothing at 10% in the large-view protocol.
- At 10% focus views, the share the method was designed for, the foveated focus views are no better
  than with every view at 252 px (AbsRel ratio 1.00–1.06). DA3-Base is worse than all 252 at 10%
  in Sec. 6.

**Even a working foveation would have little to gain on these benchmarks.**
- Context views at full resolution improve focus-view AbsRel by 3–5% and focus-pair AUC@30 by
  about 2 points (G1b ceiling). In the large-view protocol, the focus views alone (10% of the views)
  match all views at 504 px on the focus pairs (81.1 vs 81.7 at 100 views).
- Plain downscaling is almost free. For DA3-Base, all 378 px matches all 504 px in the probe:
  - AbsRel 0.0450 vs 0.0448;
  - AUC@30 83.6 vs 83.6;
  - 56% of the tokens and 53% of the time.

  At 336 px Sec. 6 HiRoom is unchanged (83.4 vs 83.3). The report's risk "uniform 336 px may tie"
  came true.
- **Ranking at equal tokens** (large views, all four view counts), by focus-pair AUC@30:
  1. fewer views at 504 (78.5–81.9);
  2. uniform 294 (72.4–76.6);
  3. foveated (61.0–63.1).

  All 252 (67.9–73.4) uses 22% fewer tokens and is still better than foveated. "Runs where all-504
  goes out of memory" is not specific to foveation: every equal-token alternative fits at 1000 views.
- **The two-pass control** (not novel) beats the one-pass method on every pose and consistency
  metric, by +24 to +53 AUC points. It is also faster in the large-view protocol (0.70–0.94× the
  foveated time). This is the bar any repaired one-pass method would have to beat.

**The mitigations do not close the gap.**
- 280-px context views improve the cross pairs by 3 points.
- Attention-mass balancing:
  - makes the focus views worse in every protocol (focus AbsRel +4–9% in the probe, −5 to −9 AUC
    points in the large-view protocol);
  - costs 7–30% more time.
- The rescue analysis selects nothing.

**Side observation: the A/A comparison is not a pure null.** Moving the reference to a middle focus
view changes AbsRel systematically on both models: 0.962 [0.934, 0.991] for Small, 0.970
[0.931, 1.002] for Base. The reference-view choice has a real effect on DA3's depth, of about 3–4%.
The A/A noise floor therefore includes that shift, and that is why it exceeds the cap.

## 6. Limitations and honest claims

- **Cross-scale token grids were never trained.** DA3 saw one resolution per batch with the short
  side ≥ 280 px; a 252-px context view is outside that range. *Full run: the risk materialised as a
  systematic depth-scale offset between the groups (§4.7). It is still present at 378 px, inside
  the training range, so it is about mixing grids in one pass, not about the absolute resolution.*
- **Uniform downscaling may tie.** *Full run: uniform downscaling, fewer full-resolution views and
  the two-pass control all do better than foveation at equal tokens.*
- **The efficiency bar is Spark3R** (training-free token reduction on DA3, ~17× at 1000 frames,
  accuracy unchanged). It was not re-run, because foveation already loses to simpler equal-token
  baselines.
- **Sec. 6 is not token-bound at ≤ 100 views** on a T4, so the paper's tables do not show the
  efficiency gain. The large-view and region protocols are new and never mixed into the Sec. 6
  tables. F1 was not run: with two depth scales, TSDF fusion would mostly measure the scale gap.
- **Gap in the pre-registered gates.** No gate covers cross-view consistency between the groups
  (focus–context pose pairs, scene-scale depth), and the data show this is the largest effect. G3
  C2 uses only context–context pairs, which is why DA3-Small's G3 is INCONCLUSIVE while its
  all-pair pose AUC falls by 34 points. The consistency results are reported descriptively and were
  not added as a gate after the fact.
- **The A/A comparison is not a pure null.** The reference view has a systematic effect of about
  3–4% on AbsRel (§5). The noise floor measured from it includes that shift.
- **Unequal subsets per scene.** 24 HiRoom scenes have 1 probe subset instead of 3 (§4.8).
- **The scale analysis covers depth only.** That one scale per group explains the depth error is
  measured. That it would also repair the camera positions is plausible (rotations are intact, the
  translation errors are what a group scale offset would cause), but it is not measured.
- **Region zoom (G2) is untested.** It does not change the decision, because G1 / G3 already
  give no-go. Points to read if it is ever run:
  - **R_v.** The R_v error of `zoom_unposed` is the informative one: D_C may copy an E_c input.
  - **Multi-view contribution.** Compare `zoom_posed_da3cams` with `zoom_mono`. If they match, the
    region variant reduces to monocular tiling, which is prior art.
  - **Grid.** The zoom view keeps the scene views' token grid, so it does not hit the grid-scale
    mismatch found here. It does, however, have 4× the pixel density.
- **7Scenes GT depth** is Kinect depth not registered to the RGB camera; compare methods with each
  other, not with HiRoom.
- **May claim:**
  - native mixed per-view resolutions in one training-free pass of a feed-forward multi-view model,
    bit-identical to the original at equal resolution (*the routing claim holds; that the outputs
    are useful does not, with the released weights*);
  - attention-mass balancing of the cross-view softmax as the training-free correction for unequal
    token counts (*exact as a mechanism, but harmful in every protocol*);
  - **the measured finding:** with the released weights, DA3's depth scale depends on the per-view
    token grid in a mixed pass, as (resolution ratio)^≈0.7, while rotations stay consistent. This
    is an observation about DA3, not a method. Its novelty was not checked separately.
- **Must cite, not claim:**
  - token merging and pruning (Spark3R, PrePARE, TurboVGGT);
  - trained high-resolution branches (HD-VGGT);
  - monocular tiling (PatchFusion, BoostingDepth);
  - perspective crops (Perspective Crop Layers);
  - the two-pass control (a plain baseline).

## 7. Next steps

**Done (2026-11-23):**
1. **Limited run**, inspected and logged in the deviations log
   ([docs/FOVEATED_ANALYSIS_PLAN.md](../docs/FOVEATED_ANALYSIS_PLAN.md), entry 1).
2. **Exploratory diagnostics**: cross pairs split into rotation / translation, per-view and group
   scales, the G1b ceiling, `diag_up252`, the 336 / 378 px context sweep, and the two-pass control
   (code in `scripts/foveated_eval.py`, `metrics.py`, `analysis.py`; tests in
   `tests/test_foveated_stats.py`).
3. **Full confirmatory run**: parity, probe and Sec. 6 (pose) on both models; large-view protocol
   on DA3-Small at 100–1000 views with the two-pass control (§4).

**Decision:** the pre-registered decision is **no-go** on both models. The training-free one-pass
claim is closed as a **documented negative result**. Following report §5, a runner-up enhancement
is promoted rather than the idea forced.

**Remaining housekeeping:**
4. Deviations log entry 2:
   - HiRoom scenes with 1 probe subset;
   - G2 and the DA3-Base large-view protocol not run;
   - the G3 blind spot for cross pairs, noted after the results.
5. Move the result records from `configs/results/foveated/results/` to
   `experiments/results/foveated/` (and `experiments/results/baseline/`) and register the run in
   `experiments/`.

**Optional, only if the idea is revisited:**
- **Group-scale correction, training-free.** Rescale the context group's depth and camera
  translations by (r_f / r_c)^γ, or by a scale estimated where the two groups overlap. This first
  needs the predicted cameras saved in the probe records, to check that it repairs the poses.
  Even if it does, its ceiling is the two-pass control or the focus views alone, which already give
  ≤ 5% over the focus views alone.
- **Training fix.** A LoRA on the cross-view blocks, self-distilled from the uniform-504 pass on
  mixed-grid batches. It needs the LoRA setup of enhancement #3.
- **Region zoom (G2)**, for completeness only.

## 8. Reproduce

Notebook `06_foveated` on Kaggle (GPU); see [docs/ENHANCEMENTS_RUNBOOK.md](../docs/ENHANCEMENTS_RUNBOOK.md) §3.
The full run used `MAX_SCENES = 0`, `SCALING_VIEWS = [100, 250, 500, 1000]`, `SEC6_MODES = ["pose"]`,
`PROBE_DIAGNOSTICS = True`, `SCALING_TWO_PASS = True`, `RUN["zoom"] = False`, and the large-view protocol on
DA3-Small. The main commands it runs per model:

```
python scripts/foveated_eval.py parity --weights official:da3-small --images <frames> --out <out>/parity_da3-small.json
python scripts/foveated_eval.py probe --official-src <official>/src --bench-root /tmp/da3bench --work-root <work> \
    --weights official:da3-small --datasets hiroom 7scenes --diagnostics --out <out>/probe_da3-small.json
python scripts/foveated_eval.py scaling --official-src <official>/src --bench-root /tmp/da3bench --work-root <work> \
    --weights official:da3-small --scenes chess office --n-views 100 250 500 1000 --balance --two-pass \
    --out <out>/scaling_da3-small.json
python scripts/eval_bench.py --official-src <official>/src --bench-root /tmp/da3bench --work-root <work> \
    --weights official:da3-small --datasets hiroom 7scenes --modes pose \
    --tag fov_da3-small_hiroom-7scenes__fov25_504-252 --enhancement foveated --variant fov25_504-252 \
    --foveated "fraction=0.25,context_res=252" --results-root <out>/results
python scripts/foveated_eval.py sec6views --official-src <official>/src --bench-root /tmp/da3bench --work-root <work> \
    --tag fov_da3-small_hiroom-7scenes__fov25_504-252 --config fov25_504-252 --datasets hiroom 7scenes \
    --out <out>/sec6views_fov_da3-small_hiroom-7scenes__fov25_504-252.json
python scripts/foveated_eval.py zoom --official-src <official>/src --bench-root /tmp/da3bench --work-root <work> \
    --weights official:da3-small --eth3d-zip /tmp/zips/eth3d.zip --fig-dir <out>/zoom_da3-small --out <out>/zoom_da3-small.json
python scripts/foveated_eval.py analyse --out-dir <out> --results-root <out>/results --models da3-small da3-base
```
