# Enhancement 2: Foveated DA3, mixed per-view resolutions in one pass

**Status (2026-11-22): implemented, not yet run.**
- Method, setup and claims below are final.
- The result sections (§4) are filled from the Kaggle run of `notebooks/06_foveated`, setting
  **[official]** (released DA3-Small / DA3-Base weights run by us).
- Until then every number in this file is a placeholder (–).

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

## 4. Results [official]

_Pending the Kaggle run. Fill from `foveated/summary.md` and `foveated/gates.json`; the records are
in `experiments/results/` (`baseline/` for the uniform rows, `foveated/` for the rest)._

### 4.1 Routing check
| Check | DA3-Small | DA3-Base |
|---|---|---|
| equal resolution, fp32 / fp16 / reference selection / posed: bit-identical | – | – |
| mixed resolution, view-order equivariance | – | – |

### 4.2 Probe (16 views per scene, 8 focus)
| Model | Configuration | AbsRel focus | AbsRel context | AUC@30 focus pairs | AUC@30 context pairs | tokens |
|---|---|---|---|---|---|---|
| DA3-Small | all 504 | – | – | – | – | – |
| DA3-Small | all 252 | – | – | – | – | – |
| DA3-Small | 504 / 252 | – | – | – | – | – |
| DA3-Small | 504 / 280 | – | – | – | – | – |
| DA3-Small | 504 / 252 balanced | – | – | – | – | – |

### 4.3 Large-view protocol (DA3-Small, T4)
| Views | Configuration | time (s) | peak memory (GB) | speed-up | AUC@30 focus pairs | AbsRel focus |
|---|---|---|---|---|---|---|
| 1000 | all 504 | – | – | 1× | – | – |
| 1000 | foveated 10% | – | – | – | – | – |
| 1000 | uniform, equal tokens | – | – | – | – | – |
| 1000 | fewer 504-px views, equal tokens | – | – | – | – | – |

### 4.4 Sec. 6 protocol
| Model | Configuration | HiRoom AUC@3 / AUC@30 / F1 | 7Scenes AUC@3 / AUC@30 / F1 | Δ vs uniform at equal tokens |
|---|---|---|---|---|
| DA3-Small | all 504 (original) | – | – | – |
| DA3-Small | foveated 25% | – | – | – |

### 4.5 Region zoom (ETH3D)
| Model | Method | region AbsRel | F1@1% | R_v error (°) |
|---|---|---|---|---|
| DA3-Small | global 504 / 896 | – | – | – |
| DA3-Small | naive crop | – | – | – |
| DA3-Small | zoom view through E_c (DA3 cameras) | – | – | – |

### 4.6 Gates (decision, estimate [95% CI], margins at which the decision changes)
| Gate | DA3-Small | DA3-Base |
|---|---|---|
| A/A noise floor → margin used (depth / pose) | – | – |
| G0 routing parity | – | – |
| G1a headroom (u252 / u504 AbsRel) | – | – |
| G1b context informative (with / without context) | – | – |
| G3 C1 focus AbsRel (504/252 / u504) | – | – |
| G3 C2 context AUC@30 (504/252 / u252) | – | – |
| G3 secondary: 504/280, balanced (97.5% CIs) | – | – |
| G3 robustness: 3 random focus choices | – | – |
| G3 rescue (only if G3 does not pass) | – | – |
| G2 C1 region AbsRel (oracle / random / texture) | – | – |
| G2 C2 median R_v error | – | – |

## 5. Discussion

_After the run: whether low-resolution context views help or hurt the focus views (and vice versa),
whether balancing or 280-px context views close the gap, how foveation compares with uniform
downscaling at the same token budget, how far the view count goes on one T4, and whether DA3
follows the virtual zoom camera._

## 6. Limitations and honest claims

- **Cross-scale token matching was never trained.** DA3 saw one resolution per batch with the
  short side ≥ 280 px; a 252-px context view is outside that range. Gate 3 measures the effect;
  280-px context views and attention-mass balancing are the training-free mitigations.
- **Uniform downscaling may tie** at the same token budget. That is a reportable finding: the
  claim is detail where it is needed (the focus views), not raw speed.
- **The efficiency bar is Spark3R** (training-free token reduction on DA3, ~17× at 1000 frames,
  accuracy unchanged). It is not re-run here; a comparison at equal FLOPs and stacking the two
  are future work.
- **Sec. 6 is not token-bound at ≤ 100 views** on a T4, so the paper's tables do not show the
  efficiency gain; the large-view protocol and the region protocol are new and never mixed into the
  Sec. 6 tables.
- **Region zoom:** zoom views have a narrow field of view and no baseline of their own, so their
  detail is mostly monocular; regions are chosen with GT (oracle); ETH3D photos are distorted
  (thin-prism fisheye) and treated as pinhole, like the released harness does.
- **7Scenes GT depth** is Kinect depth not registered to the RGB camera; compare methods with each
  other, not with HiRoom.
- **Small number of scenes in places.** The large-view protocol has 7 scenes and the zoom 11, so
  their CIs are wide and INCONCLUSIVE is a likely, honest outcome. The view subsets of a scene are
  not independent samples and are not counted as such.
- **May claim** (not found in the literature, report §3.4):
  - native mixed per-view resolutions in one training-free pass of a feed-forward multi-view model,
    bit-identical to the original at equal resolution;
  - attention-mass balancing of the cross-view softmax as the training-free correction for unequal
    token counts;
  - virtual zoom cameras through E_c as the region-level use (weak on its own).
- **Must cite, not claim:** token merging and pruning (Spark3R, PrePARE, TurboVGGT); trained
  high-resolution branches (HD-VGGT); monocular tiling (PatchFusion, BoostingDepth); perspective
  crops (Perspective Crop Layers).

## 7. Reproduce

Notebook `06_foveated` on Kaggle (GPU); see [docs/ENHANCEMENTS_RUNBOOK.md](../docs/ENHANCEMENTS_RUNBOOK.md) §3.
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
