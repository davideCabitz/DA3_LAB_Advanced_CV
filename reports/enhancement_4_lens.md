# Enhancement 4: lens-consistent DA3, original DA3 vs DA3 + P1 on mixed-lens photo sets

**Status (2026-11-27): full run done on DA3-Small [official] (50 paired scenes, 4 protocols). DA3-Base and the photo
demo were not run.**

**Next (implemented, waiting for the Kaggle run):** plan v2 (`configs/lens_analysis_v2.yaml`, docs/LENS.md §13).
- Part A confirms padded P1 on three protocols it has never seen.
- Part B tests D1, "nested detail views" (`nest_oracle`): zoomed photos add detail instead of noise.
- The v1 results below are final.

- **The problem is real and large.** When 30% of the views are zoomed 2×, released DA3 puts them at about half their
  depth (scale ratio 1.89). Pose AUC@30 drops from 79.8 to 48.9.
- **P1 with the true zoom** recovers about half of that loss: +16.0 AUC@30, depth error halved, 22% fewer tokens.
- **The pre-registered deployable P1** (zoom estimated by DA3 itself) does nothing measurable. DA3's camera head
  barely sees zoom.
- Gates: L1 PASS, L2 FAIL, L3 PASS but negligible, L4 FAIL.
- Method, protocol and gates: [docs/LENS.md](../docs/LENS.md) (§12 results, §13 next step); `configs/lens_analysis.yaml`
  fixed before any run. Deviations: LENS.md §10.

## 1. Idea in one paragraph

Enhancement #2 measured that DA3 judges distance partly from how large things look in patch units. In a photo set taken
with different lenses (phone telephoto or ultrawide, zoom on details), that cue differs between views. So DA3 may put
zoomed views at the wrong depth scale and move their cameras along the viewing direction. P1 gives every view the same
angular resolution (pixels per degree) by processing it at its own resolution, in one pass, with native aspect ratios.
The focal lengths come from DA3's own first pass or from EXIF. On a single-lens set it does nothing (bit-identical).

## 2. What was implemented

| Part | Code | What it does |
|---|---|---|
| Geometry | `src/da3/enhancements/lens/geometry.py` | crop / resize / pad as exact affine maps; intrinsics and depth between frames; zoomed photos with exactly centred crops |
| Plan | `plan.py` | arms; which views are zoomed (view 0 never); r_i = 504 · κ_ref / κ_i with a dead band; EXIF focal length |
| Inference | `inference.py` | released preprocessing in memory (bit-identical to the benchmark adapter); canonical and padded views; one forward pass over any view sizes (mixed-resolution routing of #2) |
| Adapter | `adapter.py` | `LensAPI`: mixed-lens protocol inside the released harness, exact mapping back, `lens_meta.npz` |
| Metrics | `metrics.py` | depth-scale ratio zoomed / normal; pose AUC on view subsets (rotation and translation apart); focal ratio; zoom error; GT-free reprojection consistency |
| Experiments | `scripts/eval_bench.py --lens`, `scripts/lens_eval.py` | Sec. 6 runs per arm; parity (gate L0); per-view analysis; gates, tables, figures, records |
| Photo demo | `scripts/lens_demo.py`, `scripts/colmap_reference.py` | released vs orientation-only vs P1 on one photo set; COLMAP pseudo ground truth |
| Records | `src/da3/compare/schema.py` | protocol field `input`: mixed-lens records are compared only with each other |

## 3. Experimental setup

- **Weights:** DA3-Small, released (Apache-2.0); camera-head poses (API default); reference view "first".
- **Data and protocol:**
  - released harness (commit 3d835ec): HiRoom (30 scenes), 7Scenes (7), DTU-64 (13, pose only); ≤ 100 frames (seed
    42); evo + RANSAC; TSDF F1;
  - primary mixed-lens protocol z2_f30: 30% of the views zoomed 2× (centred crop resized back; never view 0);
  - secondary protocols, pose only: z1p5_f30, z3_f30, z2_f70.
- **Arms:**
  - `unzoomed`: the ceiling (no zoom);
  - `native`: DA3 as released on the zoomed set (the reference row);
  - `canon_oracle`: P1 with the true zoom;
  - `canonpad_oracle`: diagnostic; equal angular resolution, padded back to the full canvas;
  - `canon_self`: the deployable P1, zoom estimated from DA3's own first pass.
- **Gates** (LENS.md §6):
  - L0: parity;
  - L1: the depth-scale offset exists;
  - L2: P1 removes it (within ±5%);
  - **L3, the primary endpoint:** canon_self − native AUC@30 > 0;
  - L4: zoom recovered within 10%.
- **Run setup:**
  - F1 only for `unzoomed`, `native` and `canon_self` (F1 is descriptive; RAM and time; LENS.md §10);
  - 2 parallel TSDF fusions (4 ran out of RAM).

## 4. Results [official], DA3-Small

### 4.0 Checks (gate L0)
| Check | DA3-Small |
|---|---|
| unit tests (`tests/test_lens.py`, `test_compare.py`, `test_foveated.py`) | 61 passed |
| no zoomed view: unzoomed / native / canon_oracle / canonpad_oracle bit-identical | PASS (max abs diff 0 in depth, conf, extrinsics, intrinsics) |
| no zoomed view: canon_self | PASS, bit-identical; 1 pass; z_est in [0.977, 1.015] |
| zoom sanity (export border 0, principal point centred, smaller grid) | PASS (native, canon_oracle) |

### 4.1 Primary protocol z2_f30 (50 paired scenes; in brackets: paired difference to native)
| Arm | AUC@3 | AUC@30 | AUC@30 zoom–normal (rot / trans only) | F1 w/o p. | depth-scale ratio zoom / normal | focal ratio (zoomed) | zoom error | scene AbsRel | tokens | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| unzoomed (ceiling) | 15.9 (+9.2) | 79.8 (+30.9) | 79.7 (92.5 / 81.0) | 22.2 (+7.7) | 0.999 [0.993, 1.005] | 1.079 | – | 0.062 | 38109 | 2.94 |
| native (DA3 as released) | 6.7 | 48.9 | 18.3 (80.4 / 19.3) | 14.5 | **1.893** [1.866, 1.918] | 0.583 | – | 0.191 | 38109 | 2.95 |
| canon_oracle (P1, true zoom) | 7.8 (+1.1) | **64.8 (+16.0)** | 50.4 (86.0 / 51.8) | – | 1.168 [1.156, 1.180] | 0.702 | – | 0.090 | 29579 | 2.07 |
| canonpad_oracle (diagnostic) | 7.9 (+1.3) | **68.1 (+19.3)** | 61.9 (81.8 / 64.1) | – | **1.043** [1.015, 1.084] | 1.105 | – | 0.078 | 38109 | 3.14 |
| canon_self (P1, zoom from DA3) | 6.7 (+0.1) | 49.1 (+0.2) | 18.8 (80.6 / 19.8) | 14.6 (+0.0) | 1.893 [1.866, 1.918] | 0.585 | 0.471 [0.451, 0.484] | 0.191 | 37564 | 4.66 |

Paired differences vs native, 95% CIs (points):

| Arm | Δ AUC@3 | Δ AUC@30 | Δ AUC@30 zoom–normal | Δ F1 |
|---|---|---|---|---|
| unzoomed | 9.19 [7.27, 11.17] | 30.92 [28.76, 32.99] | 61.39 [57.49, 65.06] | 7.70 [4.32, 11.34] |
| canon_oracle | 1.14 [0.63, 1.69] | 15.95 [14.74, 17.20] | 32.11 [29.36, 34.84] | – |
| canonpad_oracle | 1.29 [0.37, 2.30] | 19.27 [16.42, 21.80] | 43.54 [37.88, 48.86] | – |
| canon_self | 0.05 [0.02, 0.09] | 0.21 [0.10, 0.33] | 0.52 [0.27, 0.80] | 0.04 [0.00, 0.09] |

AUC@30 per dataset:

| Arm | HiRoom | 7Scenes | DTU-64 |
|---|---|---|---|
| unzoomed | 75.2 | 78.2 | 91.2 |
| native | 44.5 | 51.5 | 57.5 |
| canon_oracle | 59.8 | 66.7 | 75.4 |
| canonpad_oracle | 61.1 | 67.1 | 84.8 |
| canon_self | 44.5 | 51.5 | 58.3 |

### 4.2 Gates
| Gate | DA3-Small |
|---|---|
| L1 offset exists | **PASS**: 1.893 [1.866, 1.918] |
| L2 P1 removes it (±5%) | **FAIL**: 1.168 [1.156, 1.180] |
| **L3 primary: Δ AUC@30 canon_self − native** | **PASS**: +0.21 [0.10, 0.33]. Statistically above 0, practically negligible |
| L4 zoom from DA3 within 10% | **FAIL**: 0.471 [0.451, 0.484] |
| hypothesis readout (H_ang / H_grid) | "mixed / unclear" by the rule: the padded CI crosses 1.05 |

### 4.3 Secondary protocols (descriptive, Bonferroni 98.3% CIs, pose only)
| Protocol | Arm | AUC@30 | AUC@30 zoom–normal | depth-scale ratio | focal ratio (zoomed) | zoom error | tokens |
|---|---|---|---|---|---|---|---|
| z1p5_f30 | native | 55.4 | 29.4 | 1.453 | 0.758 | – | 38109 |
| | canon_oracle | 72.1 (+16.7 [15.3, 18.1]) | 63.9 | 1.093 | 0.879 | – | 31733 |
| | canon_self | 55.4 (+0.02) | 29.5 | 1.453 | 0.758 | 0.318 | 38067 |
| z3_f30 | native | 44.9 | 12.2 | 2.744 | 0.397 | – | 38109 |
| | canon_oracle | 56.6 (+11.7 [9.9, 13.5]) | 36.0 | 1.261 | 0.458 | – | 27908 |
| | canon_self | 45.3 (+0.37) | 13.0 | 2.743 | 0.407 | 0.643 | 36737 |
| z2_f70 | native | 36.5 | 17.7 | 1.893 | 0.601 | – | 38109 |
| | canon_oracle | 53.9 (+17.4 [14.0, 20.1]) | 41.0 | 1.239 | 0.703 | – | 72615 |
| | canon_self | 36.4 (−0.02) | 17.7 | 1.893 | 0.601 | (0.014, not meaningful: zoomed views are the reference) | 38117 |

### 4.4 Photo demo
Not run yet. The statue set taken so far (34 photos, `Statua_No_Zoom`) uses one lens (24 mm), so P1 would be a no-op on
it. A zoomed set is needed: the same walk plus 2–3× shots of the statue and details, and some 0.5× shots, as JPG with
EXIF. The released-DA3 reconstruction of the no-zoom set is discussed in §6.

## 5. Discussion

1. **DA3 reads zoom as moving closer, inside one multi-view reconstruction.**
   - The zoomed / normal depth-scale ratio follows zoom^0.92 at all three zooms (1.45, 1.89 and 2.74 against 1.45, 1.89
     and 2.75 predicted).
   - The damage is in camera *positions*: on zoomed–normal pairs, translation-only AUC@30 falls from 81.0 to 19.3, and
     rotation-only from 92.5 to 80.4. DA3 slides each zoomed camera along its viewing axis.
   - For monocular depth models the same effect was published two weeks before our run (arXiv 2611.17903; see §6).
     What our numbers add is its effect on a multi-view reconstruction: poses, cross-view scale and fusion.
2. **The camera head barely notices zoom.** The zoomed focal ratio over the general bias (0.583 / 1.079 at 2×) means
   DA3 sees a 2× zoom as about 1.08×. That is why the self-estimating P1 (`canon_self`) almost never re-processes a
   view (L4 FAIL), and why L3, although formally a PASS, is negligible.
3. **Equal angular resolution works when the zoom is known.**
   - It recovers 52% of the pose accuracy lost to zoom, consistently on HiRoom, 7Scenes and DTU (+15 to +18 AUC@30) and
     at every zoom.
   - It halves the scene-level depth error and costs fewer tokens.
   - On phones the zoom comes from EXIF, so this is the practical form of P1. This reading came after the results: the
     pre-registered deployable arm was `canon_self`.
4. **Which explanation holds?** The angular size per patch (H_ang) explains most of the offset.
   - The padded arm (equal angular resolution on the full canvas) brings the scale to 1.043 and the focal length back to
     the unzoomed value (1.105 vs 1.079), with the best pose accuracy (+19.3).
   - The leftover offset of the canonical arm grows about as zoom^0.22 and comes from the smaller grid.
   - Interpretation; the pre-registered readout is "unclear".
5. **What P1 cannot do:** AUC@3 barely moves (+1.1), because processing zoomed views smaller discards their fine
   detail. Using that detail is the next step (LENS.md §13).

## 6. Limitations and honest claims

- **Novelty (LENS.md §7):**
  - The phenomenon "zoom read as moving closer, camera estimate blind to zoom" is published for *monocular* depth
    models: "Zoom In, Scale Out", arXiv 2611.17903, 13 Nov 2026, including DA3-Nested. Cite it.
  - Ours to claim: the multi-view effect (pose collapse, cross-view scale inconsistency), the mixed-lens Sec. 6
    protocol, training-free equal-angular-resolution inference in one multi-view pass, and the padded diagnostic.
  - Canonical-camera resampling: cite Metric3D. Known-intrinsics inputs: cite Pow3R / MapAnything.
- **Synthetic zoom** (crop and upsample) on the benchmark; real extra detail only on DTU (1600 × 1200).
- **After-the-fact readings:** the padded variant as the best P1, and "P1 with EXIF zoom" as the deployable form. A
  version-2 pre-registered plan should confirm them.
- **Not computed:** F1 for the true-zoom arms; DA3-Base; ETH3D / ScanNet++ (compute budget).
- **Released-DA3 photo reconstruction (statue, no zoom, 34 photos around the fountain): a mess.**
  - Not caused by P1 (not involved).
  - DA3 collapses the first half of the 360° walk. Photos 0–16 are placed almost on top of each other, and the camera
    head and the ray map disagree by up to 149°. Photos 17–33 form a clean arc.
  - DA3 also gives the cloudy sky a finite depth close to the fountain.
  - Likely cause: the fountain looks nearly the same from every side. This is a separate DA3 limitation (360° walks
    around a near-symmetric object), not part of this enhancement.

## 7. Reproduce

Notebook `10_lens` on Kaggle (GPU T4 ×2, Internet on); see [docs/ENHANCEMENTS_RUNBOOK.md](../docs/ENHANCEMENTS_RUNBOOK.md)
§3c. The commands per model and arm:

```
python scripts/eval_bench.py --official-src <official>/src --bench-root /tmp/da3bench --work-root /tmp/lens_work \
    --weights official:da3-small --datasets hiroom 7scenes dtu64 --modes pose recon_unposed \
    --tag lens_da3-small_hiroom-7scenes-dtu64__z2_f30__canon_self --enhancement lens --variant canon_self \
    --baseline-tag lens_da3-small_hiroom-7scenes-dtu64__z2_f30__native --lens "arm=canon_self,zoom=2,fraction=0.3" \
    --results-root <out>/results --fusion-workers 2
python scripts/lens_eval.py views --official-src <official>/src --bench-root /tmp/da3bench --work-root /tmp/lens_work \
    --tag <tag> --datasets hiroom 7scenes dtu64 --weights official:da3-small --out <out>/views_<tag>.json
python scripts/lens_eval.py analyse --out-dir <out> --results-root <out>/results
python scripts/lens_demo.py --images <photos> --out <out>/demo/da3-base --weights official:da3-base \
    --colmap <out>/demo/colmap/colmap_cameras.json
```

Result records of this run: `lens_results.zip` → `lens/results/{baseline,lens}/` (to be copied into
`experiments/results/` and committed).
