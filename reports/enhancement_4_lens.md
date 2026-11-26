# Enhancement 4: lens-consistent DA3, original DA3 vs DA3 + P1 on mixed-lens photo sets

**Status (2026-11-26): implemented, waiting for the Kaggle run.**
- Method, protocol and the pre-registered gates are final: [docs/LENS.md](../docs/LENS.md),
  `configs/lens_analysis.yaml` (fixed before any run).
- Sections §4–§6 hold placeholders (–) until notebook `10_lens` has run: first `MAX_SCENES = 2` (smoke test, not
  reportable), then `MAX_SCENES = 0` (DA3-Small, then DA3-Base if time allows). Setting **[official]**: released weights,
  network unchanged.

## 1. Idea in one paragraph

Enhancement #2 measured that DA3 judges distance partly from how large things look in patch units. In a photo set taken
with different lenses (phone telephoto / ultrawide, zoom on details) that cue differs between views, so DA3 may put
zoomed views at the wrong depth scale and move their cameras along the viewing direction. P1 gives every view the same
angular resolution (pixels per degree) by processing it at its own resolution, in one pass, with native aspect ratios;
the focal lengths come from DA3's own first pass or from EXIF. On a single-lens set it does nothing (bit-identical).

## 2. What was implemented

| Part | Code | What it does |
|---|---|---|
| Geometry | `src/da3/enhancements/lens/geometry.py` | crop / resize / pad as exact affine maps; intrinsics and depth between frames; zoomed photos with exactly centred crops |
| Plan | `plan.py` | arms, which views are zoomed (view 0 never), r_i = 504 · κ_ref / κ_i with a dead band, EXIF focal length |
| Inference | `inference.py` | released preprocessing in memory (bit-identical to the benchmark adapter), canonical and padded views, one forward pass over any view sizes (mixed-resolution routing of #2) |
| Adapter | `adapter.py` | `LensAPI`: mixed-lens protocol inside the released harness, exact mapping back, `lens_meta.npz` |
| Metrics | `metrics.py` | depth-scale ratio zoomed / normal, pose AUC on view subsets (rotation / translation apart), focal ratio, zoom error; GT-free reprojection consistency |
| Experiments | `scripts/eval_bench.py --lens`, `scripts/lens_eval.py` | Sec. 6 runs per arm; parity (gate L0), per-view analysis, gates / tables / figures / records |
| Photo demo | `scripts/lens_demo.py`, `scripts/colmap_reference.py` | released vs orientation-only vs P1 on one photo set; COLMAP pseudo ground truth |
| Records | `src/da3/compare/schema.py` | protocol field `input`: mixed-lens records are compared only with each other |

## 3. Experimental setup

- **Weights:** DA3-Small (and DA3-Base), released (Apache-2.0); camera-head poses (API default); reference view "first".
- **Data and protocol:** released harness (commit 3d835ec), HiRoom, 7Scenes, DTU-64 (pose only), ≤ 100 frames (seed
  42), evo + RANSAC, TSDF F1. Primary mixed-lens protocol: 30% of the views zoomed 2× (z2_f30). Secondary: z1p5_f30,
  z3_f30, z2_f70 (pose only).
- **Arms:** `unzoomed` (ceiling), `native` (DA3 as released, reference row), `canon_oracle` (P1, true zoom),
  `canonpad_oracle` (diagnostic), `canon_self` (P1 deployable).
- **Gates** (docs/LENS.md §6): L0 parity, L1 the offset exists, L2 P1 removes it, **L3 primary endpoint** (canon_self −
  native AUC@30 > 0), L4 zoom recovered within 10%.

## 4. Results [official]

### 4.0 Checks (gate L0)
| Check | DA3-Small | DA3-Base |
|---|---|---|
| unit tests (`tests/test_lens.py`) | – | – |
| no zoomed view: unzoomed / native / canon_oracle / canonpad_oracle bit-identical | – | – |
| no zoomed view: canon_self (passes; z_est range) | – | – |
| zoom sanity (export border 0, principal point centred, smaller grid) | – | – |

### 4.1 Primary protocol z2_f30 (paired scenes; in brackets: difference to native)
| Arm | AUC@3 | AUC@30 | AUC@30 zoom–normal (rot / trans) | F1 w/o p. | depth-scale ratio zoom / normal | focal ratio (zoomed) | zoom error |
|---|---|---|---|---|---|---|---|
| unzoomed (ceiling) | – | – | – | – | – | – | – |
| native (DA3 as released) | – | – | – | – | – | – | – |
| canon_oracle (P1, true zoom) | – | – | – | – | – | – | – |
| canonpad_oracle (diagnostic) | – | – | – | – | – | – | – |
| canon_self (P1) | – | – | – | – | – | – | – |

### 4.2 Gates
| Gate | DA3-Small | DA3-Base |
|---|---|---|
| L1 offset exists | – | – |
| L2 P1 removes it | – | – |
| **L3 primary: Δ AUC@30 canon_self − native** | – | – |
| L4 zoom from DA3 within 10% | – | – |
| hypothesis readout (H_ang / H_grid) | – | – |

### 4.3 Secondary protocols (descriptive)
_Fill from `lens/summary.md`._

### 4.4 Photo demo (Trento building)
| Arm | reprojection disagreement (median %) | per lens | AUC@3 / AUC@30 vs COLMAP | camera-centre error per lens | focal ratio per lens |
|---|---|---|---|---|---|
| released | – | – | – | – | – |
| orient | – | – | – | – | – |
| lens (P1) | – | – | – | – | – |

Figures: `lens/demo/<model>/compare/{depth_compare,ray_compare,cameras_topdown}.png`; Blender packages
`lens/demo/<model>/{released,orient,lens}.zip`.

## 5. Discussion

_After the run: which hypothesis the arms support, how much of the native → ceiling gap P1 closes, whether DA3's own focal
estimate is good enough (L4), and what the photo demo shows on real lenses._

## 6. Limitations and honest claims

See docs/LENS.md §7 (novelty: may claim / must cite) and §9 (limitations). In short: synthetic zoom on the benchmark,
canonicalisation discards the zoomed views' extra pixels, COLMAP is only a pseudo ground truth, ETH3D / ScanNet++ not
evaluated (compute budget).

## 7. Reproduce

Notebook `10_lens` on Kaggle (GPU T4 ×2, Internet on); see [docs/ENHANCEMENTS_RUNBOOK.md](../docs/ENHANCEMENTS_RUNBOOK.md)
§3c. The commands per model and arm:

```
python scripts/eval_bench.py --official-src <official>/src --bench-root /tmp/da3bench --work-root /tmp/lens_work \
    --weights official:da3-small --datasets hiroom 7scenes dtu64 --modes pose recon_unposed \
    --tag lens_da3-small_hiroom-7scenes-dtu64__z2_f30__canon_self --enhancement lens --variant canon_self \
    --baseline-tag lens_da3-small_hiroom-7scenes-dtu64__z2_f30__native --lens "arm=canon_self,zoom=2,fraction=0.3" \
    --results-root <out>/results
python scripts/lens_eval.py views --official-src <official>/src --bench-root /tmp/da3bench --work-root /tmp/lens_work \
    --tag <tag> --datasets hiroom 7scenes dtu64 --weights official:da3-small --out <out>/views_<tag>.json
python scripts/lens_eval.py analyse --out-dir <out> --results-root <out>/results
python scripts/lens_demo.py --images <photos> --out <out>/demo/da3-base --weights official:da3-base \
    --colmap <out>/demo/colmap/colmap_cameras.json
```
