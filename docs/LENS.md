# Enhancement #4: lens-consistent DA3 (photo sets with mixed zoom, lenses and orientations)

**Status (2026-11-26): implemented, waiting for the Kaggle run** (notebook `10_lens`). Training-free, on the released
weights (setting **[official]**). Results go to [reports/enhancement_4_lens.md](../reports/enhancement_4_lens.md);
how to run: [ENHANCEMENTS_RUNBOOK.md](ENHANCEMENTS_RUNBOOK.md) §3c. Statistics pre-registered in
`configs/lens_analysis.yaml` (§6), fixed before any run.

---

## 1. How we got here

Our first three enhancements were chosen by a multi-agent study ([da3_enhancement_report.md](da3_enhancement_report.md))
whose main criterion was **novelty of mechanism**: each one exploited something only DA3 has (two pose decoders, the
camera encoder E_c, grid-free cross-view attention). The gains were estimates. None of the three started from a
*measured* weakness of DA3, and all three found little or no headroom behind the mechanism:

| | What it did | Outcome | Why it failed |
|---|---|---|---|
| #1 Self-Audit | DA3's redundant camera estimates (head vs ray map, E_c re-pass) as reliability signals, per-view pose choice, GT-free calibration | 2-scene smoke test: gates G2, G3, G4 fail; full run pending | head and ray are decoded from the same tokens (correlated errors, oracle headroom +2-5 AUC@3); almost no gross failures to detect; triple collocation assumes independent view errors, DA3 predicts views jointly |
| #2 Foveated | mixed per-view resolutions in one pass (focus 504 px, context 252 px) | full run, **no-go** on both models | DA3's depth scale follows the token grid: two inconsistently scaled reconstructions; small ceiling anyway |
| #3 LoRA (E3) | the pose-conditioned DA3 as teacher of the unposed one | **no-go** at the gate (reproduced exactly) | conditioning on true poses changes the cameras (2x better) but not the depth (ratio 0.995) |

What the failures *measured* (all [official], released weights):
- **F1, scale from the patch footprint** (#2, [report §4.7](../reports/enhancement_2_foveated.md)): in one pass, a view on a
  coarser token grid gets a different depth scale, ratio ≈ (resolution ratio)^0.7 (0.62 at 252 vs 504 px), stable over
  37 scenes. Rotations stay correct, translations collapse. Feeding the same blurred image on the full grid removes the
  effect: the cause is the grid, not the missing detail.
- **F2, poses go into the cameras, not the depth** (#3).
- **F3, depth is mostly per view**: 8 more full-resolution views improve per-view depth by 3-5% (#2, G1b ceiling).
- **F4, DA3 depends on the reference view** (A/A control of #2): 3-4% in depth, ~2 points AUC@3.

**New rule for this enhancement:** start from a measured limitation, prove the problem exists with a cheap gate before
claiming anything, and be a no-op (bit-identical) where the problem does not arise. F1 is the strongest of the four
findings, and it points at a real-world situation the paper never tests.

## 2. The problem in DA3

The paper's promise is geometry from **any views** (Sec. 1, 3.1): any number of photos, with or without poses. Real photo
sets of one scene, however, are rarely taken with one lens:
- phones switch between 0.5× (ultrawide), 1×, 2-3× (telephoto) lenses, and people zoom on details;
- crowd-sourced photo collections (landmarks) mix cameras and focal lengths;
- portrait and landscape photos are mixed.

What DA3 does with them:
- **Training never mixes angular resolutions within a scene.** Sec. 3.4: one resolution per batch (504² and a few aspect
  ratios); the datasets of Tab. 1 are mostly captured with one camera per scene. Nothing forces the depth scale of a view
  to be independent of how many pixels per degree it has.
- **The released preprocessing** brings every photo to 504 px on its longest side and then centre-crops all of them to
  one common size (`io_utils.load_views`, as in the official API). Mixed orientations lose their top or sides; a zoomed
  photo is processed on the same grid as the others, so its *angular resolution* (pixels per degree) is z times finer.
- **F1 says the depth scale follows the size of things in patch units.** That is the right cue when every view has the
  same angular resolution (bigger in the image = closer). It is the wrong cue across lenses: a 2× zoomed photo of a
  facade looks like a photo taken from half the distance.

**Prediction:** in a mixed-lens set the zoomed views get a different depth scale from the others (by ~z^0.7 if the
foveated exponent carries over), their cameras slide along the viewing direction, and the fused reconstruction splits
into layers. This is a concrete failure of the "any views" claim, and no released setting avoids it.

## 3. Hypothesis, and how the protocol tests it

F1 has two possible explanations, which the foveated experiment could not separate (there, both changed together):
- **H_ang:** the scale cue is the angular size of structures per patch (patch footprint in degrees).
- **H_grid:** the scale cue is the token grid itself (number of patches per view).

Zooming separates them, because a zoom changes the angular resolution without changing the grid:

| Arm (zoomed views) | Grid | Angular resolution | H_ang predicts | H_grid predicts |
|---|---|---|---|---|
| `native` (DA3 as released) | same as the others | z× finer | **offset** (ratio ≈ z^γ) | no offset |
| `canon_oracle` (P1) | smaller (1/z) | equal | no offset | **offset** (as in #2) |
| `canonpad_oracle` (diagnostic) | same (padded) | equal | no offset | no offset |

- H_ang is supported if `native` shows the offset and `canon_oracle` removes it: P1 is the fix.
- H_grid is supported if `native` is consistent and `canon_oracle` breaks: then DA3 already handles zoom on one grid,
  there is nothing to fix, and that is the (reported) result.

## 4. Method (P1): equal angular resolution in one pass

1. **Focal length of every photo.** Three sources, all supported:
   - *oracle* (benchmark only): the known zoom of the protocol;
   - *DA3 itself*: a first pass with the released preprocessing; the camera head's focal length of every view is mapped
     back to photo pixels (deployable, no metadata);
   - *EXIF*: `FocalLengthIn35mmFilm`, which phones write for every lens: f_px = f35 / 43.27 mm × image diagonal
     (photo demo, when every photo has it).
2. **Plan.** With κ_i = f_i / L_i (focal length over the photo's longest side), view i is processed at longest side
   r_i = 504 · κ_ref / κ_i, with κ_ref the lower median (the dominant lens; always an actual view). Then every view has the
   same angular resolution as the reference view at 504 px. Zoomed views get a smaller grid, wide views a larger one
   (clipped to [112, 1008] px). Views within ±15% of κ_ref keep 504 px exactly (dead band, `canon_self`).
3. **One forward pass over views of different sizes** with the mixed-resolution routing of enhancement #2
   (`foveated_forward`): bit-identical to `DA3.forward` at equal resolution (checked on GPU in #2), cross-view
   attention well defined over views with different token counts (constant RoPE position in the cross-view blocks).
   Every photo keeps its **native aspect ratio** (no centre-crop to a common size).
4. **No-op guarantee.** If every view falls in the dead band (a single-lens set), the first pass is the answer:
   bit-identical to the released DA3. So the standard Sec. 6 numbers cannot get worse by construction.

Cost: one extra pass when (and only when) lenses differ; the second pass usually has *fewer* tokens (zoomed views shrink).

## 5. Mixed-lens Sec. 6 protocol (new; compared only with itself)

Built on the released harness (pinned commit 3d835ec) so that everything except the input is the paper's protocol:
same datasets (HiRoom, 7Scenes, DTU-64), same frame sampling (≤ 100 frames, seed 42), same evo + RANSAC alignment, same
TSDF fusion and F1 thresholds, same GT.

- **Zoom inside the adapter** (`LensAPI`): a share `fraction` of the views (evenly spaced, never view 0, the reference
  of the paper protocol) is replaced by a zoomed photo: a centred crop of 1/z of each side resized back to the full size
  (focal length × z, principal point exactly centred: even margins). Primary: z = 2, fraction 0.3.
- **Exact mapping back.** Every step (crop, resize, pad) is an axis-aligned affine map of continuous pixel coordinates.
  The prediction of each view is mapped to the frame of the dataset image the harness knows:
  - depth / confidence warped with nearest neighbours; a zoomed view covers only the central 1/z of its source image,
    the rest is 0 (invalid, skipped by the TSDF fusion);
  - intrinsics K' = T_A K (so a zoom that DA3 understood correctly gives back the source focal length);
  - extrinsics unchanged.
- **Why synthetic zoom is fair.** A crop-and-upsample is blurrier than a real telephoto photo, but #2 showed that DA3's
  scale does not react to blur on a fixed grid (`diag_up252`: 252-px detail on the 504-px grid behaves like 504 px).
  DTU-64 photos (1600 × 1200) also keep real detail at 2×.
- **Records.** Every arm writes a standard record whose protocol has `input = "lens:zoom=2,fraction=0.3"`;
  `da3.compare` only pairs records with the same input, so these numbers never appear next to the standard Sec. 6 runs.
  Within the protocol, the reference row (enhancement `baseline`, variant `head`) is DA3 as released on the zoomed set.
- **Per-view diagnostics** (`lens_eval.py views`): depth-scale ratio zoomed / normal views (per-view median alignment
  to GT depth; HiRoom, 7Scenes), pose AUC on all / normal / zoom / cross pairs (rotation and translation errors apart),
  focal ratio, zoom-estimate error.
- **Secondary protocols** (pose only, descriptive): z = 1.5 and 3 at 30%, and z = 2 at 70% (the normal views become a
  wide minority, processed at up to 1008 px).
- **Not covered:** mixed orientations on the benchmark (a portrait "photo" can only be cut out of a landscape frame,
  which also narrows the field of view); they are covered by the photo demo (§8). `portrait=` exists in the spec for
  completeness.

## 6. Pre-registered analysis (`configs/lens_analysis.yaml`, fixed 2026-11-26 before any run)

- Unit = scene; paired across arms; 95% scene-bootstrap CIs (4000 resamples, seed 0); three-way decisions
  (PASS / FAIL / INCONCLUSIVE) read off the CI, as in #2.
- **Scale band ±5%.** The A/A spread of per-view scales in #2 was 2-3%, so ±5% means "as consistent as a normal pass".
- Gates (primary protocol z2_f30 only):

| Gate | Question | Rule |
|---|---|---|
| L0 | no zoomed view ⇒ every arm bit-identical to the released adapter (GPU, released weights) | must PASS before anything else |
| L1 | does the problem exist? `native` scale ratio (geometric mean over scenes) | PASS = CI entirely outside [1/1.05, 1.05]; FAIL = CI inside (DA3 is lens-consistent: P1 stops, reported) |
| L2 | does P1's mechanism remove it? `canon_oracle` scale ratio | PASS = CI inside the band |
| **L3 (primary endpoint)** | does deployable P1 help? `canon_self` − `native`, Sec. 6 AUC@30 (points), all datasets | PASS = CI of the paired mean difference > 0 |
| L4 | is the zoom recovered from DA3 itself? median relative zoom error of the zoomed views | PASS = CI ≤ 10% |

- Descriptive (no decisions): padded arm (H_ang vs H_grid), AUC@3, cross-pair AUC@30 with rotation / translation apart,
  F1, focal ratio, the gap to the unzoomed ceiling; secondary protocols with Bonferroni CIs.

## 7. Novelty check (searched 2026-11-26; abstract / snippet level, re-check before the slides)

| Searched for | Closest work found | Verdict |
|---|---|---|
| feed-forward multi-view reconstruction with mixed focal lengths / zoom among pinhole views; depth-scale consistency across lenses | **Camera3R** (arXiv 2611.13453), **Many Eyes, One World / MEOW** (2609.35658), **Any3R** (2610.36150), **Fisheye3R** (2603.28896): mixed *camera models* (fisheye, panoramas), all **trained** (LoRA experts, new backbones). "On Geometric Understanding and Learned Priors in Feed-forward 3D Reconstruction Models" (2512.11508) reports VGGT stable across 35 / 50 / 85 mm, one lens per reconstruction as far as the snippet shows | **not found** |
| resampling to a canonical focal length | **Metric3D** (Yin et al., ICCV 2023): canonical camera transform for *single-image metric* depth | mechanism exists for monocular metric depth → **cite** |
| DA3 and zoom / lenses / per-view resolution | nothing | not found |

- **May claim:** (i) the measured finding that DA3's multi-view depth scale is not lens-invariant (if L1 passes), and the
  H_ang vs H_grid separation; (ii) training-free equal-angular-resolution inference of a feed-forward multi-view model
  *in one pass* over views of different sizes, a no-op on single-lens sets; (iii) the mixed-lens Sec. 6 protocol.
- **Must cite, not claim:** canonical-camera resampling (Metric3D); heterogeneous camera models (Camera3R, MEOW, Any3R,
  Fisheye3R); the mixed-resolution routing is our enhancement #2; token pruning / merging (Spark3R) if efficiency comes up.
- Risk signal: 2512.11508 suggests VGGT copes with different focal lengths; L1 decides for DA3 in one GPU-hour.

## 8. Photo demo (a building in Trento)

**Capture recipe** (what lets P1 show; keep the original files with EXIF: no messaging apps):
- 30-40 photos walking along / around the building, ~60-70% overlap between neighbours, one time of day, sun out of frame;
- ~24 at 1×, half portrait and half landscape;
- ~8 at 2-3× of details (portal, windows, cornice), taken near some of the 1× positions;
- ~5 at 0.5× showing the whole building from close.

**What runs** (`scripts/lens_demo.py`, notebook 10 part 4), per model (DA3-Small, DA3-Base):
- `released`: `scripts/infer.py` unchanged (504 px, centre-crop to one common size);
- `orient`: native aspect ratio at 504 px, one pass (the orientation half of P1);
- `lens`: P1, equal angular resolution (EXIF focal lengths; DA3's own when a photo has none).

Each is an `infer.py`-style folder (depth / ray maps, cameras, PLY, GLB, transforms.json, Blender import) and a zip.

**How it is compared** (no GT exists):
- **reprojection consistency** (GT-free, scale-free): median disagreement between each view's depth and its neighbours'
  depth where they see the same surface, overall and per lens;
- **vs COLMAP** (`scripts/colmap_reference.py`: SIFT, exhaustive matching, one camera per image, on EXIF-oriented copies
  of the full photos): pose AUC@3 / AUC@30, camera-centre error after Sim(3) per lens, focal ratio per lens. COLMAP is a
  *pseudo* ground truth; every number says so;
- **figures:** depth maps of the same photos side by side on one colour scale (each arm normalised by the median depth of
  its dominant-lens photos: zoomed photos that DA3 puts too close turn red), ray maps (the field of view each arm assigns),
  top-down camera layout coloured by lens.

## 9. Limitations and honest claims

- **Synthetic zoom on the benchmark.** A crop of a 640 × 480 frame resized up is blurrier than a real telephoto photo
  (§5 explains why this should not matter for the scale; the photo demo has real lenses).
- **Canonicalisation throws pixels away** on zoomed views (they shrink to the common angular resolution). Their extra
  detail is not used for geometry; recovering it (a per-view full-resolution pass aligned to the canonical depth) is
  monocular tiling, which is prior art, and is not part of the claim.
- **Wide views can get many tokens** (up to 1008 px); the cap keeps a T4 within memory but then the equality is approximate.
- **canon_self depends on DA3's own focal estimate.** If DA3 misjudges the zoom (L4), the plan is off; EXIF avoids that
  on real photos.
- **A/A caveat from #2:** the reference view affects DA3 by 3-4%; the reference is always view 0 (never zoomed), the same
  in every arm.
- **Not evaluated:** ETH3D, ScanNet++ (compute budget, as for #1-#3), NVS, monocular depth (unchanged by construction).

## 10. Deviations log

Any change to the protocol, the arms or the plan after the first Kaggle run is logged here with its date and reason,
before the affected results are read. `configs/lens_analysis.yaml` is unchanged.

- **2026-11-27, run setup (no gate affected).** The first full run (DA3-Small, commit 0da40ae) scored `unzoomed` and
  `native`, then the scoring of `canon_oracle` was killed by the system (exit -9, out of RAM) when the 7Scenes TSDF
  fusion started. The harness fuses 7Scenes with no depth truncation, 4 fusions ran in parallel, and the mixed-lens
  arms fuse more scattered geometry (scoring took 31 min for `unzoomed`, 45 min for `native`, and more than 36 min
  for `canon_oracle` before it was killed). Changes:
  1. the number of parallel fusions now comes from the RAM (~10 GB each, at most 4: 2 on Kaggle);
  2. the TSDF F1 is computed only for `unzoomed`, `native` and `canon_self` (`RECON_ARMS`). F1 is descriptive in the
     plan, and no gate uses it: L1 and L2 use the per-view scale ratio, L3 uses AUC@30, L4 uses the zoom estimates.
     The pose and per-view numbers of all five arms are unchanged and paired on the same frames (frame sampling does
     not depend on the modes).

  Seen before this change: the parity output and the last lines of the two finished scoring logs (7Scenes F1 of
  `unzoomed` and `native`, descriptive). No gate quantity (scale ratio, AUC@30, zoom error) was read. The change is
  driven by RAM and time, not by those numbers.

## 11. Code map

| Part | Where |
|---|---|
| exact geometry (affine maps, intrinsics, depth warps, zoom / portrait photos) | `src/da3/enhancements/lens/geometry.py` |
| spec, arms, view assignment, angular-resolution plan, EXIF focal | `src/da3/enhancements/lens/plan.py` |
| released / canonical / padded preprocessing, one forward over any view sizes | `src/da3/enhancements/lens/inference.py` |
| benchmark adapter (`LensAPI`, `lens_meta.npz`) | `src/da3/enhancements/lens/adapter.py` |
| per-view metrics, GT-free reprojection consistency | `src/da3/enhancements/lens/metrics.py` |
| Sec. 6 runs | `scripts/eval_bench.py --lens "arm=...,zoom=2,fraction=0.3"` |
| parity, per-view analysis, gates / tables / figures / records | `scripts/lens_eval.py parity | views | analyse` |
| photo demo, COLMAP reference | `scripts/lens_demo.py`, `scripts/colmap_reference.py` |
| tests | `tests/test_lens.py` |
| notebook | `notebooks/10_lens.ipynb` (generated by `notebooks/make_notebooks.py`) |
