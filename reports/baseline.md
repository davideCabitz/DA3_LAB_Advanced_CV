# DA3 baseline reproduction: status report

> **Strategy (2026-11-17): the baseline is the released DA3 weights.** Reproduction = parity (§3.2)
> + the Sec. 6 benchmark run by us with DA3-SMALL / DA3-BASE vs Tab. 2 / Tab. 3 (§3.4, pending on
> Kaggle). Enhancements will be measured against the released weights. From-scratch training of the
> `paper-*` model (§3.5) is optional, for architectural ablations only.

Written for the ACV lab project (reproduce DA3, then improve on it). Status as of 2026-11-17.
Every number carries its setting: **[paper]** = copied from the paper; **[official]** = released
weights run by us; **[reduced-scale]** = our training; **[sanity]** = pipeline checks, not results.

## 1. What is implemented (our code, `src/da3`)

| Component | Paper reference | Where |
|---|---|---|
| DINOv2 ViT with within-view / cross-view token reordering, L_s : L_g = 2 : 1 | Sec. 3.2 | `backbone/vit.py` |
| Camera tokens: E_c(f, q, t) MLP when poses are known, learnable tokens otherwise (reference view + one shared by the others, see §2); pose conditioning with p = 0.2 in training | Sec. 3.2, 3.4 | `camera/modules.py`, `scripts/train.py` |
| Dual-DPT: shared reassemble, separate fusion stacks and output layers; depth + confidence at H x W, 6-channel ray map at H/2 x W/2 | Sec. 3.2, Fig. 3 | `heads/dualdpt.py` |
| Camera head D_C: FOV, quaternion, translation from camera tokens | Sec. 3.1 | `camera/modules.py` |
| P = t + D·d; Eq. 1 (mean origin); Eq. 2 (DLT + RQ) | Sec. 3.1 | `geometry/rays.py` |
| L = L_D (confidence) + L_M + L_P + β L_C + α L_grad, α = β = 1, ℓ1; GT normalized by mean ℓ2 point norm | Sec. 3.3 | `losses/da3_loss.py` |
| Multi-resolution, 2-8 views, constant token budget, warm-up + cosine LR, fp16, gradient checkpointing, exact resume | Sec. 3.4 (scaled down) | `data/multiview.py`, `scripts/train.py` |
| Pose AUC@3/30, Umeyama + RANSAC Sim(3), TSDF, Chamfer / F1 | Sec. 6 | `eval/metrics.py` (+ released harness) |
| Inference, PLY / transforms.json / COLMAP / GLB export, Blender import | (demo) | `scripts/infer.py`, `export/`, `scripts/blender_import.py` |

Two families of model presets (`src/da3/config.py`):
- **`paper-*`: our trainable model** (optional from-scratch training). Only what the paper describes. Unspecified details are taken from the
  release (tapped layers, head widths, activations) or are our stated choices (λ_c = 0.2, predicted
  rays in L_P, AdamW).
- **`da3-*`: the released architecture.** Loads the official checkpoints, i.e. the baseline. It differs from the paper
  text in the layer layout, the camera tokens, QK-norm/RoPE, reference-view selection, head inputs,
  ray resolution, ray confidence and ray channel order (full table in `docs/PLAN.md`).

Borrowed: DINOv2 weights (init), DA3 checkpoints (parity and [official] eval), and the released
benchmark harness (loaders, sampling, evo alignment, Open3D TSDF), so the Sec. 6 protocol is identical.

## 2. Deviations from the paper and why

| Item | Paper | Here | Why |
|---|---|---|---|
| Learnable camera token | one shared token | reference-view token + one shared by the other views | with one shared token the view-0 pose targets are unidentifiable (camera loss 0.97 vs 0.065 in the overfit test, §3.5); decided by the project owner |
| Scale | ViT-L/G, 128 x H100, 200k steps, 504 px, 2-18 views, ~22 datasets, teacher labels after 120k | ViT-S, 1 x T4/P100, ≤ 20k steps, 252-px class, 2-8 views, 2 datasets, GT depth only | Kaggle budget; teacher out of scope |
| Resolutions | {504², 504x378, ...} | the same set x 0.5 (189 -> 196 so it is a multiple of 14) | compute |
| λ_c | not given | 0.2 | needed; DUSt3R default |
| d, t in L_P | ambiguous | predicted rays | couples depth and ray branch |
| D_C architecture | "lightweight ... transformer", no depth given | 2-layer MLP (release; matches Tab. 8 parameter counts) | open point |
| Benchmark harness | described in Sec. 6 | released code | protocol identity |
| Open3D | – | pinned to 0.19.0 | 0.20.0's ScalableTSDFVolume returned empty meshes in our test |

## 3. Results

### 3.1 Unit tests
`pytest -q`: **25 passed** (CPU). These cover ray map -> pose round trip (exact, plus noisy rays within
0.1° and 0.5% focal), depth + ray -> points consistency and reprojection, N_v = 1 vs N_v > 1
shapes, single view == monocular, token reordering preserves per-view content, permutation
equivariance, GT normalization, losses, and metrics.

### 3.2 Parity with the released weights [official], CPU fp32, 6 frames 280 x 504
Our modules vs the official code, same safetensors and same input tensor. Max abs difference:

| Model | Mode | depth (rel.) | depth conf | ray map | pose w2c | K |
|---|---|---|---|---|---|---|
| DA3-SMALL | mono | 8.9e-6 (9.7e-6) | 9.0e-5 | 4.8e-7 | 0 | 0 |
| DA3-SMALL | multi-view, unposed | 1.7e-5 (2.0e-5) | 1.6e-4 | 4.8e-7 | 4.5e-8 | 3.1e-5 |
| DA3-SMALL | multi-view, posed | 1.9e-5 (2.2e-5) | 1.8e-4 | 4.8e-7 | 3.0e-8 | 9.2e-5 |
| DA3-BASE | mono | 4.6e-6 (5.9e-6) | 3.0e-5 | 4.8e-7 | 0 | 0 |
| DA3-BASE | multi-view, unposed | 8.3e-6 (9.1e-6) | 1.2e-4 | 4.8e-7 | 2.6e-8 | 3.1e-5 |
| DA3-BASE | multi-view, posed | 7.7e-6 (9.0e-6) | 8.4e-5 | 4.8e-7 | 7.5e-8 | 9.2e-5 |

Ray map -> pose, on the same official ray maps: our plain Eq. 1 + Eq. 2 vs the official
confidence-weighted RANSAC + QL routine differ by at most 0.32° (Small) / 0.64° (Base) in rotation,
1.8e-4 in camera center and 0.05% in focal length.
Files: `experiments/local_cpu/da3-*_cpu_robot6.json` (local CPU, before the move to Kaggle). The GPU
rerun comes from `notebooks/01_parity`.

Findings while matching weights: (i) the released checkpoints store the ray-output LayerNorm only
under pyramid level 0, because one LayerNorm instance is shared by all four levels; (ii) 42 released
tensors (ray output layers for levels 0-2) are never used in the forward pass.

### 3.3 Metric implementation vs released harness
On 20 random trials our pose AUC and Sim(3) alignment equal the harness exactly (≤ 1e-15), and
Chamfer / F1 agree to ≤ 7e-18 (`experiments/local_cpu/metrics_crosscheck.json`). In a 2-scene
HiRoom smoke test, AUCs recomputed by our code from the harness' saved predictions were identical
(diff 0.0). `notebooks/03_eval` repeats the cross-check on Kaggle.

### 3.4 Benchmark (Sec. 6 protocol)
Not run yet: `notebooks/03_eval` (HiRoom, 7Scenes, DTU-64 first; DA3-SMALL and DA3-BASE [official],
then our checkpoint [reduced-scale]). `scripts/make_tables.py` builds the table with rows (a) paper,
(b) official run by us, (c) reduced-scale. A local CPU run on the full HiRoom set was stopped
unfinished when the project moved to Kaggle-only compute.

### 3.5 Training [sanity] / [reduced-scale]
- [sanity] Overfit one batch, DINOv2 init, CPU, synthetic scenes, 150 steps. Paper text literally
  (one shared camera token): total loss 8.93 -> 1.24, but the **camera term stalls at ~0.97**.
  With two learnable camera tokens (reference view + shared others), everything else identical:
  total 8.11 -> 0.29, camera term **0.065**. The single shared token makes the network
  permutation-equivariant over views, so it cannot identify view 0, whose frame the pose targets
  are expressed in. **Decision: the baseline (`paper-small`) uses the two-token variant**; the final
  preset reproduces the 0.29 / 0.065 run exactly.
- [sanity] Resume: 20 uninterrupted steps vs 10 steps + `--resume` give identical losses at steps 11 and 20.
- Both sanity checks above ran on the local CPU; `notebooks/00_setup` reruns them on the Kaggle GPU
  with fp16 (`scripts/resume_check.py`).
- [reduced-scale] Not run yet: `notebooks/02_train`.

## 4. Inference demo and Blender export [official]
(Local CPU run; `notebooks/04_demo_export` regenerates it on Kaggle. Added since and not yet run:
ray-map PNGs and a `--per-image` mode for per-image depth + ray maps; see docs/KAGGLE_EXAMPLES.md.)
`scripts/infer.py` on (a) 6 frames of the official `robot_unitree.mp4` with DA3-SMALL and (b) the
2-image `SOH` set with DA3-BASE. Each output folder has per-view depth / confidence / ray maps,
cameras from D_C and from the ray map, `points.ply`, `mesh.glb` (Open3D TSDF), `transforms.json`, a
COLMAP text model, previews and a zip including `blender_import.py`.
- Cameras from D_C vs from the ray map: max rotation difference 0.41° (robot) / 1.78° (SOH). Median
  relative point difference between depth + camera and P = t + D·d: 0.8% / 0.6%.
- Headless Blender (bpy 5.0.1): points (Geometry Nodes, vertex colours), mesh and 6 cameras with
  background photos. Points projected through each Blender camera land within **0.76 px** (max) of the
  OpenCV projection. fx ≠ fy is modelled with the scene pixel aspect (`pixel_aspect_y = fx/fy`), which
  is exact for one camera; the residual comes from per-camera fx/fy differences (1.013-1.021).

## 5. Failure cases observed
- **Dynamic content** (robot video, moving subject, static camera): violates the static-scene
  assumption (Sec. 8 lists dynamic scenes as future work). Not evaluated quantitatively; treat the
  robot geometry in that demo as unreliable.
- **Sky** (SOH, DA3-BASE has no sky head): the sky gets finite depth and appears as a floating layer
  in the point cloud.
- **Single shared camera token** (paper text): poses relative to view 0 are not identifiable (§3.5); fixed in the baseline.
- Open3D 0.20.0: legacy TSDF integration produced empty meshes (pinned to 0.19.0).

## 6. Not done yet
All remaining compute runs on Kaggle (project rule). Ready but not yet run:
- `00_setup`: Kaggle limits (GPU / RAM / disk), unit tests, GPU fp16 overfit + resume checks.
- `01_parity`: parity on GPU.
- `02_train`: data preparation (TartanAir, vKITTI2) and the reduced-scale run.
- `03_eval`: benchmark (HiRoom / 7Scenes / DTU-64 first, then DTU / ScanNet++ / ETH3D as disk allows).
- `04_demo_export`: demo + Blender package on sample sets and your photos.
- Not implemented: VGGT / Pi3 / DA2 baselines.

## 7. Compute used so far
Before the move to Kaggle: local CPU only (AMD Ryzen 7 PRO 7840U, 30.7 GB RAM, torch 2.11.0+cpu) for
the unit tests, parity, metric cross-check, sanity runs and demo. Parity: 4.6 s (Small) and 13.1 s
(Base) for one 6-view comparison (ours + official). Kaggle GPU hours: none yet.

## 8. Reproduce (Kaggle)
Push the repo, then run the notebooks in order (GPU, Internet on). Each notebook calls the scripts
listed here; outputs land in `/kaggle/working`.

| Result | Notebook | Script |
|---|---|---|
| Environment, unit tests, overfit + resume sanity | `00_setup` | `env_report.py`, `pytest`, `train.py --overfit-one-batch`, `resume_check.py` |
| Parity | `01_parity` | `extract_frames.py`, `parity_check.py` |
| Reduced-scale training | `02_train` | `prepare_tartanair.py`, `prepare_vkitti2.py`, `check_dataset.py`, `train.py --resume` |
| Metric cross-check, benchmark, tables | `03_eval` | `check_metrics_vs_official.py`, `eval_bench.py`, `make_tables.py` |
| Demo, Blender package, Blender check | `04_demo_export` | `infer.py`, `render_preview.py`, `blender_smoke_test.py` |

Official repo pinned at github.com/ByteDance-Seed/Depth-Anything-3 commit 3d835ec (cloned by the notebooks).

## 9. Next steps: your own photos -> multi-view reconstruction -> Blender
DA3 is **feed-forward**: a folder of your photos goes straight through the pretrained model, with no
training needed. Upload the photos as a Kaggle dataset, attach it to `04_demo_export`, set
`MY_IMAGES`, run the notebook, and download `/kaggle/working/demo/my_images.zip`. Unzip it and run
`blender_import.py` in Blender (Scripting > Run Script). Fine-tuning only matters if the quality is
not good enough.

Already supported: any number of views; optional known poses (`--poses`, E_c conditioning, depth
rescaled to the given poses); cameras from D_C or from the ray map; confidence filtering; PLY / GLB /
transforms.json / COLMAP export; Blender import with correct OpenCV -> Blender conversion.

Missing / to do:
- Images with different aspect ratios are center-cropped to a common size (as in the release);
  intrinsics in that case are only adjusted for scale.
- No sky / background masking for Small/Base (sky becomes floating geometry); a simple confidence
  threshold (`--conf-percentile`) is the current workaround.
- Larger checkpoints (DA3-LARGE-1.1 / GIANT-1.1) are CC BY-NC and need a GPU; loading them uses the
  same `da3-large` preset (Giant preset not added).
- Fine-tuning on your own scenes needs depth / pose supervision (e.g. COLMAP of your photos converted
  to the canonical format); the training script supports this once a converter for COLMAP output is
  written. Self-supervised (photometric) fine-tuning is not implemented and not part of the paper.
- Mesh quality: TSDF voxel size is set from the median depth (no metric scale); for large scenes,
  tune `voxel` in `export/exporters.py::tsdf_mesh`.
