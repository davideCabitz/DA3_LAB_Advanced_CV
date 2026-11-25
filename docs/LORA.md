# LoRA fine-tuning of the released DA3 (enhancement #3 platform): status, expectations, method

**Status: proposal, 2026-11-25. Nothing in this document is implemented yet; it waits for the project
owner's approval.** The statistics are pre-registered separately in
[LORA_ANALYSIS_PLAN.md](LORA_ANALYSIS_PLAN.md) (constants in `configs/lora_analysis.yaml`).

Sources: [da3_enhancement_report.md](da3_enhancement_report.md) §2.2 (K8), §3.2–3.5 (novelty),
§4.4 (E3), §5; [DYNAMIC_SCENES.md](DYNAMIC_SCENES.md) C4; the paper (Sec. 3, Sec. 6, Tab. 4, 7, 8);
the code under `src/da3/`. Setting tags: **[paper]**, **[official]** (released weights, run by us),
**[official+LoRA]** (released weights + our adapter).

---

## Phase 0: what the repository has today

Checked on 2026-11-25 against commit `95468b0` (working tree: only `docs/prompts/` untracked).

| Requirement | Status | Evidence | What is missing |
|---|---|---|---|
| Released weights load (DA3-SMALL, DA3-BASE) | done | `src/da3/checkpoint_io.py:16-17` (HF repos), `:76-90` (`official_model`, strict load); presets `src/da3/config.py:85-86` | – |
| Licences | done | Apache-2.0 for Small and Base (`docs/PLAN.md:90-93`); Large-1.1 tag/README disagree (CC BY-NC) | – |
| Parity vs official code | partial | CPU parity passes: depth ≤ 1.9e-5 S/B, monocular / unposed / posed (`experiments/local_cpu/da3-small_cpu_robot6.json`, `da3-base_cpu_robot6.json`; `scripts/parity_check.py`) | GPU parity (`01_parity`) still `[!]` in `TODO.md:14` |
| Training script | partial | `scripts/train.py:197-340`: fp16 autocast + GradScaler (`:221`, `:250`), exact resume (`:252-261`), `--time-budget-h` (`:205`), registry (`:338`); init from released weights via `init: {type: official}` (`:187-189`) | Built for from-scratch training: the optimizer takes **all** parameters (`:240-246`); no freezing, no teacher, no anchor, no adapter-only checkpoint; GT must be dense depth + poses |
| Losses | done | `src/da3/losses/da3_loss.py`: `normalize_gt` (`:33-62`, view-0 frame, mean ℓ2 point norm), `conf_l1` = L_D (`:71-76`), `grad_l1` = L_grad (`:79-88`), L_M (`:105`), L_P (`:107-109`), L_C (`:111-112`), weights in `LossConfig` (`:21-29`); tests `tests/test_losses.py` | Distillation targets (teacher depth instead of GT), per-clip scale alignment of targets, anchor term |
| Gradient checkpointing | partial | Per backbone block, `src/da3/backbone/vit.py:188-191`, active only in `train()` mode | Not for the Dual-DPT head, which runs in fp32 (`src/da3/model.py:185-186`) and holds most activation memory |
| LoRA module, config keys, merge / unmerge, adapter save / load | **missing** | `grep -i lora\|peft` over `src/ scripts/ tests/` finds nothing | Everything; `TODO.md:46` lists it as open |
| "LoRA off = bit-identical" test | **missing** | – | Test plus B = 0 at init, gradients only in A/B, merged = unmerged |
| LoRA training data (KITTI raw) | **missing** | No KITTI converter or loader; only `scripts/prepare_vkitti2.py`, `scripts/prepare_tartanair.py` (both datasets are in DA3's Tab. 1 training set) | User download after registration at cvlibs.net (raw drives: `image_02`, `oxts`, `velodyne_points`, calib), private Kaggle dataset, converter to the canonical format |
| Train/test scene disjointness | partial | Sec. 6 sets (HiRoom, ETH3D, DTU, 7Scenes, ScanNet++) contain no KITTI, so a KITTI LoRA is disjoint from them by construction | KITTI Eigen test drives (Tab. 4 monocular) must be excluded from the LoRA training and validation drives; a drive-level split file with a check |
| Benchmark harness | done | Pinned clone 3d835ec, our model plugged in via `src/da3/eval/bench_adapter.py:34-98`; metric cross-check `scripts/check_metrics_vs_official.py` | Loading a base model + adapter in `scripts/eval_bench.py:53-63` (`build_ours` knows only `official:` and full checkpoints) |
| Sec. 6 data on Kaggle / Colab | partial | Downloaded per run from HF `depth-anything/DA3-BENCH`: HiRoom 0.72 GB, DTU-64 1.78 GB, 7Scenes 3.55 GB, DTU 8.87 GB, ScanNet++ 10.83 GB, ETH3D 15.1 GB (`docs/PLAN.md:99-100`) | ETH3D and ScanNet++ need their own session (disk); `03_eval` baseline runs still `[!]` (`TODO.md:15-17`) |
| Open3D / TSDF | done (workaround) | `scripts/check_open3d.py`; Python 3.12 + Open3D 0.19.0 side environment and `--infer-only` / `--eval-only` split (`scripts/eval_bench.py:113-116`, `docs/ENHANCEMENTS_RUNBOOK.md:72-88`) | – |
| Result record and comparison | partial | Schema `da3-result/1` (`src/da3/compare/schema.py:29-63`), `scripts/compare_results.py`, `09_compare_all` | Records are paired only when `weights` match (`schema.py:86-89`), so a LoRA record must keep `weights="official:da3-small"` and store the adapter in a new optional field; per-seed records and the new setting tag `official+LoRA`; runbook names #3 `pose_distill` (`docs/ENHANCEMENTS_RUNBOOK.md:22`), the prompt asks for `experiments/results/lora/` (to be decided, see "Open decisions") |
| Monocular-depth benchmark (Tab. 4: KITTI, NYU, Sintel, ETH3D, DIODE) | **missing** | No loader. `src/da3/eval/gt_depth.py` decodes per-frame GT depth of HiRoom / 7Scenes / ETH3D from DA3-BENCH (used by Self-Audit), which allows an N_v = 1 depth evaluation on ETH3D without new downloads | NYUv2, Sintel, DIODE downloads and loaders; KITTI Eigen test comes with the KITTI raw download. The DA3 paper does not state its Tab. 4 alignment (DA2 [113] aligns affine-invariant disparity): open point |
| NVS (GS-DPT head, DL3DV / T&T / MegaDepth) | **missing** | No GS head, no 3DGS renderer, no NVS loader in the repo | **Not evaluated.** Building the Sec. 5 head is a separate project |
| Demo path | done | `scripts/infer.py` (multi-view, `--per-image`, `--poses`), `src/da3/export/exporters.py` (PLY, GLB, transforms.json, COLMAP text, TSDF / depth mesh), `scripts/blender_import.py`, `scripts/blender_turntable.py` | `--lora` option in `infer.py`; the turntable frames each scene from its own points (`blender_turntable.py:24-36`), so two models get different framings: needs a shared framing file and a Sim(3) alignment of the two reconstructions (Phase 5) |
| GPU budget | estimated | See the table below (not measured; no run was made) | A memory probe cell as the first step of notebook 07 |
| Self-Audit (#1) results | not arrived | `experiments/results/` contains only `README.md`; `05_self_audit` still `[!]` (`TODO.md:31`) | Not needed: the LoRA is measured against the released weights |

### GPU budget (estimate, not measured)

Assumptions: LoRA on blocks 4–11, base weights frozen (no gradients or optimizer state for them),
backbone checkpointing on, fp16 autocast (the head runs fp32), batch = 1 clip, SDPA attention.
Trainable state (A, B, Adam moments) is < 20 MB and does not matter. Activation memory is dominated
by the Dual-DPT head, which keeps full-resolution fp32 feature maps for the backward pass (about
0.35–0.5 GB per 504×280 view for Small and about twice that for Base, by counting the stored tensors
in `src/da3/heads/dualdpt.py`). Gradients must flow through the frozen head into the backbone, so
freezing the head does not remove these activations.

| Model, views × resolution | Estimated peak | T4 / P100 (16 GB) | L4 (24 GB) | A100 (40 GB) |
|---|---|---|---|---|
| Small, 8 × 504×280 (KITTI crop) | 4–6 GB | fits | fits | fits |
| Small, 16 × 504×280 | 8–11 GB | fits | fits | fits |
| Small, 8 × 504×504 | 7–10 GB | fits | fits | fits |
| Base, 8 × 504×280 | 8–11 GB | fits, little headroom | fits | fits |
| Base, 16 × 504×280 | 16–21 GB | no (needs head checkpointing) | borderline | fits |

Time (rough, from Tab. 8's A100 speeds scaled to a T4 and ×4 for forward + recompute + backward):
Small ≈ 1.5 s and Base ≈ 3 s per 8-view step, so 2000 steps ≈ 1 h (Small) or ≈ 2 h (Base) per seed.
Full fine-tuning would also fit in memory (Adam state for all weights: Small ≈ 0.5 GB, Base ≈ 2.2 GB);
memory is therefore not the reason to prefer LoRA (Phase 2).

---

## Phase 1: what the LoRA should improve, and why

**Setting.** E3 (report §4.4): a domain with known poses but no dense depth. The **teacher** is the
frozen released DA3 conditioned on the known poses through E_c; its outputs are cached once. The
**student** is the same DA3, unposed, plus a LoRA. At test time the student needs no poses.
First domain: KITTI raw (real driving; DA3's Tab. 1 has only synthetic vKITTI2), poses from OXTS,
LiDAR used for evaluation only.

**Expected effects** (report §2.2 / §4.4, pessimistic / realistic / optimistic):

| Where | Metric | Expected change |
|---|---|---|
| Held-out KITTI drives (in domain) | ATE, RPE | −5% / −20–35% / −50% |
| Held-out KITTI drives | AbsRel vs LiDAR | 0% / −3–8% / −15% |
| Gap between unposed and posed DA3 on KITTI | fraction closed | 5% / 20–40% / 60% |
| Sec. 6 (out of domain), LoRA on | AUC@3 | 0 to −2 points (forgetting) |
| Monocular (N_v = 1) out of domain | δ1, AbsRel | unchanged within the margins |

**Why gains are plausible.**
- Domain gap: DA3 has seen no real driving data; long forward motion, 3.3:1 frames (cropped) and
  outdoor depth ranges are under-represented.
- Privileged information: the teacher sees the true poses, which remove the scale / rotation
  ambiguities that hurt unposed DA3 most on forward motion; its depth is made consistent with those
  poses. Tab. 7 (f vs g) shows conditioning helps on 4 of 5 datasets [paper].
- Low-rank updates are known to be enough to adapt a pretrained ViT to a new domain at a small
  parameter cost, and the low rank limits how far the model can move (less forgetting).

**Why it could fail.**
- Out of domain, conditioned depth may be no better than unposed depth (Tab. 7: −12.4 F1 on
  7Scenes with conditioning [paper]). The training-free go/no-go (posed vs unposed DA3 on 2 drives)
  checks this before any training.
- E_c may simply copy its input poses (SpatialBench). The teacher then adds only depth made
  consistent with GT poses: still useful, but reported as such.
- If only the poses have headroom, E3 is ordinary pose-supervised fine-tuning (not new). The
  pose-only control isolates this.
- Forgetting outside the domain (Sec. 6, monocular).
- Gauge mismatch: the posed path skips the reference-view selection and normalises poses by the
  median camera distance; the unposed path picks a reference view and has its own scale. Phase 3
  fixes the reference view and aligns the scale per clip.

**Outcomes, stated before any run** (decisions in [LORA_ANALYSIS_PLAN.md](LORA_ANALYSIS_PLAN.md) §3):
- **Success:** the primary endpoint (held-out KITTI AbsRel) improves significantly, *and* the
  Sec. 6 / monocular non-inferiority components all pass. The novelty claim additionally needs E3 to
  beat the pose-only control.
- **Null:** the CI of the primary effect lies within ±3% (equivalence). Reported as "no effect".
- **Regression:** the primary CI lies entirely on the worse side, or any non-inferiority component
  fails. Reported as such.
- Anything else is **inconclusive** and is reported as inconclusive.

---

## Phase 2: methods similar to LoRA (need some training, much less than full training)

Parameter counts are computed for the released ViT-S / ViT-B (12 blocks, D = 384 / 768) restricted
to blocks 4–11 unless stated. Full model ≈ 34M (Small) / ≈ 135M (Base) parameters, from the fp32
file sizes (Tab. 8 lists only backbone + DPT + D_C). "Memory" is for 8 views at 504×280 on a T4.

| Method | What is trained | Params Small / Base | Memory on T4 | Forgetting risk | Switchable at inference | Already published on DA3 / VGGT-like models? |
|---|---|---|---|---|---|---|
| Full fine-tuning | all weights | 34M / 135M | +0.5 / +2.2 GB over LoRA; fits | high | only by keeping two models | yes: how DA3, VGGT, π³ are trained; plain fine-tuning with pose supervision is cited, not claimed (report §3.4) |
| Partial FT: last blocks (8–11) | 4 blocks | 7.1M / 28.4M | ≈ LoRA | medium–high | two copies of 4 blocks | generic; not checked for DA3 |
| Partial FT: heads only (Dual-DPT + D_C) | head weights | ≈ 4M / ≈ 19M | low (no backprop into the backbone) | low (backbone unchanged) | yes (head copy) | trained light modules on frozen VGGT / π³ / DA3: PrePARE (2605.08371) |
| Partial FT: LayerNorm only | LN γ, β (all 12 blocks) | 18k / 37k | ≈ LoRA | low | yes | generic (TENT-style); not checked for DA3 |
| BitFit | biases (all 12 blocks) | 51k / 101k | ≈ LoRA | low | yes | generic; not checked for DA3 |
| (IA)³ | scaling vectors on k, v, MLP hidden | 18k / 37k | ≈ LoRA | low | yes | generic; not checked for DA3 |
| **LoRA** (r = 8 / 16, qkv + proj + fc1 + fc2) | A, B per linear | 0.39M / 0.79M (r 8); 0.79M / 1.57M (r 16) | 4–6 / 8–11 GB | low–medium | yes, exact | yes as **test-time** adaptation of DA3: Free Geometry (2604.14048), Self-Geometry (2608.10708). The pose-conditioned teacher is what is new (report §3.4) |
| DoRA | LoRA + per-column magnitude | LoRA + 28k / 55k | ≈ LoRA | low–medium | yes | generic; not checked for DA3 |
| Adapters (Houlsby / parallel), bottleneck 64 | 2 small MLPs per block | 0.79M / 1.59M | ≈ LoRA | low–medium | yes (skip) | generic; not checked for DA3 |
| Visual prompt / prefix tuning (VPT-deep, 10 tokens) | prompt tokens per block | 31k / 61k | ≈ LoRA + tokens | low | yes | test-time prompt tuning of DUSt3R: Test3R (2506.13750). On DA3 it interacts with the camera-token slot and RoPE positions of special tokens |
| New head on the frozen backbone | e.g. a copy of the ray branch + D_C | ≈ 1–2M / 5–8M | low | none for existing outputs | yes | PrePARE; add-on heads (report K7) |
| Test-time adaptation (per scene) | LoRA or LN per test scene | as LoRA / LN | as LoRA, per scene | per scene only | yes | yes on DA3: Free Geometry, Self-Geometry; TCO (2604.03878) optimises weights with pose priors at test time |
| Knowledge distillation into a smaller model | the student | all | – | – | – | yes: Distill3R (2602.00865), OTT3R (2609.36374) |

"Not checked for DA3" means our 2026-11-19 novelty search did not look for it; we claim nothing
about these methods.

**Recommended controls next to the E3 LoRA** (same data, steps, seeds, rank and evaluation):
1. **Pose-only LoRA** (L_M + L_C against the known poses, no teacher depth). Required: it isolates
   the novel term. If E3 does not beat it, E3 is ordinary pose-supervised fine-tuning.
2. **Own-output LoRA** (all targets = the frozen unposed DA3 outputs on the same KITTI clips).
   Required: no new information enters, so it measures what the training pipeline does by itself
   (drift, resolution / crop effects). It is also the treatment-blind noise floor of training.
3. **Free Geometry** (published test-time LoRA on DA3, public code). Recommended if its code runs
   with our weights in ≤ 4 GPU-h; it is the closest published baseline.
4. Optional, one seed: **heads-only fine-tuning** with the E3 losses. Cheap; tells whether
   backbone adaptation is needed at all.

Not recommended as controls: DoRA / adapters / IA³ / BitFit (same family, they would answer a PEFT
question, not the E3 question), full fine-tuning (loses switchability, more forgetting).

---

## Phase 3: what LoRA does inside DA3, step by step

### 3.1 The math

For a frozen linear layer `y = W x + b` with W ∈ R^{out×in}, LoRA adds a trainable low-rank path:

```
y = W x + b + (alpha / r) · B (A · dropout(x)),     A ∈ R^{r×in},  B ∈ R^{out×r}
```

- **Init:** A random (Kaiming-uniform, as in PEFT), **B = 0**. At step 0 the extra path is exactly
  zero, so the model is the released DA3. The first gradient reaches B (∂L/∂B = (α/r)·g·(Ax)ᵀ ≠ 0)
  while ∂L/∂A = 0 until B moves; this is the intended asymmetry.
- **Hyper-parameters (proposed):** r = 16, α = 16 (scale α/r = 1), dropout 0.0 (proposed: the data
  is small but the targets are smooth; dropout would also make LoRA-on runs non-deterministic).
  Ablation r = 8 only if budget remains.
- **Trainable parameters** (blocks 4–11, qkv + proj + fc1 + fc2): r·(in + out) per layer →
  Small: 0.79M at r = 16 (≈ 2.3% of 34M), 0.39M at r = 8; Base: 1.57M at r = 16 (≈ 1.2% of 135M),
  0.79M at r = 8. Attention-only (qkv + proj): Small 0.29M, Base 0.59M at r = 16.

### 3.2 Where the adapters go (verified module names)

Released layout (`da3-small`, `da3-base`; `src/da3/config.py:70-74`, `src/da3/backbone/vit.py:150-153`):
`alt_start = 4`, `cross_first = False`, so

| Blocks | Attention | Paper name | LoRA |
|---|---|---|---|
| `backbone.blocks.0–3` | within-view only | L_s | **no** |
| `backbone.blocks.4, 6, 8, 10` | within-view | L_g | yes |
| `backbone.blocks.5, 7, 9, 11` | cross-view (all views' tokens jointly) | L_g | yes |

In each LoRA block: `attn.qkv` (D→3D), `attn.proj` (D→D), `mlp.fc1` (D→4D), `mlp.fc2` (4D→D)
(`src/da3/backbone/layers.py:259-299`).

Why blocks 4–11 and not 0–3:
- The camera token enters at block 4: it overwrites the cls slot when `i == alt_start`
  (`vit.py:226-227`). Blocks 4–11 are exactly the part of the network that sees the camera token and
  does cross-view reasoning.
- **Gauge safety:** the reference view is chosen from the cls tokens *before* block 3, i.e. from the
  output of block 2 (`vit.py:221-225`). A LoRA on blocks ≥ 4 cannot change which view becomes the
  reference, so the LoRA cannot move the gauge by itself.
- Blocks 0–3 are the most generic (DINOv2-like) features; leaving them frozen limits forgetting.

Frozen, with reasons:
- **E_c** (`cam_enc`: `pose_branch` MLP 9→D/2→D, `token_norm`, 4 `_TrunkBlock`s, `trunk_norm`,
  `src/da3/camera/modules.py:280-302`): only the teacher uses it, and the teacher must stay the
  released model. Its `_TrunkBlock`s use the same `Attention` / `Mlp` classes as the backbone, so the
  LoRA injector must select modules by **name** (`backbone.blocks.{4..11}.…`), never by class.
- **Dual-DPT head** (`head`: shared `projects` / `resize_layers` / `adapters`, `depth_fusion` →
  `depth_neck` → `depth_out`, `ray_fusion` → `ray_neck` (5 convs) → `ray_out`;
  `src/da3/heads/dualdpt.py:96-181`) and **D_C** (`cam_head`: `backbone` MLP, `fc_t`, `fc_qvec`,
  `fc_fov`, `modules.py:305-323`): frozen in the main arm. The domain gap is mostly in matching and
  geometry (backbone), and keeping the heads fixed keeps the output parameterisation identical. The
  heads-only control (Phase 2, option 4) tests the alternative.
- Learnable camera tokens (`backbone.camera_token`, 2 tokens: reference / others) and positional
  embeddings: frozen.

### 3.3 The forward pass with the adapters

`DA3.forward` (`src/da3/model.py:161-190`), unposed student:
1. images (B, S, 3, H, W), ImageNet-normalised → `backbone.patch_embed` (14×14 conv) + cls token +
   interpolated `pos_embed` → tokens (B, S, 1 + h·w, D).
2. Blocks 0–2 within-view; before block 3 the reference view is selected (`ref_strategy`; we use
   `"first"`, the benchmark's setting) and views are permuted so it comes first.
3. Block 4: the cls slot is overwritten with the camera token: the learnable `camera_token`
   ([reference, others], `vit.py:171-176`) for the unposed student, or `cam_enc(f, q, t)` for the
   posed teacher. RoPE and QK-norm are active from block 4.
4. Blocks 4–11: within-view blocks reshape tokens to (B·S, N, D), cross-view blocks to (B, S·N, D)
   (`vit.py:41-58`); in each, `qkv`, `proj`, `fc1`, `fc2` add their LoRA path.
5. At `out_layers` (5, 7, 9, 11) the features `[last within-view output, LN(current)]` (2D channels)
   are collected and the views are permuted back (`vit.py:238-250`).
6. `head` (fp32): depth = exp(·), confidence = 1 + exp(·) at H×W; ray map (6 channels,
   [origin, direction]) + ray confidence at the 8× patch grid. `cam_head` on the camera-token
   features of block 11 → pose encoding [t, q, fov] (9).
7. Geometry: P = t + D·d (`da3.geometry.points_from_depth_ray`); cameras from D_C
   (`DA3.cameras_from_head`, the API default) or from the rays by Eq. 1 + Eq. 2
   (`DA3.cameras_from_rays`, DLT + RQ), or the released RANSAC solver in the benchmark.

### 3.4 One training step

Data: a KITTI clip of S = 8 consecutive `image_02` frames (stride chosen so neighbouring frames are
≈ 1–3 m apart), centre-cropped to 1.8:1 and resized to 504×280; K from calib, w2c from OXTS → camera
(velo-to-cam and imu-to-velo calibrations).

1. **Teacher (offline, cached once per clip):** released DA3, `model(x, w2c=w2c_gt, K=K)`. The posed
   path skips the reference selection, so the reference is view 0. Cached: depth, depth confidence,
   ray map, D_C pose. The teacher depth is in the gauge of the normalised input poses (median camera
   distance = 1, `pose_encoding.py:93-102`); it is brought to metres with the Umeyama scale between
   the teacher's D_C poses and the input poses, exactly as the released API does
   (`bench_adapter.py:79-83`). Validity mask: teacher confidence above the clip's 30th percentile
   (proposed) and depth < 80 m.
2. **Targets:** `normalize_gt(D_T, valid_T, K, w2c_gt, ray_hw)` (`da3_loss.py:33-62`) gives depth,
   points, ray map and pose encoding in view 0's frame, divided by the mean ℓ2 norm of the valid
   points (Sec. 3.3). Because the released training normalisation is unpublished, the student's
   output scale may differ: a per-clip scale s* = median(D̂ / D) over valid pixels (detached) multiplies
   the target depth, points, ray origins and translations before the loss.
3. **Student:** `model(x, w2c=None, ref_strategy="first")` with the LoRA on: the reference is view 0,
   the same as the teacher's, so both are in one gauge.
4. **Loss:** `DA3Loss` with its defaults (`LossConfig`: L_D confidence-weighted vs teacher depth,
   L_grad, L_M vs the GT-pose ray map, L_P, L_C vs the GT pose encoding; α = β = 1, λ_c = 0.2), plus an
   **anchor** λ_a · ℓ1(student, frozen unposed DA3) on depth, ray map and pose encoding, computed on a
   small *replay* set outside KITTI and outside every test set (TartanAir via
   `scripts/prepare_tartanair.py`; it is in DA3's Tab. 1, so it is not new information). The anchor
   targets are cached once. λ_a = 0.1 (proposed).
   - Pose-only control: only L_M + L_C (+ anchor). Own-output control: every target = the frozen
     unposed outputs on the same KITTI clips (+ anchor).
5. **Backward:** every base parameter has `requires_grad = False`; gradients flow through the frozen
   head and blocks (activations are kept, with checkpointing for the blocks) and reach only A and B.
6. **Optimiser:** AdamW over the LoRA parameters only, lr 2e-4, weight decay 0, betas (0.9, 0.999),
   warm-up 100 steps, cosine to 1% (`train_utils.lr_lambda`), 2000 steps, grad clip
   1.0, fp16 autocast + GradScaler, 1 clip per step. Seeds 0, 1, 2 (they set A's init and the clip
   order). Hyper-parameters are frozen in `configs/lora_analysis.yaml`; any tuning uses the KITTI
   validation drives only, never the test drives or Sec. 6.

### 3.5 Inference

- **Switch:** `set_lora_enabled(model, False)` makes every LoRA layer call only the base layer (the
  low-rank path is skipped, not multiplied by 0), so the output is bit-identical to the released DA3.
- **Merge:** W ← W + (α/r)·B·A for zero inference overhead. Merging changes floating-point rounding,
  so "merged = unmerged" is tested with a tolerance (fp32, atol 1e-5); unmerge restores the saved
  original W (bit-exact), not W − ΔW.
- **Checkpoints:** adapter-only file `{lora_config, base_preset, base_weights_sha256, A/B tensors,
  seed, commit}` (≈ 3 MB for Small r = 16). Loading refuses a base whose SHA-256 differs.
- **Tests (to write):** LoRA off = bit-identical to the released model (all outputs, `torch.equal`);
  B = 0 at init → identical outputs; after one optimiser step only A/B changed and only they have
  gradients; merged ≈ unmerged; unmerge restores W exactly; E_c untouched by the injector;
  save → load round trip.

---

## Phase 5 (design, implemented after approval): same photos, two models

- Image sets: benchmark scenes (with GT) and the personal photo sets of
  [KAGGLE_EXAMPLES.md](KAGGLE_EXAMPLES.md), first the **50-photo fountain** (requested 2026-11-25).
  The fountain has no GT, so it shows how the reconstruction *changes*, never whether it is better.
  For a KITTI-trained LoRA it is out of domain, so it is a visual forgetting check, reported next to
  (not instead of) N1–N6.
- Arms on the fountain: released, E3 (each seed), pose-only, own-output; the posed teacher only if
  the fountain gets poses (e.g. COLMAP), otherwise left out.
- **Training snapshots:** the same 50 photos are reconstructed with the adapter saved every 500 steps
  (and at step 0 = released, as a bit-identity check), so the drift over training can be shown as a
  sequence of renders and as Chamfer-to-released vs step. Descriptive only.
- `scripts/infer.py --weights official:da3-small [--lora adapter.pt]` on identical image sets,
  frame selection, resolution (504) and seed; both write depth, confidence, ray maps, cameras,
  PLY / GLB / transforms.json.
- A new `scripts/compare_reconstructions.py`: per-view depth difference maps (after one per-scene
  scale), pose differences after Sim(3) alignment (rotation °, translation), Chamfer / F1 between the
  two point clouds (and vs GT where available), confidence statistics, number of valid points.
- Blender: align the LoRA reconstruction to the released one (Sim(3) on camera centres), compute the
  framing (mode, centre, size, up, zoom) **once** from the released reconstruction and pass it to
  both renders (a new `--framing-from <json>` option in `blender_turntable.py`); same resolution,
  frame count, lighting; then a side-by-side video (and optionally points coloured by their distance
  to the other cloud). Outputs saved next to the result records and referenced in
  `reports/enhancement_3_lora.md`.

---

## Open decisions for the project owner

1. **Go/no-go first:** run the training-free check (posed vs unposed DA3-Small on 2 KITTI drives,
   ~1 GPU-h) before any LoRA code is used for E3. The LoRA platform itself is useful anyway.
2. **KITTI download:** which drives. Proposal in the analysis plan: training and validation drives
   from the KITTI raw *city / residential / road* categories that are **not** in the Eigen test
   split; test drives = Eigen test drives (so Tab. 4's KITTI number is the in-domain test as well).
3. **Results folder:** `experiments/results/lora/` (as asked) vs `pose_distill` (current runbook).
   Proposal: enhancement `"lora"`, variants `e3`, `pose_only`, `own_output`, `free_geometry`.
4. **Sizes:** DA3-Small = primary (all arms × 3 seeds); DA3-Base = replication (E3 + pose-only,
   2 seeds) only if the Small budget allows.
5. **Monocular NYU / Sintel / DIODE:** download and loaders (≈ 2 days of work + a few GB) or restrict
   the monocular no-regression check to ETH3D from DA3-BENCH.
