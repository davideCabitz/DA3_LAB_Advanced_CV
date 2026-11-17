# DA3 reproduction plan (Kaggle)

Every result is tagged with its setting: **[paper]** (numbers copied from the paper),
**[official]** (released weights, run by us), **[reduced-scale]** (our training).

## Strategy (decided 2026-11-17)
1. **Baseline = the released DA3 weights** (DA3-SMALL and DA3-BASE, Apache-2.0; DA3-LARGE-1.1 optional).
   Reproduction = (a) parity: our code gives the released model's outputs, and (b) the Sec. 6
   benchmark run by us with those weights, compared with Tab. 2 / Tab. 3. Steps: notebooks
   `00_setup` -> `01_parity` -> `03_eval` (docs/BASELINE_RUNBOOK.md).
2. **Enhancements are measured against the released weights, one change at a time, same protocol.**
   Routes in order of preference: training-free changes; fine-tuning the released weights with vs
   without the change (LoRA or full, same data / steps / seeds; LoRA support planned); a new head on a
   frozen backbone; from-scratch reduced-scale training (`02_train`, `paper-*` presets) only for
   backbone-architecture ideas, with several seeds and two sizes, presented as scaled-down ablations.
3. Training from scratch (`02_train`) is therefore **optional**, not part of the baseline.

## Guiding rule for our own implementation (from the project owner)
Our trainable model (`paper-*` presets, used by `02_train`) implements **only what the paper
describes**. Where the paper is silent on a detail that is needed to build the model, we take it from
the authors' release and say so. The released architecture differs from the paper text in several
places. It is kept as a separate set of presets (`da3-*`), used to load the released checkpoints:
parity, the baseline evaluation and, later, fine-tuning experiments on the released weights. It is
never used for from-scratch training.

## Official release (checked 2026-11-17)
- Code: github.com/ByteDance-Seed/Depth-Anything-3, Apache-2.0, commit `3d835ec` (2026-07-27).
  Provides an inference API, a CLI, exporters and the benchmark harness (`depth_anything_3.bench`).
  **No training code and no loss code are released.**
- Weights (HF `depth-anything/*`):

  | Model | File | HF license tag | README license |
  |---|---|---|---|
  | DA3-SMALL | 137 MB | apache-2.0 | Apache 2.0 |
  | DA3-BASE | 542 MB | apache-2.0 | Apache 2.0 |
  | DA3-LARGE-1.1 | 1.64 GB | apache-2.0 | CC BY-NC 4.0 (tag and README disagree) |
  | DA3-GIANT-1.1 | 5.42 GB | cc-by-nc-4.0 | CC BY-NC 4.0 |

  The `-1.1` checkpoints were retrained after a training-bug fix; the paper's numbers come from the
  original checkpoints (no `-1.1` variant exists for Small and Base).
- Benchmark data: HF `depth-anything/DA3-BENCH` (license tag "other"): HiRoom 0.72 GB, DTU-64 1.78 GB,
  7Scenes 3.55 GB, DTU 8.87 GB, ScanNet++ 10.83 GB, ETH3D 15.1 GB.

## Paper text (`paper-*` presets, from-scratch training) vs released architecture (`da3-*` presets)

| Aspect | Paper text (`paper-*`) | Released code (`da3-*`) |
|---|---|---|
| Layer layout | first L_s within-view, then L_g alternating cross / within, L_s : L_g = 2 : 1 -> ViT-S/B: 8 + 4, ViT-L: 16 + 8 | blocks 0-3 within-view, then alternating from block 4 with cross-view at odd indices (4 + 8) |
| Camera token | prepended to every view, takes part in all attention; E_c(f, q, t) or **one shared** learnable token (`paper-*`: 2 tokens, see below) | overwrites the cls token at block 4; **two** learnable tokens (reference view, other views) |
| E_c | "lightweight MLP" | MLP + LayerNorm + 4 transformer blocks across views + LayerNorm |
| D_C | "lightweight camera head ... transformer" on camera tokens (no depth given) | 2-layer MLP + linear heads (matches Tab. 8 parameter counts; we use it, see below) |
| Attention | vanilla DINOv2 | + QK-norm and 2D RoPE from block 4 |
| Reference view | not described | chosen from cls-token statistics ("saddle_balanced") for >= 3 views |
| Head input | backbone features | concatenation [last within-view output, current output] (2D channels) |
| Dual-DPT | shared reassemble, two fusion stacks, two output layers; ray map H/2 x W/2 x 6 | + token LayerNorm, UV sin-cos embeddings, ray map at 8 x patch grid (4/7 H), 5-conv ray neck + LayerNorm, **ray confidence** |
| Ray channels | [origin, direction] (Sec. 3.1, Eq. 1) | [direction, origin] |
| Ray -> pose | Eq. 1 mean origin; Eq. 2 DLT + RQ | confidence-weighted RANSAC homography + QL |

Details the paper leaves open, for which we use the release (or a stated choice):
- Tapped layers [5, 7, 9, 11] (S/B) and [11, 15, 19, 23] (L); head widths 64 / (48, 96, 192, 384) (S).
- Output activations: depth exp(.), confidence 1 + exp(.); ray linear.
- D_C: the 2-layer MLP above (the paper's Tab. 8 lists ~1M parameters for Small, which matches the
  release's MLP; the text calls it a transformer). Flagged as an open point.
- λ_c in L_D: **not given**. We use 0.2 (DUSt3R's ConfLoss default; our choice).
- `d, t` in L_P: ambiguous; we use the **predicted** ray map upsampled to depth resolution (D̂ ⊙ d̂ + t̂).
- L_C: ℓ1 on [t, q, fov] against ground truth (quaternions sign-standardized).
- Poses fed to E_c are expressed relative to view 0 and scaled by the median camera distance
  (the release does this at inference); ground truth for the losses is in view 0's frame, normalized
  by the mean ℓ2 norm of valid points (Sec. 3.3).
- Optimizer: AdamW (not stated for DA3; the paper uses AdamW for the metric model).

### Deliberate deviation: reference-view camera token (decided 2026-11-17)
With the paper's **single shared** camera token and no other per-view signal, the network is
permutation-equivariant over views. Pose targets are expressed in view 0's frame, so the model
cannot know which view is view 0. Overfitting one batch on CPU [sanity, synthetic data, 150 steps]:
camera loss 0.97 with 1 shared token, vs 0.065 with 2 tokens (reference + shared others, as
released), everything else identical. **Decision (project owner): the `paper-*` presets use two
learnable tokens, [reference view, shared by all other views].** This is the only intentional
departure from the paper text in `paper-*`.

## Decision: own implementation, official repo only as reference / oracle / harness
- **Ours (`src/da3`):** backbone, Dual-DPT, E_c, D_C, geometry (rays, Eq. 1, Eq. 2 DLT + RQ), losses,
  data pipeline, training loop, metrics (pose AUC, Umeyama + RANSAC, Chamfer / F1), exporters, Blender script.
- **Borrowed:** pretrained weights (DINOv2 for init; DA3 checkpoints for parity / [official] eval);
  unspecified hyperparameters listed above; the **benchmark harness** (dataset loaders, frame
  sampling, evo alignment, Open3D TSDF), imported from a pinned clone so the Sec. 6 protocol is
  identical, with our model plugged in through `src/da3/eval/bench_adapter.py`. Our own metric code
  matches the harness to round-off (`scripts/check_metrics_vs_official.py`).

## Compute plan (Kaggle only)
Since 2026-11-17 **every run happens on Kaggle** (project rule); the procedure is in
`docs/BASELINE_RUNBOOK.md`. Earlier CPU-only checks (unit tests, parity S/B, metric cross-check,
sanity runs, demo) are recorded in `experiments/local_cpu/` and are rerun on Kaggle by the notebooks.
Kaggle GPU (T4 / P100), main track: environment check, parity, and the benchmark with the released
weights (HiRoom / 7Scenes / DTU-64 first, then DTU / ScanNet++ / ETH3D one per session as `/tmp`
disk allows). Optional: reduced-scale training (ViT-S, 252-px class, 2-8 views, fp16 + GradScaler,
SDPA, gradient checkpointing, checkpoint every 500 steps, `--resume`, `--time-budget-h` to stop
before the session limit). Data lives in `/tmp`; outputs in `/kaggle/working`.
