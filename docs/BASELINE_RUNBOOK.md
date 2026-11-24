# Baseline runbook: from a new Kaggle account to the reproduced DA3 results

**Our baseline is the released DA3 weights** (DA3-SMALL and DA3-BASE, Apache-2.0; DA3-LARGE
optional). The reproduction has two parts:
1. show that our code is the released model (parity), and
2. run the paper's Sec. 6 benchmark with those weights and compare it with Tab. 2 / Tab. 3.

Enhancements are then measured against this baseline (see "After the baseline" at the end).
Training from scratch (`02_train`) is **optional**.

| # | Notebook | Main track? | What it does | GPU time (estimate) |
|---|---|---|---|---|
| 0 | `00_setup` | yes | environment limits, unit tests, GPU sanity checks | ~15 min |
| 1 | `01_parity` | yes | our code vs official code on the released weights | ~10 min |
| 3 | `03_eval` | yes | Sec. 6 benchmark with the released weights (pose AUC, F1 / Chamfer) | hours, depends on datasets |
| 4 | `04_demo_export` | optional | your photos -> 3D reconstruction / depth + ray maps -> Blender ([KAGGLE_EXAMPLES.md](KAGGLE_EXAMPLES.md)) | ~10 min |
| 2 | `02_train` | optional | from-scratch training at reduced scale (for architectural ablations) | 1+ sessions |

The times are estimates. The first real runs print the actual numbers.

---

## Step 0: one-time setup (~10 minutes)

### 0.1 Kaggle account
1. **kaggle.com > Register** (Google sign-in is quickest). It's free.
2. **Profile picture (top right) > Settings > Phone verification.** Without it, notebooks get
   **no GPU and no Internet**, and every notebook here needs both.

### 0.2 Code on GitHub
The notebooks `git clone` `github.com/davideCabitz/DA3_LAB_Advanced_CV` (branch `main`, set in
the first code cell) **at the start of every run**, and print the commit hash they used.
1. **Commit and push** your current state. Every Kaggle run uses whatever is pushed at that moment.
2. **Private repo?** Create a token that Kaggle can use:
   - GitHub > Settings > Developer settings > Personal access tokens > **Fine-grained tokens** >
     Generate new token.
   - Repository access: **only this repo**. Permissions: **Contents: Read-only**.
   - If the repo belongs to someone else, the owner may have to approve the token or add you as a
     collaborator.

### 0.3 Importing a notebook (do this once per notebook)
1. Kaggle > **Create (+) > New Notebook** > **File > Import Notebook**, then upload
   `notebooks/0X_*.ipynb`. Keep one Kaggle notebook per file and name it like the file.
2. Right panel > **Session options**:
   - **Accelerator: GPU T4 x2** (or P100). The code uses **one** GPU.
   - **Internet: On.** If the switch is greyed out, the phone verification is missing.
3. Private repo only: **Add-ons > Secrets > Add secret**. Name it `GITHUB_TOKEN`, paste the token,
   and **tick it for this notebook**.

### 0.4 Running a notebook
- **Run All (interactive):** you watch the cells run. The outputs in `/kaggle/working` are **lost
  when the session ends**, unless you download them from the *Output* panel first.
- **Save Version > Save & Run All (Commit):** runs in the background (you can close the browser) and
  **keeps `/kaggle/working`** under the version's *Output* tab. Use this for every result you want
  to keep.
- **Using a previous run's files:** *Add Input > Your Work / Notebook Output Files*, pick the
  notebook version, then use **Copy path** in the Input panel.

### 0.5 GPU quota
Kaggle gives a **weekly GPU allowance**, shown next to the accelerator setting. It counts **while a
GPU session is open, even if nothing runs**: stop the session (power icon) when you're done. One
session stops after about 12 h.

---

## Step 1: `00_setup` (environment + sanity)

**Run All** (or *Save & Run All*). Pass criteria:
- `env_report.json`: a GPU is listed (T4 ~15 GB or P100 16 GB), `sdpa_fp16_ok: true`, and the
  `internet` entry has an HTTP 200 status line. **Note the free disk space of `/tmp`**; it decides
  which benchmark datasets fit in one session.
- `pytest.txt` ends with `87 passed` and no failures (the count grows when tests are added; it
  was 87 on 2026-11-22, with the enhancement tests).
- The overfit cell: the total loss drops from ~8 to well below 1 within 150 steps.
- `resume_check.json`: `"identical": true`.

**Send me:** `env_report.json`, `pytest.txt`, `resume_check.json`, and the overfit log.

## Step 2: `01_parity` (our code = released model)

**Save & Run All.** It loads DA3-SMALL and DA3-BASE into our modules and into the official code,
then compares their outputs in fp32 on the GPU.

Pass criteria for every mode in `experiments/parity/*.json`: `depth_max_rel < 1e-3`,
`ray_max_abs < 1e-3`, `pose_w2c_max_abs < 1e-3`. On CPU we measured ~1e-5; on the GPU up to ~1e-4
is normal. Anything larger means the environment changed: stop and send me the file.

**Send me:** `/kaggle/working/experiments/parity/*.json`.

## Step 3: `03_eval` (reproduce Tab. 2 / Tab. 3 with the released weights)

The released benchmark harness runs with our code plugged in, so the protocol is identical to the
paper: up to 100 frames per scene with seed 42, evo + RANSAC alignment, TSDF fusion, F1 / Chamfer.
Leave `CKPT = ""`; that variable is only for a model we trained ourselves.

1. **Timing run.** Set `DATASETS = ["hiroom"]` and `MAX_SCENES = 2`, then **Save & Run All**.
   - The metric cross-check cell must print differences around 0.
   - Note the run time. Results are tagged `_limited2`, which marks them as not reportable.
2. **Main run.** Set `MAX_SCENES = 0` (all scenes) and `DATASETS = ["hiroom", "7scenes", "dtu64"]`,
   then **Save & Run All**.
3. **Remaining datasets, one or two per session**, as long as the free `/tmp` disk allows:
   `["dtu"]` (8.9 GB zip), `["scannetpp"]` (10.8 GB), `["eth3d"]` (15.1 GB). Each session writes
   `experiments/eval/<model>_<datasets>.json`, so sessions never overwrite each other.
4. `results_tables.md` shows (a) the paper's numbers next to (b) the released weights run by us.

**Send me:** `results_tables.md`, `metrics_crosscheck.json`, and
`/kaggle/working/DA3_LAB_Advanced_CV/experiments/eval/*.json`.

What "reproduced" means here: our DA3-SMALL / DA3-BASE numbers should be close to the paper's
DA3-Small / DA3-Base rows. Differences are expected to be small: different GPU, fp16 vs bf16,
library versions. Clear gaps are findings to explain, and we look into them together.
DA3-LARGE: the available checkpoint is the retrained `-1.1` (not the one behind the paper's
numbers), its README license is CC BY-NC, and we haven't parity-checked it yet. Treat it as an
extra, not as part of the baseline.

## Step 4: report
With those files I fill in `reports/baseline.md`. You commit.

---

## After the baseline: testing an enhancement

Every enhancement is compared **against the released weights, with exactly one change**. Same
benchmark, datasets, scenes and evaluation settings. Routes, from cheapest and most comparable:

1. **Training-free** (the change acts after the network runs: pose from the ray map, view
   selection, depth fusion, ...): evaluate released weights with vs without the change in
   `03_eval`.
2. **Fine-tuning the released weights** (loss changes, domain adaptation): fine-tune *with* and
   *without* the change, using the same data, steps and seed (LoRA or full fine-tuning; LoRA support is
   planned). Use 2-3 seeds.
3. **New head on a frozen backbone:** train the head, compare with the released head.
4. **From scratch at reduced scale (`02_train`):** only for backbone-architecture ideas. Run several
   seeds and two model sizes, and present it as a scaled-down ablation (like the paper's Sec. 7.2).

Fine-tuning data must never contain benchmark scenes (HiRoom, ETH3D, DTU, 7Scenes, ScanNet++
test scenes).

## Optional: `02_train` (from-scratch training at reduced scale)
Only needed for route 4. The procedure is the same as before:
- attach the TartanAir dataset and check the folder listing printed by the first cells;
- data preparation, then the overfit check, then training with `--time-budget-h` (a cap that stops
  cleanly before Kaggle's session limit; it is not the expected duration);
- resume in later sessions with `RESUME_FROM`, `CANON_FROM` and `RUN_OVERFIT = False`.

Pass criteria: `check_dataset.py` `median_rel_err` below ~0.02 for each dataset, and finite,
decreasing losses. Decide `train.steps` from the measured seconds per step **before** the second
session.

---

## What NOT to do (breaks things or makes the numbers meaningless)

Environment:
- ❌ **Don't `pip install -e` the official Depth-Anything-3 repo.** It pins `numpy<2` and pulls
  `xformers`, which breaks the Kaggle image. The notebooks only put it on `sys.path`.
- ❌ **Don't ignore the Open3D self-test** (`open3d_check.json`, printed after the install cell). If
  `legacy_tsdf_ok` is `false`, TSDF fusion is broken in that environment and reconstruction F1 would be
  wrong. `03_eval` stops automatically; send me the file.
- ❌ **Don't reinstall or upgrade `torch`.** If a Kaggle image update breaks something, pin the
  notebook environment (Session options > Environment) to the date of a run that worked.
- ❌ **Don't put large data in `/kaggle/working`** (20 GB output limit). Benchmark zips and raw
  data go to `/tmp`, which is cleared after the session.
- ❌ **Don't leave a GPU session open and idle.** It still uses your weekly quota.

Code and git:
- ❌ **Don't push code changes while a series of evaluation or training sessions is running.** Every
  session clones the latest commit, so you'd mix two versions of the code. Check the printed commit
  hash.
- ❌ **Don't use the `da3-*` presets for training from scratch**, and don't change the `paper-*`
  presets or the baseline configs to "improve" results. Enhancements get **new** config files with
  a **new experiment name**.

Evaluation:
- ❌ **Don't change the protocol** when comparing to the paper or to the baseline: keep
  `max_frames 100`, reference view `first`, the default modes, and the released harness.
- ❌ **Don't report `_limited` runs or unfinished sessions** as results. Only the
  `experiments/eval/<tag>.json` written at the end of a complete run counts.
- ❌ **Don't compare across settings:** official weights vs paper numbers is the reproduction;
  enhancement vs released weights (same protocol) is the improvement. A reduced-scale model is only
  compared with another reduced-scale model.
- ❌ **Don't fine-tune on benchmark data.**
- ❌ **Don't delete or rewrite** `experiments/registry.jsonl` lines (the record of all runs).
