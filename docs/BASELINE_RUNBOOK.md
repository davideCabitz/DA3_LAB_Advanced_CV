# Baseline runbook: every step to run the DA3 baseline on Kaggle

This runbook takes you from a fresh Kaggle account to the three result groups of the report:
(a) paper numbers, (b) released weights evaluated by us, (c) our reduced-scale model. Run the
notebooks in order. After each one, check the **pass criteria** before moving on, and save the
listed files (send them to me, or put them in `experiments/`).

| # | Notebook | What it does | Typical GPU time |
|---|---|---|---|
| 0 | `00_setup` | environment limits, unit tests, GPU sanity (overfit + exact resume) | ~15 min |
| 1 | `01_parity` | our modules vs official code on the released weights | ~10 min |
| 2 | `02_train` | data preparation + reduced-scale training (several sessions) | 10 h per session |
| 3 | `03_eval` | Sec. 6 benchmark: pose AUC, reconstruction F1 / Chamfer | hours (depends on datasets) |
| 4 | `04_demo_export` | optional demo / Blender package (docs/KAGGLE_EXAMPLES.md) | ~10 min |

Times are estimates until the first real runs, which print the actual timings.

---

## Step 0: one-time setup

1. **Kaggle account, phone-verified** (Settings > Phone verification). Without it you get no GPU and
   no Internet in notebooks.
2. **Push the repo to GitHub** (you commit and push yourself). Every notebook `git clone`s the branch
   in its first code cell (`BRANCH = "main"`) at run time, and prints the commit hash it used.
   **Every Kaggle run uses whatever is pushed at that moment.**
3. **Private repo?** Create a GitHub token with read access to the repo, then in each notebook go to
   **Add-ons > Secrets > Add secret**, name it `GITHUB_TOKEN`, and tick it for the notebook.
4. **Import the 5 notebooks.** Kaggle > Code > New Notebook > File > Import Notebook. Upload
   `notebooks/0X_*.ipynb` (or paste the GitHub URL of each file). Keep one Kaggle notebook per file.
5. **Settings for every notebook** (right panel > Session options):
   - Accelerator: **GPU T4 x2** or **P100**. The code uses **one** GPU.
   - Internet: **On**.
   - Language Python, environment "Latest" (or pin it once it works, see "What NOT to do").
6. **Two ways to run** (both work for every notebook):
   - *Interactive* (Run All in the editor): good for checking things, but the outputs in
     `/kaggle/working` are lost when the session stops.
   - ***Save Version > Save & Run All (Commit)***: runs headless (you can close the browser) and
     **keeps `/kaggle/working` as the version's Output**. Use it for every real run.
7. **Attaching the output of a previous run**: open the notebook editor > **Add Input** >
   *Notebook Output Files* (or *Your Work*) > pick the notebook and version. It appears under
   `/kaggle/input/...`. Use **Copy path** in the Input panel to get the exact path.

## Step 1: `00_setup` (environment + sanity)

Run it with **Save & Run All**.

Pass criteria (Output tab > `/kaggle/working`):
- `env_report.json` lists a GPU (T4 ~15 GB or P100 16 GB), `sdpa_fp16_ok: true`, and an HTTP 200
  status line under `internet`. Note the RAM and the free disk on `/kaggle/working` and `/tmp`.
- `pytest.txt` ends with `25 passed`.
- The overfit log (cell output) shows the total loss dropping from ~8 to well below 1 within
  150 steps, with the camera term going down too.
- `resume_check.json` has `"identical": true`.

Send me: `env_report.json`, `pytest.txt`, `resume_check.json`, and the overfit log (copy from the
cell output or `sanity/`).

## Step 2: `01_parity` (released weights in our code)

Run it with **Save & Run All**. It compares DA3-SMALL and DA3-BASE (Apache-2.0) in our modules
against the official code, in fp32 on the GPU.

Pass criteria for every mode in `/kaggle/working/experiments/parity/*.json`:
`depth_max_rel < 1e-3`, `ray_max_abs < 1e-3`, `pose_w2c_max_abs < 1e-3`.
On CPU we measured ~1e-5; the GPU may be up to ~1e-4. Larger values mean the environment changed;
stop and send me the file.

## Step 3: `02_train` (reduced-scale baseline training)

### 3a. First session (data preparation + start of training)
1. **Add Input**: the Kaggle dataset
   [DASVO TartanAir RGB-D validation split](https://www.kaggle.com/datasets/pandrii000/dasvo-tartanair-rgb-d-validation-split).
   Copy its path into `TARTAN_IN` (third code cell).
2. Leave `RESUME_FROM = ""` and `CANON_FROM = ""`.
3. **First time only: run the first three code cells interactively** and look at the folder listing
   printed for `TARTAN_IN`. The converter expects TartanAir's native layout (`image_left/`,
   `depth_left/`, `pose_left.txt`). If the listing looks different, **stop and send me the
   listing**; I will adapt `scripts/prepare_tartanair.py`.
4. If the layout is fine: **Save Version > Save & Run All**. The run does the following:
   - converts TartanAir and checks it with `check_dataset.py`;
   - streams vKITTI2 (~16 GB download, only 5 scenes kept) and checks it;
   - copies the prepared data to `/kaggle/working/canonical` so later sessions can reuse it;
   - runs the overfit sanity check (300 steps);
   - trains with `--time-budget-h 10`. It stops cleanly before Kaggle's limit and saves
     `runs/reduced_vits_tartanair_vkitti2/ckpt/last.pt`.

Pass criteria (cell outputs of the committed version):
- `check_dataset.py` prints `median_rel_err` **below ~0.02** for both datasets. A value around 0.1
  or higher means a camera/pose convention problem: **stop and send me the output**. Don't train on
  it.
- The overfit total loss drops by a large factor (at least 3x) within 300 steps.
- Training logs (`step N | ... | tok/s | mem`): loss terms are finite and decreasing over the first
  thousand steps, and `mem` stays below the GPU memory.

### 3b. How many steps fit? (decide once, after the first session)
The config uses `train.steps: 20000`. From the first session's log, compute `seconds per step` =
(time between two log lines) / `log_every`. Then the total is
`sessions needed = steps x seconds_per_step / (TIME_BUDGET_H x 3600)`.
If that is more than you want to spend (weekly GPU quota), **lower `train.steps` before the second
session and restart from scratch**. Changing `steps` in the middle of a run changes the cosine
schedule.

### 3c. Next sessions (resume)
1. Open the `02_train` notebook > **Add Input** > output of the previous version.
2. Set (with **Copy path**):
   ```python
   RESUME_FROM = "/kaggle/input/<previous-output>/runs"
   CANON_FROM  = "/kaggle/input/<previous-output>/canonical"
   RUN_OVERFIT = False
   ```
3. **Save & Run All**. The log must say `resumed from ... at step N`, with N equal to the last saved
   step. If it says nothing about resuming, the paths are wrong; stop the run.
4. Repeat until the last log line says `done: {... "finished": true}`.

Send me: the full training log of each session (or the `runs/<exp>/tb` folder), `train_summary.json`,
`experiments/registry.jsonl` (in the repo folder of the output), and `/kaggle/working/registry.jsonl`.

## Step 4: `03_eval` (Sec. 6 benchmark)

1. **Add Input**: the output of the **last** `02_train` version, and set
   `CKPT = "/kaggle/input/<train-output>/runs/reduced_vits_tartanair_vkitti2/ckpt/last.pt"`.
2. **First run: time it.** Set `MAX_SCENES = 2` and `DATASETS = ["hiroom"]`, then *Save & Run All*.
   It evaluates the released DA3-SMALL and DA3-BASE plus your checkpoint on 2 scenes and prints the
   times. The cross-check cell must print diffs around 0 (`check_metrics_vs_official.py`).
3. **Full run**: `MAX_SCENES = 0` (all scenes), with
   `DATASETS = ["hiroom", "7scenes", "dtu64"]` first. Add `"dtu"`, `"scannetpp"`, `"eth3d"` in later
   sessions if disk and time allow (zips of 8.9 / 10.8 / 15.1 GB; check the free `/tmp` space in
   `env_report.json`). If a session would be too long, split the datasets over several sessions.
   Each session writes `experiments/eval/<model>_<datasets>[_limitedN].json`. The tag is built
   automatically from the dataset list and `MAX_SCENES`, so different dataset groups don't
   overwrite each other.
4. `results_tables.md` (in `/kaggle/working`) has the rows (a) paper, (b) official run by us,
   (c) reduced-scale. A row computed on a subset is marked `(limited)`.

Send me: `results_tables.md`, `/kaggle/working/DA3_LAB_Advanced_CV/experiments/eval/*.json`, and
`metrics_crosscheck.json`.

## Step 5: report
With those files I fill in `reports/baseline.md` (tables, compute used, failure cases). You commit.

---

## What NOT to do (breaks the baseline or makes the numbers meaningless)

Code and configuration:
- ❌ **Don't train with a `da3-*` preset**, and don't initialise training from the official DA3
  weights. The baseline is `paper-small`, initialised from DINOv2 (`init: {type: dinov2}`).
- ❌ **Don't change the baseline config** (`configs/train/reduced_vits_tartanair_vkitti2.yaml`) or the
  `paper-*` presets in order to "improve" results. Enhancements go into **new** config files with a
  **new `experiment:` name**, so the baseline stays comparable.
- ❌ **Don't edit anything in the config between sessions of the same run** (seed, resolutions,
  views, token budget, LR, steps). The data order and the LR schedule are functions of
  (seed, step). The only exception is lowering `steps` *before* the 2nd session, as in 3b, and then
  restarting from scratch.
- ❌ **Don't rename `experiment:`** during a run. The checkpoint path depends on it, so a renamed run
  silently restarts from step 0.
- ❌ **Don't push code changes in the middle of a training run.** Each session clones the latest
  commit, so a changed model or loss makes the resumed run a mix of two codes. Check that the
  printed commit hash is the same in every session.
- ❌ **Don't switch to bf16** or remove `amp: fp16`. T4 and P100 don't support bf16. Don't remove
  gradient checkpointing to "go faster"; you will run out of memory.
- ❌ **Don't use a resolution that isn't a multiple of 14** (ViT patch size).
- ❌ **Don't run two training sessions at once with the same output.** They overwrite each other's
  checkpoints.

Environment:
- ❌ **Don't `pip install -e` the official Depth-Anything-3 repo** (it pins `numpy<2` and pulls
  `xformers`, which breaks the Kaggle image). The notebooks put it on `sys.path` only.
- ❌ **Don't change the pinned `open3d==0.19.0`.** With 0.20.0 the TSDF meshes come out empty,
  so reconstruction F1 becomes 0.
- ❌ **Don't reinstall or upgrade `torch`.** If the Kaggle image changes and something breaks, pin the
  notebook's environment (Session options > Environment) to the date of a run that worked.
- ❌ **Don't store datasets in `/kaggle/working`** except the prepared `canonical/` data (small). The
  output limit is 20 GB. Raw data and benchmark zips belong in `/tmp` (lost after the session, which
  is fine).

Evaluation:
- ❌ **Don't change the protocol** when comparing to the paper: keep `max_frames 100`, reference
  view `first`, the default modes, and the released harness. Results on fewer scenes must stay
  marked "limited".
- ❌ **Don't compare reduced-scale numbers to the paper as if they were a failed reproduction.** The
  paper trains ViT-G/L for 200k steps on 128 H100s with teacher labels; ours is ViT-S on one T4.
  Compare reduced-scale vs reduced-scale, and the official weights vs the paper.
- ❌ **Don't report numbers you didn't get from a finished run.** Interrupted evaluation sessions
  leave partial folders; only the `experiments/eval/<tag>.json` written at the end counts.
- ❌ **Don't delete or overwrite** `experiments/registry.jsonl` lines. That file is the record of
  every run (config, seed, commit, metrics).
