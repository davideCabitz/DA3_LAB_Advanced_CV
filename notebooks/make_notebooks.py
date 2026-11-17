"""Generate the thin Kaggle notebooks (00_setup ... 04_demo_export) from the cell lists below.

All logic lives in src/ and scripts/; notebooks only clone the repo, install pinned deps, attach
inputs and call scripts. Re-run `python notebooks/make_notebooks.py` after editing this file.
"""

import json
from pathlib import Path

OFFICIAL_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"

SETUP = r'''# --- repo setup (identical in every notebook) ---------------------------------------------
REPO_URL = "https://github.com/davideCabitz/DA3_LAB_Advanced_CV.git"   # push your branch first
BRANCH = "main"
import os, subprocess, sys
os.chdir("/kaggle/working")
url = REPO_URL
try:  # private repo: add a Kaggle secret named GITHUB_TOKEN (Add-ons > Secrets)
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
    url = REPO_URL.replace("https://", f"https://{tok}@")
except Exception:
    pass
if not os.path.exists("DA3_LAB_Advanced_CV"):
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, url, "DA3_LAB_Advanced_CV"], check=True)
else:
    subprocess.run(["git", "-C", "DA3_LAB_Advanced_CV", "pull", "--ff-only"], check=True)
os.chdir("/kaggle/working/DA3_LAB_Advanced_CV")
!pip install -q -r requirements.txt && pip install -q -e . --no-deps
!git log -1 --format="commit %H  %cd"'''

OFFICIAL = f'''# --- official DA3 repo (reference / parity oracle / benchmark harness), pinned commit ----------
import os, subprocess
if not os.path.exists("/kaggle/working/Depth-Anything-3"):
    subprocess.run(["git", "clone", "https://github.com/ByteDance-Seed/Depth-Anything-3.git",
                    "/kaggle/working/Depth-Anything-3"], check=True)
subprocess.run(["git", "-C", "/kaggle/working/Depth-Anything-3", "checkout", "{OFFICIAL_COMMIT}"], check=True)
OFFICIAL_SRC = "/kaggle/working/Depth-Anything-3/src"
!pip install -q -r requirements-eval.txt'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


NOTEBOOKS = {
    "00_setup": [
        md("""# 00 · Setup and environment check (Phases 0, 1, 3)
Settings: **Accelerator = GPU (T4 x2 or P100)**, **Internet = on**. Records GPU / RAM / disk limits,
runs the unit tests, and checks the training pipeline on procedural scenes on the GPU
(fp16 overfit-one-batch, exact resume). Outputs in `/kaggle/working/`: `env_report.json`,
`pytest.txt`, `sanity/`, `resume_check.json`. Download them (or *Save Version*) and share."""),
        code(SETUP),
        code("!nvidia-smi\n!free -h\n!df -h /kaggle/working /tmp"),
        code("!python scripts/env_report.py --out /kaggle/working/env_report.json"),
        code("!python -m pytest -q 2>&1 | tee /kaggle/working/pytest.txt"),
        md("### Training pipeline on GPU: overfit one batch (synthetic scenes, ~2 min) and exact resume"),
        code('''!python -c "from da3.data.synthetic import make_scene; [make_scene(f'data/synthetic/scene{s:02d}', seed=s) for s in range(3)]"
!python scripts/train.py --config configs/train/sanity_synthetic.yaml --overfit-one-batch --out-dir /kaggle/working/sanity'''),
        code("!python scripts/resume_check.py --config configs/train/sanity_synthetic.yaml --steps 20 "
             "--work /tmp/resume_check --out /kaggle/working/resume_check.json"),
    ],
    "01_parity": [
        md("""# 01 · Parity with the released weights (Phase 2)
Loads DA3-SMALL / DA3-BASE (Apache-2.0) into our modules and into the official code, runs both in
fp32 on the same inputs, reports max abs differences (depth, confidence, ray, pose).
Setting: **[official]**."""),
        code(SETUP),
        code(OFFICIAL),
        code('''!python scripts/extract_frames.py /kaggle/working/Depth-Anything-3/assets/examples/robot_unitree.mp4 \\
    /kaggle/working/samples/robot_unitree --frames 6 --start 0.1 --end 0.5'''),
        code('''for preset in ["da3-small", "da3-base"]:
    !python scripts/parity_check.py --preset {preset} --official-src {OFFICIAL_SRC} \\
        --images /kaggle/working/samples/robot_unitree --views 6 \\
        --out /kaggle/working/experiments/parity/{preset}_gpu_robot6.json'''),
        code("!cat /kaggle/working/experiments/parity/*.json | head -80"),
    ],
    "02_train": [
        md("""# 02 · Reduced-scale training (Phase 3)
Setting: **[reduced-scale]**. Architecture = `paper-small` (paper as written, except a separate
reference-view camera token; see docs/PLAN.md), DINOv2 ViT-S init,
GT depth only (no teacher), views in [2, 8], constant token budget, warm-up + cosine.

Inputs to attach (Add data):
* TartanAir RGB-D (Kaggle dataset `pandrii000/dasvo-tartanair-rgb-d-validation-split`), set `TARTAN_IN`.
* To **resume**: attach the output of a previous run of this notebook and set `RESUME_FROM`.

Checkpoints (`last.pt`: model, optimizer, scheduler, scaler, step, RNG) are written every
`ckpt_every` steps to `/kaggle/working/runs/<exp>/ckpt/`. Use *Save Version > Save output* so the
next session can attach them."""),
        code(SETUP),
        code('''TARTAN_IN = "/kaggle/input/dasvo-tartanair-rgb-d-validation-split"   # adjust after attaching
RESUME_FROM = ""   # e.g. "/kaggle/input/<previous-output>/runs"
CANON_FROM = ""    # e.g. "/kaggle/input/<previous-output>/canonical": reuse prepared data, skip download/prep
CANON = "/tmp/canonical"     # the training config reads the data from here
if CANON_FROM:
    !rm -rf {CANON} && ln -s {CANON_FROM} {CANON} && ls {CANON}
else:
    !find {TARTAN_IN} -maxdepth 4 | head -40'''),
        md("### Data: TartanAir (Kaggle input) -> canonical format (skipped when CANON_FROM is set)"),
        code('''if not CANON_FROM:
    !python scripts/prepare_tartanair.py --raw {TARTAN_IN} --out {CANON}/tartanair --stride 2 --max-frames 300
    !python scripts/check_dataset.py {CANON}/tartanair --pairs 30
    !du -sh {CANON}/tartanair'''),
        md("""### Data: Virtual KITTI 2 subset (streamed from the official server; only the selected scenes are written)
License: CC BY-NC-SA 3.0 (academic use). The full tars (~16 GB) are streamed; only ~5 scenes are kept."""),
        code('''if not CANON_FROM:
    RAW = "/tmp/vkitti2_raw"; os.makedirs(RAW, exist_ok=True)
    URL = "https://download.europe.naverlabs.com/virtual_kitti_2.0.3"
    SCENES = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
    pat = " ".join(f"'*{s}/clone/frames/rgb/Camera_0/*'" for s in SCENES)
    !curl -sL {URL}/vkitti_2.0.3_textgt.tar.gz | tar -xz -C {RAW}
    !curl -sL {URL}/vkitti_2.0.3_rgb.tar | tar -x -C {RAW} --wildcards {pat}
    pat = pat.replace("/rgb/", "/depth/")
    !curl -sL {URL}/vkitti_2.0.3_depth.tar | tar -x -C {RAW} --wildcards {pat}
    scenes_arg = " ".join(SCENES)
    !python scripts/prepare_vkitti2.py --raw {RAW} --out {CANON}/vkitti2 --scenes {scenes_arg} --stride 2
    !python scripts/check_dataset.py {CANON}/vkitti2 --pairs 30
    !du -sh {CANON}/vkitti2 {RAW}
    # keep the prepared data in the notebook output so later sessions can attach it (CANON_FROM)
    !rm -rf {RAW} && cp -r {CANON} /kaggle/working/canonical && du -sh /kaggle/working/canonical'''),
        md("### Sanity: overfit one batch (total loss must drop by a large factor). Set RUN_OVERFIT = False on resumed sessions."),
        code('''RUN_OVERFIT = True
if RUN_OVERFIT:
    !python scripts/train.py --config configs/train/reduced_vits_tartanair_vkitti2.yaml --overfit-one-batch --steps 300'''),
        md("""### Real run (resumable)
`TIME_BUDGET_H`: the training stops cleanly (checkpoint + exit) after this many hours, so a
*Save & Run All* session never hits Kaggle's time limit with unsaved outputs. Keep it ~1.5 h below the
session limit to leave time for data preparation and saving."""),
        code('''TIME_BUDGET_H = 10.0
if RESUME_FROM:
    !mkdir -p /kaggle/working/runs && cp -rn {RESUME_FROM}/* /kaggle/working/runs/ && ls -R /kaggle/working/runs | head'''),
        code("!python scripts/train.py --config configs/train/reduced_vits_tartanair_vkitti2.yaml --resume --time-budget-h {TIME_BUDGET_H}"),
        code('''%load_ext tensorboard
%tensorboard --logdir /kaggle/working/runs'''),
        code("!cat experiments/registry.jsonl | tail -3\n!cp experiments/registry.jsonl /kaggle/working/registry.jsonl"),
    ],
    "03_eval": [
        md("""# 03 · Benchmark evaluation (Phase 4, Sec. 6 protocol)
Released benchmark harness (frame sampling seed 42, max 100 frames, evo Umeyama + RANSAC, Open3D
TSDF, F1 / Chamfer) with **our** model plugged in. Rows: **[official]** = released weights in our
modules; **[reduced-scale]** = our checkpoint (attach the 02_train output, set `CKPT`).
Data: HF `depth-anything/DA3-BENCH` (HiRoom 0.7 GB, DTU-64 1.8 GB, 7Scenes 3.6 GB, DTU 8.9 GB,
ScanNet++ 10.8 GB, ETH3D 15.1 GB). Limited runs are labelled by `--max-scenes`."""),
        code(SETUP),
        code(OFFICIAL),
        md("### Our metric code vs the released benchmark functions (should be ~0)"),
        code("!python scripts/check_metrics_vs_official.py --official-src {OFFICIAL_SRC} | tee /kaggle/working/metrics_crosscheck.json"),
        code('''from huggingface_hub import hf_hub_download
import zipfile
BENCH = "/tmp/da3bench"; os.makedirs(BENCH, exist_ok=True)
DATASETS = ["hiroom", "7scenes", "dtu64"]          # add "dtu", "scannetpp", "eth3d" if disk/time allow
for d in DATASETS:
    if not os.path.exists(f"{BENCH}/{d}"):
        z = hf_hub_download("depth-anything/DA3-BENCH", f"{d}.zip", repo_type="dataset", local_dir="/tmp/zips")
        zipfile.ZipFile(z).extractall(BENCH); os.remove(z)
!du -sh {BENCH}/*'''),
        code('''CKPT = ""   # e.g. "/kaggle/input/<02_train output>/runs/reduced_vits_tartanair_vkitti2/ckpt/last.pt"
MAX_SCENES = 0  # 0 = all scenes
for w, tag in [("official:da3-small", "official_da3-small"), ("official:da3-base", "official_da3-base")] + \\
              ([(CKPT, "reduced_vits")] if CKPT else []):
    ds = " ".join(DATASETS)
    t = f"{tag}_{'-'.join(DATASETS)}" + (f"_limited{MAX_SCENES}" if MAX_SCENES else "")
    !python scripts/eval_bench.py --official-src {OFFICIAL_SRC} --bench-root {BENCH} \\
        --work-root /tmp/eval --weights {w} --datasets {ds} \\
        --modes pose recon_unposed recon_posed --max-scenes {MAX_SCENES} --tag {t}'''),
        code("!python scripts/make_tables.py --out /kaggle/working/results_tables.md && cat /kaggle/working/results_tables.md"),
    ],
    "04_demo_export": [
        md("""# 04 · Inference demo + Blender export (Phase 5)
Feed-forward, no training. Two kinds of input (see docs/KAGGLE_EXAMPLES.md):
* **multi-view**: a folder of photos of ONE static scene -> cameras + fused point cloud + mesh,
  zipped with `blender_import.py` (download, unzip, Blender > Scripting > Run Script);
* **single images**: every photo processed on its own (N_v = 1) -> depth map + ray map per image
  (like paper Fig. 2).
Attach your photos as a Kaggle dataset and set `MY_MULTIVIEW` / `MY_SINGLE` below."""),
        code(SETUP),
        code(OFFICIAL),
        code('''# ---- your inputs (leave "" to skip) ---------------------------------------------------------
MY_MULTIVIEW = ""   # folder of one scene, OR a folder whose sub-folders are scenes, e.g. "/kaggle/input/my-photos/multiview"
MY_SINGLE = ""      # folder of independent images, e.g. "/kaggle/input/my-photos/single"
WEIGHTS = "official:da3-base"   # released weights (Apache-2.0); or a checkpoint path from 02_train
OUT = "/kaggle/working/demo"
from pathlib import Path
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
def scenes_in(folder):
    f = Path(folder)
    if any(p.suffix.lower() in IMG for p in f.iterdir()):
        return {f.name: str(f)}
    return {d.name: str(d) for d in sorted(f.iterdir()) if d.is_dir() and any(p.suffix.lower() in IMG for p in d.iterdir())}

!python scripts/extract_frames.py /kaggle/working/Depth-Anything-3/assets/examples/robot_unitree.mp4 \\
    /kaggle/working/samples/robot_unitree --frames 12 --start 0.0 --end 0.6
sets = {"SOH": "/kaggle/working/Depth-Anything-3/assets/examples/SOH",
        "robot_unitree": "/kaggle/working/samples/robot_unitree"}
if MY_MULTIVIEW:
    sets.update({f"my_{k}": v for k, v in scenes_in(MY_MULTIVIEW).items()})
print(sets)'''),
        md("### Multi-view reconstructions"),
        code('''for name, folder in sets.items():
    !python scripts/infer.py --images {folder} --out {OUT}/{name} --weights {WEIGHTS}
    !python scripts/render_preview.py {OUT}/{name}
!ls -la {OUT}'''),
        code('''from IPython.display import Image, display
for name in sets:
    print(name, "| columns: RGB, depth, ray map (direction), depth confidence")
    display(Image(f"{OUT}/{name}/preview.png", width=1000))
    display(Image(f"{OUT}/{name}/preview_3d.png", width=1000))'''),
        md("### Single images: depth map + ray map per image (N_v = 1)"),
        code('''if MY_SINGLE:
    !python scripts/infer.py --images {MY_SINGLE} --out {OUT}/single --weights {WEIGHTS} --per-image --mesh none
    for d in sorted(Path(f"{OUT}/single").iterdir()):
        if d.is_dir():
            print(d.name); display(Image(str(d / "preview.png"), width=1000))
else:
    print("set MY_SINGLE to run this part")'''),
        code("!ls -la /kaggle/working/demo/*.zip  # download these (Output panel on the right)"),
        md("""### Optional: headless Blender check of the export (needs Python 3.11 for the `bpy` wheel)
Imports each demo folder with `blender_import.py` and checks that points projected through the
Blender cameras land on the same pixels as the OpenCV projection (should be < 1 px)."""),
        code('''import sys
if sys.version_info[:2] == (3, 11):
    !pip install -q bpy
    for name in sets:
        !python scripts/blender_smoke_test.py /kaggle/working/demo/{name} /kaggle/working/demo/{name}/blender_smoke_test.json > /dev/null
        !cat /kaggle/working/demo/{name}/blender_smoke_test.json
else:
    print("skip: bpy wheels need Python 3.11, this kernel has", sys.version)'''),
    ],
}


def main():
    out = Path(__file__).parent
    for name, cells in NOTEBOOKS.items():
        nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                           "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 5}
        (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1))
        print("wrote", out / f"{name}.ipynb")


if __name__ == "__main__":
    main()
