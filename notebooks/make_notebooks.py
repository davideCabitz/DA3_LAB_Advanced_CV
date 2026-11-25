"""Generate the thin Kaggle notebooks (00_setup ... 04_demo_export, 05_self_audit, 06_foveated, 07_lora,
09_compare_all) from the cell lists below.

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
sys.path.insert(0, os.path.abspath("src"))   # the editable install is only seen by new processes, not this kernel
!git log -1 --format="commit %H  %cd"'''

OFFICIAL = f'''# --- official DA3 repo (reference / parity oracle / benchmark harness), pinned commit ----------
import os, subprocess
if not os.path.exists("/kaggle/working/Depth-Anything-3"):
    subprocess.run(["git", "clone", "https://github.com/ByteDance-Seed/Depth-Anything-3.git",
                    "/kaggle/working/Depth-Anything-3"], check=True)
subprocess.run(["git", "-C", "/kaggle/working/Depth-Anything-3", "checkout", "{OFFICIAL_COMMIT}"], check=True)
OFFICIAL_SRC = "/kaggle/working/Depth-Anything-3/src"
!pip install -q -r requirements-eval.txt
# Open3D: requirements-eval installs the newest one; 0.20.0's TSDF returns empty meshes (isl-org/Open3D#7592),
# so use the verified 0.19.0 where it exists (Python <= 3.12)
import sys
if sys.version_info[:2] <= (3, 12):
    !pip install -q "open3d==0.19.0"
# Open3D self-test: does TSDF fusion (depth maps -> mesh) work in this environment?
!python scripts/check_open3d.py --out /kaggle/working/open3d_check.json'''


COLAB_SETUP = r'''# --- repo setup (Google Colab) -----------------------------------------------------------------
REPO_URL = "https://github.com/davideCabitz/DA3_LAB_Advanced_CV.git"   # push your branch first
BRANCH = "main"
import os, subprocess
os.environ["GIT_TERMINAL_PROMPT"] = "0"   # fail fast instead of waiting for a password
os.chdir("/content")
url = REPO_URL
try:  # private repo: left bar > key icon (Secrets) > add GITHUB_TOKEN, switch on "Notebook access"
    from google.colab import userdata
    tok = userdata.get("GITHUB_TOKEN")
    if tok:
        url = REPO_URL.replace("https://", f"https://{tok}@")
except Exception:
    print("no GITHUB_TOKEN secret: cloning without a token (works only for a public repo)")
if not os.path.exists("DA3_LAB_Advanced_CV"):
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, url, "DA3_LAB_Advanced_CV"], check=True)
else:
    subprocess.run(["git", "-C", "DA3_LAB_Advanced_CV", "pull", "--ff-only"], check=True)
os.chdir("/content/DA3_LAB_Advanced_CV")
!pip install -q -r requirements.txt && pip install -q -e . --no-deps
!git log -1 --format="commit %H  %cd"'''

COLAB_OPEN3D = r'''# Open3D (TSDF meshing) + self-test: if TSDF fusion is broken here, infer.py uses a simpler mesher.
# 0.19.0 is the version whose legacy TSDF works (0.20.0 returns empty meshes); it exists up to Python 3.12.
import sys
if sys.version_info[:2] <= (3, 12):
    !pip install -q "open3d==0.19.0"
else:
    !pip install -q "open3d>=0.19.0"
!python scripts/check_open3d.py --out /content/open3d_check.json'''

COLAB_OFFICIAL = f'''# --- official DA3 repo (benchmark harness), pinned commit ---------------------------------------
import os, subprocess
if not os.path.exists("/content/Depth-Anything-3"):
    subprocess.run(["git", "clone", "https://github.com/ByteDance-Seed/Depth-Anything-3.git",
                    "/content/Depth-Anything-3"], check=True)
subprocess.run(["git", "-C", "/content/Depth-Anything-3", "checkout", "{OFFICIAL_COMMIT}"], check=True)
OFFICIAL_SRC = "/content/Depth-Anything-3/src"
!pip install -q -r requirements-eval.txt
# Open3D: requirements-eval installs the newest one; 0.20.0's legacy TSDF returns empty meshes, so use the
# verified 0.19.0 where it exists (Python <= 3.12)
import sys
if sys.version_info[:2] <= (3, 12):
    !pip install -q "open3d==0.19.0"
# Open3D self-test: does TSDF fusion (depth maps -> mesh) work in this environment?
!python scripts/check_open3d.py --out /content/open3d_check.json'''

RECON_CHECK = '''import json
chk = json.load(open("__PATH__"))
RECON = bool(chk["legacy_tsdf_ok"])   # reconstruction scores (F1 / Chamfer) need a working Open3D TSDF
if RECON:
    print("Open3D TSDF OK:", chk["open3d"], "-> reconstruction scores (F1 / Chamfer) included")
else:
    print("Open3D", chk["open3d"], "on Python", chk["python"], "has a broken TSDF fusion (known Open3D 0.20 bug, "
          "github.com/isl-org/Open3D/issues/7592; the working 0.19.0 only exists up to Python 3.12).")
    print("This run SKIPS the reconstruction scores (F1 / Chamfer). Pose accuracy (AUC), failure detection "
          "and depth uncertainty are not affected.")
MODES = "pose recon_unposed" if RECON else "pose"
RECON_VARIANTS = "" if RECON else "--recon-variants"   # an empty list = pose-only harness replays'''

SCORING_ENV = r'''# --- reconstruction scores: Python 3.12 side environment (only when the Open3D above is broken) --------
# Kaggle / Colab run Python 3.13, whose only Open3D (0.20.0) has a broken TSDF fusion (isl-org/Open3D#7592),
# and the working 0.19.0 needs Python <= 3.12. The scoring (CPU only: TSDF fusion, F1 / Chamfer, pose AUC)
# then runs in a small Python 3.12 environment; DA3 itself still runs on the GPU in this notebook's Python.
import json, os
PY312 = "/tmp/py312/bin/python"
SPLIT = False   # True = model in this Python (--infer-only), scores in PY312 (--eval-only)
if not RECON:
    !pip install -q uv
    !uv python install 3.12
    if not os.path.exists(PY312):
        !uv venv /tmp/py312 --python 3.12
    !uv pip install -q --python {PY312} torch torchvision --index-url https://download.pytorch.org/whl/cpu
    !uv pip install -q --python {PY312} numpy scipy matplotlib "open3d==0.19.0" "evo==1.38.1" addict scikit-learn tqdm omegaconf PyYAML trimesh plyfile einops safetensors huggingface_hub opencv-python-headless pillow imageio "moviepy<2"
    !uv pip install -q --python {PY312} --no-deps -e .
    !{PY312} scripts/check_open3d.py --out /tmp/open3d_check_py312.json
    try:
        SPLIT = bool(json.load(open("/tmp/open3d_check_py312.json"))["legacy_tsdf_ok"])
    except Exception as e:
        print("Python 3.12 scoring environment not usable:", repr(e))
    if SPLIT:
        MODES, RECON_VARIANTS = "__FULL_MODES__", ""
        print("OK: the reconstruction scores (F1 / Chamfer) are computed with Python 3.12 + Open3D 0.19.0.")
    else:
        print("The Python 3.12 scoring environment does not work either: this run stays POSE ONLY. "
              "Send the output of this cell.")
SCORER = PY312 if SPLIT else "python"   # the Python that scores (and runs the analyses)'''

_GPU_SMOKE = '''    !nvidia-smi -L
    _x = torch.randn(1, 4, 64, 32, device="cuda", dtype=torch.float16)   # fails here at once if this torch build
    torch.nn.functional.scaled_dot_product_attention(_x, _x, _x)         # does not support the GPU (not after the downloads)
    print("GPU OK:", torch.cuda.get_device_name(0), "| compute capability", torch.cuda.get_device_capability(0))'''

COLAB_GPU = '''import torch
if torch.cuda.is_available():
''' + _GPU_SMOKE + '''
else:
    print("No GPU in this runtime. Fine for analysis-only sessions (CPU work); "
          "for running the model choose a GPU: Runtime > Change runtime type.")'''

COLAB_GPU_REQUIRED = '''import torch
assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > T4 (or L4, A100, G4) GPU, then run all again."
if True:
''' + _GPU_SMOKE

# per-notebook metadata (Colab: open with a T4 GPU runtime)
_COLAB_T4 = {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []}}
METADATA = {"04_demo_export_colab": _COLAB_T4, "05_self_audit_colab": _COLAB_T4}


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
        md("""# 02 · OPTIONAL: from-scratch reduced-scale training
Not part of the baseline (the baseline is the released DA3 weights, see 01 / 03). Use it only for
backbone-architecture ablations. Setting: **[reduced-scale]**. Architecture = `paper-small` (paper as written, except a separate
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
        md("""# 03 · Baseline reproduction: Sec. 6 benchmark with the released DA3 weights
Released benchmark harness (frame sampling seed 42, max 100 frames, evo Umeyama + RANSAC, Open3D
TSDF, F1 / Chamfer) with **our** model plugged in. Rows: **[official]** = released weights in our
modules; **[reduced-scale]** = our checkpoint (attach the 02_train output, set `CKPT`).
Data: HF `depth-anything/DA3-BENCH` (HiRoom 0.7 GB, DTU-64 1.8 GB, 7Scenes 3.6 GB, DTU 8.9 GB,
ScanNet++ 10.8 GB, ETH3D 15.1 GB). Limited runs are labelled by `--max-scenes`."""),
        code(SETUP),
        code(OFFICIAL),
        code(RECON_CHECK.replace("__PATH__", "/kaggle/working/open3d_check.json")
             + '\nMODES = "pose recon_unposed recon_posed" if RECON else "pose"   # this notebook also scores recon_posed'),
        code(SCORING_ENV.replace("__FULL_MODES__", "pose recon_unposed recon_posed")),
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
    if SPLIT:   # the model here (GPU), the scores in the Python 3.12 environment (CPU)
        !python scripts/eval_bench.py --official-src {OFFICIAL_SRC} --bench-root {BENCH} \\
            --work-root /tmp/eval --weights {w} --datasets {ds} \\
            --modes {MODES} --max-scenes {MAX_SCENES} --tag {t} --infer-only
        !{PY312} scripts/eval_bench.py --official-src {OFFICIAL_SRC} --bench-root {BENCH} \\
            --work-root /tmp/eval --weights {w} --datasets {ds} \\
            --modes {MODES} --max-scenes {MAX_SCENES} --tag {t} --eval-only
    else:
        !python scripts/eval_bench.py --official-src {OFFICIAL_SRC} --bench-root {BENCH} \\
            --work-root /tmp/eval --weights {w} --datasets {ds} \\
            --modes {MODES} --max-scenes {MAX_SCENES} --tag {t}'''),
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
    print(name, "| columns: RGB | depth (red = near, blue = far) | ray map (direction as RGB) | depth confidence (red = high, blue = low)")
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
    "04_demo_export_colab": [
        md("""# 04 (Colab) · Multi-view reconstruction of your own photos + Blender export
Same as `04_demo_export`, for **Google Colab** (no Kaggle queue). Original DA3, released weights,
feed-forward, no training. See docs/KAGGLE_EXAMPLES.md §8.

**Before running**
1. **Runtime → Change runtime type**: the first GPU that connects, cheapest first,
   **T4 → L4 → A100 → G4** (if one does not connect within a few minutes, pick the next; all give
   the same result). Not the TPUs (the code needs CUDA).
   When the zips are saved, **Runtime → Disconnect and delete runtime** (a connected runtime keeps
   using units).
2. Private repo only: left bar → **key icon (Secrets)** → add `GITHUB_TOKEN` (the same token as on
   Kaggle) and switch on **Notebook access**.
3. Your photos, one of the two:
   * **Google Drive** (recommended): put the folder (e.g. `building/` with the JPGs) in your Drive,
     e.g. `MyDrive/DA3/building`, and set `MY_MULTIVIEW` below;
   * **upload a .zip** of the folder from your computer (`PHOTOS_FROM = "upload"`).
4. **Runtime → Run all**.

Results: `/content/demo/<name>/` and `<name>.zip`. Colab deletes `/content` when the session ends,
so the last cell copies the zips to Drive (`SAVE_TO_DRIVE`) or downloads them."""),
        code(COLAB_GPU_REQUIRED),
        code(COLAB_SETUP),
        code(COLAB_OPEN3D),
        md("### Your inputs"),
        code('''PHOTOS_FROM = "drive"   # "drive": a folder in your Google Drive | "upload": a .zip from your computer
MY_MULTIVIEW = "/content/drive/MyDrive/DA3/building"   # ONE scene (photos directly inside) or a folder of scene sub-folders
MY_SINGLE = ""          # optional: folder of independent photos -> depth + ray map per photo
WEIGHTS = "official:da3-base"            # released weights (Apache-2.0); "official:da3-small" is faster
EXTRA_ARGS = "--conf-percentile 60"      # buildings: fewer sky / low-confidence points; add "--max-views 60" if CUDA out of memory
SAVE_TO_DRIVE = "/content/drive/MyDrive/DA3/results"   # "" = download the zips instead
RUN_EXAMPLES = False    # True: also reconstruct the two built-in examples (SOH, robot_unitree)
OUT = "/content/demo"

if PHOTOS_FROM == "drive" or SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
if PHOTOS_FROM == "upload":
    import io, zipfile
    from google.colab import files
    uploaded = files.upload()   # choose the .zip of your photo folder
    for fname, data in uploaded.items():
        zipfile.ZipFile(io.BytesIO(data)).extractall("/content/my_photos")
    MY_MULTIVIEW = "/content/my_photos"

from pathlib import Path
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
def has_images(d):
    return any(p.suffix.lower() in IMG and not p.name.startswith(".") for p in Path(d).iterdir())
def scenes_in(folder):
    f = Path(folder)
    assert f.exists(), f"not found: {folder} (check the path in the Files panel on the left)"
    if has_images(f):
        return {f.name: str(f)}
    subs = [d for d in sorted(f.iterdir()) if d.is_dir() and not d.name.startswith(("__MACOSX", "."))]
    found = {d.name: str(d) for d in subs if has_images(d)}
    if not found:  # e.g. a zip containing one folder that itself contains the scene folders
        for d in subs:
            found.update({e.name: str(e) for e in sorted(d.iterdir()) if e.is_dir() and has_images(e)})
    return found

sets = {}
if RUN_EXAMPLES:
    if not os.path.exists("/content/Depth-Anything-3"):
        subprocess.run(["git", "clone", "https://github.com/ByteDance-Seed/Depth-Anything-3.git", "/content/Depth-Anything-3"], check=True)
    !python scripts/extract_frames.py /content/Depth-Anything-3/assets/examples/robot_unitree.mp4 \\
        /content/samples/robot_unitree --frames 12 --start 0.0 --end 0.6
    sets.update({"SOH": "/content/Depth-Anything-3/assets/examples/SOH", "robot_unitree": "/content/samples/robot_unitree"})
if MY_MULTIVIEW:
    sets.update({f"my_{k}": v for k, v in scenes_in(MY_MULTIVIEW).items()})
for k, v in sets.items():
    print(f"{k}: {sum(1 for p in Path(v).iterdir() if p.suffix.lower() in IMG)} photos in {v}")
assert sets or MY_SINGLE, "no photos found: check MY_MULTIVIEW"'''),
        md("### Multi-view reconstructions"),
        code('''for name, folder in sets.items():
    !python scripts/infer.py --images "{folder}" --out "{OUT}/{name}" --weights {WEIGHTS} {EXTRA_ARGS}
    !python scripts/render_preview.py "{OUT}/{name}"
!ls -la {OUT}'''),
        code('''from IPython.display import Image, display
for name in sets:
    print(name, "| columns: RGB | depth (red = near, blue = far) | ray map (direction as RGB) | depth confidence (red = high, blue = low)")
    for f in ("preview.png", "preview_3d.png"):
        p = f"{OUT}/{name}/{f}"
        if os.path.exists(p):
            display(Image(p, width=1000))
    s = f"{OUT}/{name}/summary.json"
    if os.path.exists(s):
        print(open(s).read()[:1500])'''),
        md("### Optional: single images (depth map + ray map per photo, N_v = 1)"),
        code('''if MY_SINGLE:
    !python scripts/infer.py --images "{MY_SINGLE}" --out {OUT}/single --weights {WEIGHTS} --per-image --mesh none
    for d in sorted(Path(f"{OUT}/single").iterdir()):
        if d.is_dir():
            print(d.name); display(Image(str(d / "preview.png"), width=1000))
else:
    print("set MY_SINGLE to run this part")'''),
        md("### Save the results (Colab deletes /content at the end of the session)"),
        code('''import glob, shutil
zips = sorted(glob.glob(f"{OUT}/*.zip"))
print(zips)
if SAVE_TO_DRIVE:
    os.makedirs(SAVE_TO_DRIVE, exist_ok=True)
    for z in zips:
        shutil.copy(z, SAVE_TO_DRIVE)
        print("copied to Drive:", os.path.join(SAVE_TO_DRIVE, os.path.basename(z)))
else:
    from google.colab import files
    for z in zips:
        files.download(z)'''),
    ],
    "05_self_audit": [
        md("""# 05 · Enhancement 1: DA3 Self-Audit, original DA3 vs DA3 + Self-Audit
Training-free enhancement on the **released** weights (setting **[official]**); the network is not
changed. DA3 predicts each camera twice (camera head, ray map), over-parameterises it (6-D ray map)
and can take its own cameras back through E_c: the Self-Audit turns these redundancies into
reliability information. Plan and novelty: `docs/da3_enhancement_report.md` §4.2; how to run and
how to save the results: `docs/ENHANCEMENTS_RUNBOOK.md`.

| | Original DA3 | DA3 + Self-Audit |
|---|---|---|
| **A. Failure detection** (per view) | its own depth confidence | audit model on the redundancy signals (fitted leave-one-dataset-out) |
| **B. Pose source** (paper protocol, Sec. 6) | camera-head poses (API default); ray-map poses (official option) | per-view choice / fusion of head and ray poses (+ oracle as upper bound) |
| **C. Depth uncertainty** (HiRoom, 7Scenes) | confidence calibrated with GT on *other* datasets | per-scene triple collocation, **no GT** |

Settings: **Accelerator = GPU (T4 / P100)**, **Internet = on**. One GPU pass per model (baseline
benchmark run + Self-Audit dump with one extra E_c pass per scene); everything after that is CPU
work (signals, harness replays with the same depth and different cameras, TSDF scoring).
`PHASE = "analyse"` with `RESUME_FROM` re-runs only the analysis (a CPU session is enough).
Outputs (in `/kaggle/working/`): `self_audit/<tag>/` (summary.md, figures, per_view.csv),
`self_audit/results/` (standard result records: **copy them into the repo under
`experiments/results/` and commit**, so notebook 09 can compare all enhancements),
`self_audit_results.zip`."""),
        code(SETUP),
        code(OFFICIAL),
        code(RECON_CHECK.replace("__PATH__", "/kaggle/working/open3d_check.json")),
        code(SCORING_ENV.replace("__FULL_MODES__", "pose recon_unposed")),
        md("### Configuration"),
        code('''MODELS = ["da3-small"]                      # full runs: ONE model per session ("da3-small", then "da3-base")
DATASETS = ["hiroom", "7scenes", "dtu64"]   # add "eth3d", "scannetpp" in a later session (15 / 11 GB)
MAX_SCENES = 2        # 2 = quick end-to-end check (labelled "limited"); 0 = all scenes (the real numbers)
PHASE = "all"         # "all" | "infer" (GPU: baseline run + dump) | "analyse" (CPU is enough)
RESUME_FROM = ""      # output root of an earlier "infer" run of this notebook, e.g. "/kaggle/input/<name>"
WORK = "/tmp/eval_self_audit"               # harness exports + audit dumps (large, not saved)
# split mode only: for PHASE = "infer" use WORK = "/kaggle/working/eval_self_audit" so the dump is saved
OUT = "/kaggle/working/self_audit"          # figures, CSV, summaries
RESULTS = f"{OUT}/results"                  # standard result records (commit these)
BENCH = "/tmp/da3bench"
DS = " ".join(DATASETS)
def tag_for(m):
    return f"sa_{m}_{'-'.join(DATASETS)}" + (f"_limited{MAX_SCENES}" if MAX_SCENES else "")
os.makedirs(RESULTS, exist_ok=True)
if RESUME_FROM:
    !mkdir -p {WORK} && cp -rn {RESUME_FROM}/eval_self_audit/* {WORK}/
    !cp -rn {RESUME_FROM}/self_audit/results/* {RESULTS}/ 2>/dev/null; ls {RESULTS}
print({m: tag_for(m) for m in MODELS})'''),
        md("### Data: DA3-BENCH subsets (HF `depth-anything/DA3-BENCH`)"),
        code('''from huggingface_hub import hf_hub_download
import zipfile
os.makedirs(BENCH, exist_ok=True)
for d in DATASETS:
    if not os.path.exists(f"{BENCH}/{d}"):
        z = hf_hub_download("depth-anything/DA3-BENCH", f"{d}.zip", repo_type="dataset", local_dir="/tmp/zips")
        zipfile.ZipFile(z).extractall(BENCH); os.remove(z)
!du -sh {BENCH}/*'''),
        md("""### 1 · Original DA3 on the paper protocol + Self-Audit dump (GPU)
Released harness, camera-head poses (API default). `--dump-audit` also saves, next to each scene's
export, the ray-map poses (official RANSAC routine and our DLT), a pooled ray map and the E_c
re-pass (DA3 conditioned on its own cameras). The exported depth / poses are unchanged."""),
        code('''if PHASE in ("all", "infer"):
    for m in MODELS:
        t = tag_for(m)
        if SPLIT:   # only the model here (GPU); its scores are computed in step 2 (Python 3.12, CPU)
            !python scripts/eval_bench.py --official-src {OFFICIAL_SRC} --bench-root {BENCH} --work-root {WORK} \\
                --weights official:{m} --datasets {DS} --modes {MODES} --max-scenes {MAX_SCENES} \\
                --tag {t} --dump-audit --infer-only
        else:
            !python scripts/eval_bench.py --official-src {OFFICIAL_SRC} --bench-root {BENCH} --work-root {WORK} \\
                --weights official:{m} --datasets {DS} --modes {MODES} --max-scenes {MAX_SCENES} \\
                --tag {t} --enhancement baseline --variant head --dump-audit --results-root {RESULTS}
!du -sh {WORK}'''),
        md("""### 2 · Self-Audit analysis (CPU)
Signals → per-view GT errors → audit model (leave-one-dataset-out) → A and C metrics, then the
pose-source variants are written with the **same depth** and scored by the harness (`--eval-only`)."""),
        code('''if PHASE in ("all", "analyse"):
    for m in MODELS:
        t = tag_for(m)
        if SPLIT:   # scores of the original DA3 run (Python 3.12, CPU), from the exports of step 1
            !{PY312} scripts/eval_bench.py --official-src {OFFICIAL_SRC} --bench-root {BENCH} --work-root {WORK} \\
                --weights official:{m} --datasets {DS} --modes {MODES} --max-scenes {MAX_SCENES} \\
                --tag {t} --enhancement baseline --variant head --eval-only --results-root {RESULTS}
        !{SCORER} scripts/self_audit_eval.py --official-src {OFFICIAL_SRC} --bench-root {BENCH} --work-root {WORK} \\
            --baseline-tag {t} --weights official:{m} --datasets {DS} --max-scenes {MAX_SCENES} \\
            --results-root {RESULTS} --out-dir {OUT}/{t} {RECON_VARIANTS}'''),
        md("### 3 · Results: original DA3 vs DA3 + Self-Audit"),
        code('''from IPython.display import Image, Markdown, display
for m in MODELS:
    p = f"{OUT}/{tag_for(m)}/summary.md"
    display(Markdown(open(p).read()) if os.path.exists(p) else Markdown(f"_missing: {p}_"))'''),
        code('''import matplotlib.pyplot as plt
from da3.compare import load_records
from da3.compare.plots import sec6_delta_bars
recs = load_records(RESULTS)
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, lab in zip(axes, ["AUC@3", "AUC@30", "F1 w/o p."]):
    sec6_delta_bars(recs, lab, ax=ax)
plt.tight_layout(); plt.show()'''),
        code('''for m in MODELS:
    for f in ["risk_coverage.png", "depth_reliability.png", "tc_vs_gt_sigma.png", "audit_coefficients.png", "audit_examples.png"]:
        p = f"{OUT}/{tag_for(m)}/{f}"
        if os.path.exists(p):
            print(m, "·", f); display(Image(p, width=1100))'''),
        md("### 4 · Go/no-go gates (report §4.2)"),
        code('''for r in load_records(RESULTS):
    if r["enhancement"] == "self_audit" and "gates" in r["extra"]:
        print(f"== {r['weights']} ({r['tag']})")
        for name, g in r["extra"]["gates"].items():
            print(f"  {g['result']:5s} {name}: {g['value']}")'''),
        md("### 5 · Save: download `self_audit_results.zip`, copy `self_audit/results/*` into the repo's `experiments/results/`, commit"),
        code('''!cd /kaggle/working && rm -f self_audit_results.zip && zip -qr self_audit_results.zip self_audit && ls -la self_audit_results.zip
!find {RESULTS} -name "*.json" | sort'''),
    ],
    "05_self_audit_colab": [
        md("""# 05 (Colab) · Enhancement 1: DA3 Self-Audit, original DA3 vs DA3 + Self-Audit
Same as `05_self_audit` (same scripts, same results, same record format), for **Google Colab**.
What it measures: `docs/da3_enhancement_report.md` §4.2; how to save and compare the results:
`docs/ENHANCEMENTS_RUNBOOK.md` (§2b for Colab).

**Before running**
1. **Runtime → Change runtime type**: the first GPU that connects, cheapest first,
   **T4 → L4 → A100 → G4** (about 1.2 / 1.7 / 5.4 / 8.7 compute units per hour; the real rate is
   under **Resources**). If one does not connect within a few minutes, pick the next. All four give
   the same numbers (the model runs in fp16 on every GPU); the first cell checks the GPU in seconds.
   Not the TPUs (the code needs CUDA). **CPU** is enough for `PHASE = "analyse"`.
2. Private repo only: left bar → **key icon (Secrets)** → add `GITHUB_TOKEN` and switch on
   **Notebook access**.
3. **Runtime → Run all**, and allow Google Drive access when asked.

**Checkpoints (on by default).** Colab deletes `/content` when the connection is lost, so the
progress is saved to Drive (`DRIVE_DIR`) every few minutes and after every step.
- **After a disconnect:** reconnect (any GPU; CPU if only the analysis is left), change nothing,
  **Runtime → Run all**. Finished scenes and finished scorings are skipped; it continues from the
  last save. At most the scoring that was running when the connection dropped is redone.
- It needs free Drive space: a few GB for a full model (the exports). When a model is finished,
  its exports are deleted from Drive (`FREE_CHECKPOINT_WHEN_DONE`); its results stay.
- To avoid disconnects: keep this tab open and the computer awake (no sleep). In Chrome, turn off
  Memory Saver for colab.research.google.com (Settings → Performance).

**Compute units.** Colab charges for every hour the runtime is connected, even when it is idle: when
a run ends, **Runtime → Disconnect and delete runtime**.
- Start with the quick check (`MAX_SCENES = 2`, the default), `PHASE = "all"`.
- Full runs: **one model per session**, `PHASE = "all"`. To save units you can stop after the
  model (`PHASE = "infer"`, GPU) and do the rest on a CPU runtime (`PHASE = "analyse"`): the
  checkpoint carries the files over.
- The GPU type changes only rounding, not the protocol; the first cell prints which GPU was used."""),
        code(COLAB_GPU),
        code(COLAB_SETUP),
        code(COLAB_OFFICIAL),
        code(RECON_CHECK.replace("__PATH__", "/content/open3d_check.json")),
        code(SCORING_ENV.replace("__FULL_MODES__", "pose recon_unposed")),
        md("### Configuration"),
        code('''MODELS = ["da3-small"]                      # full runs: ONE model per session ("da3-small", then "da3-base")
DATASETS = ["hiroom", "7scenes", "dtu64"]
MAX_SCENES = 2        # 2 = quick end-to-end check (labelled "limited"); 0 = all scenes (the real numbers)
PHASE = "all"         # "all" | "infer" (GPU: model run + dump only) | "analyse" (CPU: scoring + analysis only)
DRIVE_DIR = "/content/drive/MyDrive/DA3/self_audit_colab"   # results + checkpoints; "" = no Drive (a disconnect loses everything)
CHECKPOINT = True     # save the progress to Drive; after a disconnect just "Run all" again and it continues
SAVE_EVERY_MIN = 5    # background save interval (also saved after every step)
FREE_CHECKPOINT_WHEN_DONE = True   # delete a finished model's exports from Drive (several GB); its results stay
import psutil
FUSION_WORKERS = max(1, min(os.cpu_count() or 1, int(psutil.virtual_memory().total / 2**30 // 6), 8))
print("FUSION_WORKERS =", FUSION_WORKERS)   # parallel TSDF fusions, ~6 GB RAM each: 2 on a standard runtime, more on A100 / G4
WORK = "/content/eval_self_audit"           # harness exports + audit dumps (large)
OUT = "/content/self_audit"                 # figures, CSV, summaries
RESULTS = f"{OUT}/results"                  # standard result records (commit these)
BENCH = "/content/da3bench"
DS = " ".join(DATASETS)
def tag_for(m):
    return f"sa_{m}_{'-'.join(DATASETS)}" + (f"_limited{MAX_SCENES}" if MAX_SCENES else "")

import shutil, subprocess, threading, time
os.makedirs(RESULTS, exist_ok=True)
if DRIVE_DIR:
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs(DRIVE_DIR, exist_ok=True)
if shutil.which("rsync") is None:
    !apt-get -qq install -y rsync > /dev/null
CKPT_WORK, CKPT_OUT = f"{DRIVE_DIR}/eval_self_audit", f"{DRIVE_DIR}/self_audit"
CKPT_ON = bool(DRIVE_DIR and CHECKPOINT)
_ckpt_lock = threading.Lock()

def _rsync(src, dst, *opts):
    os.makedirs(dst, exist_ok=True)
    r = subprocess.run(["rsync", "-r", *opts, f"{src}/", f"{dst}/"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode not in (0, 24):   # 24 = a file vanished while copying (normal while scripts run)
        print(f"[checkpoint] rsync {src} -> {dst}: exit {r.returncode}: {r.stderr[-300:]}")

def save_checkpoint():
    """Copy what changed to Drive: exports + dumps (not the benchmark data, not the temporary replay
    copies, not half-written files) and the results / figures."""
    if not CKPT_ON:
        return
    with _ckpt_lock:
        if os.path.exists(WORK):
            _rsync(WORK, CKPT_WORK, "--size-only", "--exclude=benchmark_dataset", "--exclude=*__*", "--exclude=*.partial.npz")
        if os.path.exists(OUT):
            _rsync(OUT, CKPT_OUT, "--checksum")

def _autosave():
    while True:
        time.sleep(SAVE_EVERY_MIN * 60)
        try:
            save_checkpoint()
        except Exception as e:
            print("[checkpoint] save failed:", e)

def is_done(path):
    """A finished step leaves a complete JSON record (a truncated copy does not count)."""
    try:
        json.load(open(path)); return True
    except Exception:
        return False
def record(enh, tag):
    return f"{RESULTS}/{enh}/{tag}.json"
def infer_marker(t):
    return f"{WORK}/.done_infer_{t}"

if CKPT_ON:   # restore: never overwrites files of this session, only brings back what is missing
    for src, dst in ((CKPT_WORK, WORK), (CKPT_OUT, OUT)):
        if os.path.exists(src):
            _rsync(src, dst, "--ignore-existing", "--exclude=*.partial.npz")
    if not globals().get("_AUTOSAVE_STARTED"):
        threading.Thread(target=_autosave, daemon=True).start()
        _AUTOSAVE_STARTED = True
for m in MODELS:
    t = tag_for(m)
    print(f"{t}: model run {'DONE' if os.path.exists(infer_marker(t)) or is_done(record('baseline', t)) else 'to do'} | "
          f"original DA3 scored {'DONE' if is_done(record('baseline', t)) else 'to do'} | "
          f"Self-Audit {'DONE' if is_done(record('self_audit', t + '__self_audit')) else 'to do'}")
print("checkpoints:", f"on, every {SAVE_EVERY_MIN} min -> {DRIVE_DIR}" if CKPT_ON else "OFF (nothing survives a disconnect)")'''),
        md("### Data: DA3-BENCH subsets (HF `depth-anything/DA3-BENCH`)"),
        code('''from huggingface_hub import hf_hub_download
import zipfile
os.makedirs(BENCH, exist_ok=True)
for d in DATASETS:
    if not os.path.exists(f"{BENCH}/{d}"):
        z = hf_hub_download("depth-anything/DA3-BENCH", f"{d}.zip", repo_type="dataset", local_dir="/content/zips")
        zipfile.ZipFile(z).extractall(BENCH); os.remove(z)
!du -sh {BENCH}/*'''),
        md("""### 1 · Original DA3 on the paper protocol + Self-Audit dump (GPU)
Released harness, camera-head poses (API default). `--dump-audit` also saves, next to each scene's
export, the ray-map poses (official RANSAC routine and our DLT), a pooled ray map and the E_c
re-pass (DA3 conditioned on its own cameras). The exported depth / poses are unchanged.
Resumable: scenes already saved (checkpoint) are not run again (`--skip-existing`)."""),
        code('''if PHASE in ("all", "infer"):
    for m in MODELS:
        t = tag_for(m)
        if os.path.exists(infer_marker(t)) or is_done(record("baseline", t)):
            print(f"[checkpoint] {t}: model run already complete, skipped")
            continue
        assert torch.cuda.is_available(), ("This step runs the model and needs a GPU: Runtime > Change runtime type > "
                                           "T4 / L4 / A100 / G4 GPU (or set PHASE = 'analyse' for a CPU session)")
        _exit_code = None
        if SPLIT:   # only the model here (GPU); its scores are computed in step 2 (Python 3.12, CPU)
            !python scripts/eval_bench.py --official-src {OFFICIAL_SRC} --bench-root {BENCH} --work-root {WORK} \\
                --weights official:{m} --datasets {DS} --modes {MODES} --max-scenes {MAX_SCENES} \\
                --tag {t} --dump-audit --infer-only --skip-existing
        else:
            !python scripts/eval_bench.py --official-src {OFFICIAL_SRC} --bench-root {BENCH} --work-root {WORK} \\
                --weights official:{m} --datasets {DS} --modes {MODES} --max-scenes {MAX_SCENES} \\
                --tag {t} --enhancement baseline --variant head --dump-audit --results-root {RESULTS} \\
                --fusion-workers {FUSION_WORKERS} --skip-existing
        if _exit_code == 0:   # set by the "!" command; marks the model run as complete
            open(infer_marker(t), "w").close()
        save_checkpoint()
!du -sh {WORK}'''),
        md("""### 2 · Self-Audit analysis (CPU)
Signals → per-view GT errors → audit model (leave-one-dataset-out) → A and C metrics, then the
pose-source variants are written with the **same depth** and scored by the harness (`--eval-only`).
Resumable: every finished scoring (original DA3, each variant) is skipped (`--resume`)."""),
        code('''if PHASE in ("all", "analyse"):
    for m in MODELS:
        t = tag_for(m)
        if is_done(record("self_audit", f"{t}__self_audit")):
            print(f"[checkpoint] {t}: Self-Audit analysis already complete, skipped")
            continue
        if SPLIT and not is_done(record("baseline", t)):   # scores of the original DA3 run (Python 3.12, CPU)
            !{PY312} scripts/eval_bench.py --official-src {OFFICIAL_SRC} --bench-root {BENCH} --work-root {WORK} \\
                --weights official:{m} --datasets {DS} --modes {MODES} --max-scenes {MAX_SCENES} \\
                --tag {t} --enhancement baseline --variant head --eval-only --results-root {RESULTS} \\
                --fusion-workers {FUSION_WORKERS}
            save_checkpoint()
        elif SPLIT:
            print(f"[checkpoint] {t}: original DA3 already scored, skipped")
        !{SCORER} scripts/self_audit_eval.py --official-src {OFFICIAL_SRC} --bench-root {BENCH} --work-root {WORK} \\
            --baseline-tag {t} --weights official:{m} --datasets {DS} --max-scenes {MAX_SCENES} \\
            --results-root {RESULTS} --out-dir {OUT}/{t} --fusion-workers {FUSION_WORKERS} {RECON_VARIANTS} --resume
        save_checkpoint()   # summary, figures, CSV and records of this model
        if CKPT_ON and FREE_CHECKPOINT_WHEN_DONE and is_done(record("self_audit", f"{t}__self_audit")):
            with _ckpt_lock:   # exports no longer needed (results saved); the "done" marker stays
                for root in (CKPT_WORK, WORK):
                    shutil.rmtree(f"{root}/workspace/evaluation/{t}", ignore_errors=True)
            print(f"[checkpoint] {t}: finished; its exports were removed from Drive (results kept in {CKPT_OUT})")'''),
        md("### 3 · Results: original DA3 vs DA3 + Self-Audit"),
        code('''from IPython.display import Image, Markdown, display
for m in MODELS:
    p = f"{OUT}/{tag_for(m)}/summary.md"
    display(Markdown(open(p).read()) if os.path.exists(p) else Markdown(f"_missing: {p}_"))'''),
        code('''import matplotlib.pyplot as plt
from da3.compare import load_records
from da3.compare.plots import sec6_delta_bars
recs = load_records(RESULTS)
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, lab in zip(axes, ["AUC@3", "AUC@30", "F1 w/o p."]):
    sec6_delta_bars(recs, lab, ax=ax)
plt.tight_layout(); plt.show()'''),
        code('''for m in MODELS:
    for f in ["risk_coverage.png", "depth_reliability.png", "tc_vs_gt_sigma.png", "audit_coefficients.png", "audit_examples.png"]:
        p = f"{OUT}/{tag_for(m)}/{f}"
        if os.path.exists(p):
            print(m, "·", f); display(Image(p, width=1100))'''),
        md("### 4 · Go/no-go gates (report §4.2)"),
        code('''for r in load_records(RESULTS):
    if r["enhancement"] == "self_audit" and "gates" in r["extra"]:
        print(f"== {r['weights']} ({r['tag']})")
        for name, g in r["extra"]["gates"].items():
            print(f"  {g['result']:5s} {name}: {g['value']}")'''),
        md("""### 5 · Save
`self_audit_results.zip` (summaries, figures, records) goes to Drive (or is downloaded). Then copy
`self_audit/results/*` into the repo's `experiments/results/` and commit, as with the Kaggle run."""),
        code('''!cd /content && rm -f self_audit_results.zip && zip -qr self_audit_results.zip self_audit && ls -la self_audit_results.zip
!find {RESULTS} -name "*.json" | sort
if DRIVE_DIR:
    shutil.copy("/content/self_audit_results.zip", DRIVE_DIR)
    print("saved to Drive:", f"{DRIVE_DIR}/self_audit_results.zip")
else:
    from google.colab import files
    files.download("/content/self_audit_results.zip")'''),
    ],
    "06_foveated": [
        md("""# 06 · Enhancement 2: Foveated DA3, mixed per-view resolutions in one pass
Training-free enhancement on the **released** weights (setting **[official]**); the network is not
changed. DA3 runs every view at one resolution, so the token count grows with views × resolution and
cross-view attention grows quadratically with it. Foveated DA3 gives a few **focus views** the full
resolution (504 px) and the other **context views** a quarter of the tokens (252 px), all in **one
forward pass**. Region level: a **virtual zoom camera** (exact rotation of the source camera) turns a
region of a 6048 × 4032 photo into a 504-px view that DA3 processes with the scene views, its camera
given through E_c. Plan and novelty: `docs/da3_enhancement_report.md` §4.3 · how to run and save:
`docs/ENHANCEMENTS_RUNBOOK.md` §3 · write-up: `reports/enhancement_2_foveated.md`.

Why it is possible: the released backbone gives every patch token the same constant RoPE position in
the cross-view blocks, so cross-view attention is well defined over views with different token
counts; patch embedding, position embedding, within-view blocks, Dual-DPT and the camera head work
per view and run once per resolution group (`src/da3/enhancements/foveated/routing.py`). With all
views at one resolution the routing is **bit-identical** to the original (part 0 checks it).

**Statistics are pre-registered** (`docs/FOVEATED_ANALYSIS_PLAN.md`, constants in
`configs/foveated_analysis.yaml`, whose SHA-256 goes into every result record): the scene is the unit
of analysis; every comparison is paired, with a 95% scene-bootstrap CI; every gate is **PASS / FAIL /
INCONCLUSIVE** read off its CI, with the margins at which the decision would change. Margins: the
pre-registered 3%, raised to the noise floor of a treatment-blind A/A run (the original DA3 with only
the reference view moved), capped at 6%. One primary configuration per gate; mitigations are
secondary (simultaneous CIs); any selection happens on a dev half of the scenes and is confirmed on
the held-out half.

| Part | What | Compared with | Data | Time (T4, both models) |
|---|---|---|---|---|
| 0 · routing check | equal resolutions → bit-identical outputs | `DA3.forward` | robot video frames | ~5 min |
| 1 · probe = gates 1 + 3 | 3 × 16 views / scene: 8 focus at 504 px + 8 context at 252 / 280 px (plain, attention-balanced, 3 random focus choices), focus views alone, A/A | all-504, uniform 378 / 336 / 280, all-252 | HiRoom, 7Scenes | ~1 h |
| 2 · large-view protocol | 100 … 1000 views in one pass, 10% focus views, repeated timings | all-504, uniform at equal tokens, fewer 504-px views at equal tokens, all-252 | 7Scenes, all 7 sequences | ~1.5 h (DA3-Small) |
| 3 · Sec. 6 protocol | released harness (pose AUC, TSDF F1), 25% / 10% focus views, paired CIs | uniform 504 (original DA3) / 336 / 252 | HiRoom, 7Scenes | 3–6 h (TSDF on CPU); ~1 h pose-only |
| 4 · region zoom = gate 2 | zoom views from the full-resolution photos through E_c; oracle, random and texture regions | global 504 / 896 px, naive crop, monocular | ETH3D, 11 scenes (15 GB): **own session** | ~1.5 h |

Settings: **Accelerator = GPU (T4 / P100)**, **Internet = on**. Choose the parts with `RUN`. A
session that stops early can be resumed: attach its output and list it in `RESUME_FROM`; finished
parts are skipped. Outputs in `/kaggle/working/foveated/`: JSON per part, figures, `summary.md`,
`gates.json`, and `results/` (standard result records: **copy them into the repo under
`experiments/results/` and commit**, so notebook 09 can compare all enhancements)."""),
        code(SETUP),
        code(OFFICIAL),
        code('''import json
chk = json.load(open("/kaggle/working/open3d_check.json"))
print("Open3D TSDF OK:", chk["legacy_tsdf_ok"], "| open3d", chk.get("open3d"))'''),
        md("### Configuration"),
        code('''MODELS = ["da3-small", "da3-base"]          # released weights (Apache-2.0)
DATASETS = ["hiroom", "7scenes"]            # view level: the datasets with per-frame GT depth
MAX_SCENES = 0                              # 0 = all scenes; e.g. 2 for a quick end-to-end check ("limited")
RUN = {"parity": True, "probe": True, "scaling": True, "sec6": True, "zoom": False}   # zoom: own session
RESUME_FROM = []                            # outputs of earlier sessions, e.g. ["/kaggle/input/<name>/foveated"]

PLAN = "configs/foveated_analysis.yaml"      # pre-registered analysis plan: do NOT edit after the runs

# 1 · probe (gates 1 and 3): half of the views are focus views at 504 px
PROBE_VIEWS = 16
PROBE_SUBSETS = 3                           # interleaved view subsets per scene (the scene stays the unit)
PROBE_UNIFORM = [504, 378, 336, 280, 252]
PROBE_FOVEATED = ["context_res=252", "context_res=280", "context_res=252,balance=1",   # first = primary (plan)
                  "context_res=336", "context_res=378"]   # exploratory sweep (not in the plan: never gated)
PROBE_FOCUS_SEEDS = [1, 2, 3]               # robustness: random focus views for the primary configuration
PROBE_DIAGNOSTICS = True                    # exploratory: upsampled-context pass + two-pass control (no gates)
# 2 · large-view protocol (7Scenes sequences: 1000 frames each, stairs 500)
SCALING_MODELS = ["da3-small"]
SCALING_SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
SCALING_VIEWS = [100, 250, 500, 1000]
SCALING_FRACTION = 0.1
TIMING_REPEATS, TIMING_SCENES = 3, 2        # median of 3 timed passes on the first 2 scenes
SCALING_TWO_PASS = True                     # exploratory: focus views alone + two-pass control
# 3 · Sec. 6 protocol (released harness, <= 100 frames, seed 42)
SEC6_MODES = ["pose"]                       # TSDF skipped (deviations log); ["pose", "recon_unposed"] = with F1
SEC6_UNIFORM = [504, 336, 252]              # 504 = the original DA3, the reference row
SEC6_FOVEATED = ["fraction=0.25,context_res=252", "fraction=0.25,context_res=252,balance=1",
                 "fraction=0.1,context_res=252"]
KEEP_EXPORTS = False                        # ~2 GB per run; the per-view analysis runs before they are deleted
# 4 · region zoom (ETH3D zip is 15 GB; only the photos and the source views' GT maps are extracted)
ZOOM_SCENES = ["courtyard", "electro", "kicker", "pipes", "relief", "delivery_area", "facade", "office",
               "playground", "relief_2", "terrains"]
ZOOM_SELECTORS = ["oracle", "random", "texture"]   # oracle = primary (report); random / texture = deployable

OUT = "/kaggle/working/foveated"            # JSON per part, figures, summary.md, gates.json
RESULTS = f"{OUT}/results"                  # standard result records (commit these)
WORK = "/tmp/fov_work"                      # harness workspace and Sec. 6 exports (large, not kept)
BENCH = "/tmp/da3bench"
DS = " ".join(DATASETS)
LIM = f"_limited{MAX_SCENES}" if MAX_SCENES else ""
H = f"--official-src {OFFICIAL_SRC} --bench-root {BENCH} --work-root {WORK}"
def sec6_tag(m, cfg):
    return f"fov_{m}_{'-'.join(DATASETS)}{LIM}__{cfg}"
os.makedirs(RESULTS, exist_ok=True)
for src in RESUME_FROM:
    !cp -rn {src}/* {OUT}/ && ls {OUT}'''),
        md("### Data: DA3-BENCH subsets (HF `depth-anything/DA3-BENCH`)"),
        code('''from huggingface_hub import hf_hub_download
import zipfile
os.makedirs(BENCH, exist_ok=True)
need = set(DATASETS) if (RUN["probe"] or RUN["sec6"]) else set()
if RUN["scaling"]:
    need.add("7scenes")
for d in sorted(need):
    if not os.path.exists(f"{BENCH}/{d}"):
        z = hf_hub_download("depth-anything/DA3-BENCH", f"{d}.zip", repo_type="dataset", local_dir="/tmp/zips")
        zipfile.ZipFile(z).extractall(BENCH); os.remove(z)
!du -sh {BENCH}/* 2>/dev/null; df -h /tmp | tail -1'''),
        md("""### 0 · Routing check: equal resolutions must reproduce the original DA3
Bit-identical outputs (depth, confidence, rays, camera head) in fp32 and fp16, with the released
reference-view selection and with poses through E_c; round-off only when same-size views are split
into two groups; permuting views at mixed resolution only permutes the outputs. The notebook stops
if a check fails."""),
        code('''if RUN["parity"]:
    !python scripts/extract_frames.py /kaggle/working/Depth-Anything-3/assets/examples/robot_unitree.mp4 \\
        /kaggle/working/samples/robot_unitree --frames 8 --start 0.1 --end 0.6
    for m in MODELS:
        !python scripts/foveated_eval.py parity --weights official:{m} --images /kaggle/working/samples/robot_unitree \\
            --views 8 --out {OUT}/parity_{m}.json
for m in MODELS:
    p = f"{OUT}/parity_{m}.json"
    if os.path.exists(p):
        assert json.load(open(p))["all_pass"], f"routing check failed for {m}: see {p}"'''),
        md("""### 1 · Probe: gate 1 (uniform-resolution curve) and gate 3 (cross-scale probe)
3 interleaved subsets of 16 evenly spaced views per scene; every other view is a focus view.
- **G1a headroom:** all-252 is worse than all-504 by more than the margin.
- **G1b context informative:** the focus views are better with the context views than alone.
- **G3 (primary 504/252):** focus-view AbsRel non-inferior to all-504 **and** context-pair AUC@30
  non-inferior to all-252.
- Secondary: 280-px context views, attention-mass balancing. Robustness: 3 random focus choices.
- The A/A run (all-504, reference view moved) sets the noise floor of the margins.
- **Exploratory, never gated** (added after the limited run of 2026-11-23, see the plan's deviations
  log): 336 / 378-px context views, context views with 252-px detail on the 504-px grid, the two-pass
  control (all views at 252 px + focus views alone, aligned), cross-pair AUC with rotation and
  translation errors apart, one depth scale per resolution group."""),
        code('''if RUN["probe"]:
    unif = " ".join(map(str, PROBE_UNIFORM))
    fov = " ".join(f'"{s}"' for s in PROBE_FOVEATED)
    seeds = " ".join(map(str, PROBE_FOCUS_SEEDS))
    diag = "--diagnostics" if PROBE_DIAGNOSTICS else ""
    for m in MODELS:
        if not os.path.exists(f"{OUT}/probe_{m}.json"):
            !python scripts/foveated_eval.py probe {H} --weights official:{m} --datasets {DS} \\
                --max-scenes {MAX_SCENES} --n-views {PROBE_VIEWS} --subsets {PROBE_SUBSETS} --uniform {unif} \\
                --foveated {fov} --focus-seeds {seeds} {diag} \\
                --plan {PLAN} --out {OUT}/probe_{m}.json
for m in MODELS:
    p = f"{OUT}/probe_{m}.json"
    if os.path.exists(p):
        an = json.load(open(p))["analysis"]
        print(f"{m} · margins: depth {100 * an['margins']['depth']:.1f}%, pose {100 * an['margins']['pose']:.1f}%")
        for k, g in an["gates"].items():
            print(f"{m} · {g['result']:12s} [{g['role']}] {k}: {g['value']}")'''),
        md("""### 2 · Large-view protocol: time, memory and accuracy from 100 to 1000 views per pass
Full 7Scenes sequences (GT poses for every frame). 10% focus views at 504 px, the rest at 252 px, vs
all at 504 px (until it runs out of memory), uniform resolution with the same token count, fewer
504-px views with the same token count (the focus views included), and all at 252 px. Accuracy is
read on the focus views, which every configuration contains."""),
        code('''if RUN["scaling"]:
    scenes, views = " ".join(SCALING_SCENES), " ".join(map(str, SCALING_VIEWS))
    two_pass = "--two-pass" if SCALING_TWO_PASS else ""
    for m in SCALING_MODELS:
        if not os.path.exists(f"{OUT}/scaling_{m}.json"):
            !python scripts/foveated_eval.py scaling {H} --weights official:{m} --scenes {scenes} \\
                --n-views {views} --fraction {SCALING_FRACTION} --balance --timing-repeats {TIMING_REPEATS} \\
                --timing-scenes {TIMING_SCENES} {two_pass} \\
                --out {OUT}/scaling_{m}.json'''),
        md("""### 3 · Sec. 6 protocol: released harness with uniform and foveated resolutions
Every run goes through `scripts/eval_bench.py` (`--process-res` / `--meter` for uniform resolutions,
`--foveated` for mixed ones) and writes a standard record; the reference row is the original DA3
(504 px, camera-head poses). Then the per-view analysis (focus vs context views, depth vs GT, pose
AUC on view subsets) reads the exports before they are deleted."""),
        code('''from da3.enhancements.foveated import FoveationSpec
if RUN["sec6"]:
    if "recon_unposed" in SEC6_MODES:
        assert chk["legacy_tsdf_ok"], "Open3D TSDF fusion is broken here: use SEC6_MODES = ['pose'] or send open3d_check.json"
    runs = [(f"u{r}", f"--process-res {r} --meter", "baseline", "head" if r == 504 else f"head_res{r}") for r in SEC6_UNIFORM]
    runs += [(FoveationSpec.parse(s).name, f'--foveated "{s}"', "foveated", FoveationSpec.parse(s).name) for s in SEC6_FOVEATED]
    fractions = " ".join(sorted({f"{FoveationSpec.parse(s).fraction:g}" for s in SEC6_FOVEATED}))
    modes = " ".join(SEC6_MODES)
    for m in MODELS:
        ref = sec6_tag(m, "u504")
        for cfg, flags, enh, var in runs:
            t = sec6_tag(m, cfg)
            if os.path.exists(f"{OUT}/sec6views_{t}.json"):
                print("already done:", t)
                continue
            !python scripts/eval_bench.py {H} --weights official:{m} --datasets {DS} --modes {modes} \\
                --max-scenes {MAX_SCENES} --tag {t} --enhancement {enh} --variant {var} --baseline-tag {ref} \\
                --results-root {RESULTS} {flags}
            !python scripts/foveated_eval.py sec6views {H} --tag {t} --config {cfg} --datasets {DS} \\
                --fractions {fractions} --out {OUT}/sec6views_{t}.json
            if not KEEP_EXPORTS:
                !rm -rf {WORK}/workspace/evaluation/{t}'''),
        md("""### 4 · Region zoom (gate 2): virtual zoom cameras on ETH3D
Run it in its own session (`RUN = {"zoom": True}`, the other parts False; list the first session's
output in `RESUME_FROM` so the final analysis has every part). Per scene, 2 source views; per view one
region (1/4 of the image side, so the zoom view has 4× the global view's pixel density) from each
selector: **oracle** (most geometric detail in the GT: the report's feasibility gate, primary),
**random** and **texture** (image only): deployable, secondary. Gate 2: region AbsRel ≥ 10% better
than global 896 px and the zoom rotation R_v recovered within 1°, read off scene-bootstrap CIs."""),
        code('''if RUN["zoom"]:
    from huggingface_hub import hf_hub_download
    zip_path = hf_hub_download("depth-anything/DA3-BENCH", "eth3d.zip", repo_type="dataset", local_dir="/tmp/zips")
    scenes, selectors = " ".join(ZOOM_SCENES), " ".join(ZOOM_SELECTORS)
    for m in MODELS:
        if not os.path.exists(f"{OUT}/zoom_{m}.json"):
            !python scripts/foveated_eval.py zoom {H} --weights official:{m} --eth3d-zip {zip_path} \\
                --scenes {scenes} --selectors {selectors} --plan {PLAN} --fig-dir {OUT}/zoom_{m} \\
                --out {OUT}/zoom_{m}.json'''),
        md("""### 5 · Results: pre-registered gates, tables, figures, result records
The analysis re-derives every decision from the saved per-scene data and the plan, and warns if a part
ran under a different plan version. `gates_<model>.png`: one row per criterion, CI bar and estimate
coloured by the decision (written next to it), dashed tick = threshold."""),
        code('''models = " ".join(sorted(set(MODELS) | set(SCALING_MODELS)))
!python scripts/foveated_eval.py analyse --out-dir {OUT} --results-root {RESULTS} --models {models} --plan {PLAN}'''),
        code('''from IPython.display import Image, Markdown, display
display(Markdown(open(f"{OUT}/summary.md").read()))'''),
        code('''for m in sorted(set(MODELS) | set(SCALING_MODELS)):
    for f in [f"gates_{m}.png", f"probe_{m}.png", f"sec6_{m}.png", f"scaling_{m}.png", f"gates_zoom_{m}.png",
              f"zoom_{m}/zoom_examples.png"]:
        p = f"{OUT}/{f}"
        if os.path.exists(p):
            print(m, "·", f); display(Image(p, width=1100))'''),
        md("### 6 · Save: download `foveated_results.zip`, copy `foveated/results/*` into the repo's `experiments/results/`, commit"),
        code('''!cd /kaggle/working && rm -f foveated_results.zip && zip -qr foveated_results.zip foveated && ls -la foveated_results.zip
!find {RESULTS} -name "*.json" | sort'''),
    ],
    "07_lora": [
        md("""# 07 · Enhancement 3: LoRA platform + E3 go/no-go (part 0)
Part 0 of the LoRA work (`docs/LORA.md`, statistics pre-registered in `docs/LORA_ANALYSIS_PLAN.md`).
**No training happens in this notebook yet.** It checks that everything the training will rely on
works, and answers the go/no-go question of E3 before any LoRA is trained:

| Part | What | Needs | Time (T4) |
|---|---|---|---|
| A · platform check | unit tests; with the **released** weights on the GPU: LoRA off = bit-identical, B = 0 = bit-identical; measured memory / time of one LoRA training step (replaces the estimates in docs/LORA.md) | nothing | ~10 min |
| B · KITTI conversion | raw drives -> canonical format, LiDAR depth (evaluation only), pre-registered drive split with the Eigen-test leakage guard | your KITTI dataset | ~5 min / drive |
| C · go/no-go | released DA3 unposed vs conditioned on the OXTS poses, same clips: **GO if the posed depth is ≥ 5% better** (AbsRel vs LiDAR) | B | ~10 min |
| D · fountain reference | released-model reconstruction of your fountain photos: the "step 0" every later LoRA reconstruction is compared with | your photos | ~5 min |

Settings: **Accelerator = GPU (T4 / P100)**, **Internet = on** (clone, weights, Eigen list).

**Your inputs** (right panel > Add Input > Your Datasets):
* KITTI raw, uploaded as a private dataset: for each drive the *synced+rectified* zip
  (`<date>_drive_XXXX_sync.zip`: image_02, oxts, velodyne_points) **and** the calibration zip of its
  date (`<date>_calib.zip`). Kaggle unpacks the zips; the converter finds the folders itself.
  For part C use **2 drives that are NOT in the Eigen test list** (they become validation drives;
  test drives are never looked at before a LoRA exists).
* The fountain photos (a folder of images of one scene).

Outputs in `/kaggle/working/lora/` (zipped as `lora_part0.zip`): `lora_check.json`, `kitti/split.json`,
`gonogo_<model>.json/.md`, `fountain/released_<model>/` (reconstruction + Blender package).
**Send me** `lora_check.json` and `gonogo_<model>.md`."""),
        code(SETUP),
        code('''import torch
assert torch.cuda.is_available(), "No GPU: right panel > Settings > Accelerator > GPU T4 (or P100), then run all again."
if True:
''' + _GPU_SMOKE),
        code('''# Open3D (TSDF meshing of the fountain) + self-test; without a working TSDF the fountain uses depth-grid meshing
import sys, json
if sys.version_info[:2] <= (3, 12):
    !pip install -q "open3d==0.19.0"
else:
    !pip install -q "open3d>=0.19.0"
!python scripts/check_open3d.py --out /kaggle/working/open3d_check.json
try:
    MESH = "tsdf" if json.load(open("/kaggle/working/open3d_check.json"))["legacy_tsdf_ok"] else "depth"
except Exception:
    MESH = "depth"
print("fountain mesh:", MESH)'''),
        md("### Configuration"),
        code('''PRESET = "da3-small"     # primary model of the pre-registered plan; "da3-base" = replication
KITTI_RAW = ""           # Copy path from the Input panel, e.g. "/kaggle/input/kitti-raw"; "" = skip parts B, C
FOUNTAIN = ""            # folder with the fountain photos, e.g. "/kaggle/input/my-photos/multiview/fountain"; "" = skip D
RUN = {"check": True, "kitti": True, "gonogo": True, "fountain": True}
EIGEN_URL = "https://raw.githubusercontent.com/nianticlabs/monodepth2/master/splits/eigen/test_files.txt"
OUT = "/kaggle/working/lora"
CANON = "/tmp/canonical/kitti"            # converted drives (large, not saved)
import os
os.makedirs(OUT, exist_ok=True)
if not KITTI_RAW:
    print("KITTI_RAW is empty: parts B and C are skipped")
if not FOUNTAIN:
    print("FOUNTAIN is empty: part D is skipped")'''),
        md("### A · Platform check (unit tests + released weights on the GPU)"),
        code('''if RUN["check"]:
    !python -m pytest -q tests/test_lora.py tests/test_kitti.py 2>&1 | tail -n 3
    !python scripts/lora_check.py --presets {PRESET} --out {OUT}/lora_check.json
    chk = json.load(open(f"{OUT}/lora_check.json"))
    for p, r in chk["bit"].items():
        print(p, "| deterministic repeat:", r["deterministic"], "| LoRA off / B=0 bit-identical:", r["pass"],
              "| trainable LoRA params:", r["lora_params"])
    for k, r in chk["probe"].items():
        print(f"{k:28s}", f"peak {r['peak_gb']:.2f} GB, {r['s_per_step']:.2f} s/step" if r["ok"] else "OUT OF MEMORY")'''),
        md("### B · KITTI raw -> canonical format + drive split"),
        code('''if RUN["kitti"] and KITTI_RAW:
    import urllib.request
    urllib.request.urlretrieve(EIGEN_URL, "/tmp/eigen_test_files.txt")
    !python scripts/prepare_kitti.py --raw {KITTI_RAW} --out {CANON} --eigen-test-list /tmp/eigen_test_files.txt
    !mkdir -p {OUT}/kitti && cp {CANON}/split.json {OUT}/kitti/split.json
    split = json.load(open(f"{CANON}/split.json"))
    print("validation drives (used by part C):", split["val"])
    print("test drives (NOT used here):", split["test"], "| training drives:", split["train"])'''),
        md("""### C · Go/no-go: does the pose-conditioned DA3 (E3's teacher) give better depth than the unposed one?
Same clips for both: 8 frames (every 3rd), 10 clips per drive, 1.8:1 centre crop at 504x280, depth
vs LiDAR with one median scale per clip, cameras from the camera head. With only 2 drives the
numbers are descriptive (no CI); the rule is the report's: **GO if posed / unposed AbsRel ≤ 0.95**.
If only the trajectory errors (ATE / RPE) improve, E3 would reduce to ordinary pose-supervised
fine-tuning (report §4.4)."""),
        code('''if RUN["gonogo"] and KITTI_RAW:
    !python scripts/kitti_eval.py --root {CANON} --drives val --arms released posed --preset {PRESET} \\
        --out {OUT}/gonogo_{PRESET}.json
    from IPython.display import Markdown, display
    display(Markdown(open(f"{OUT}/gonogo_{PRESET}.md").read()))'''),
        md("""### D · Fountain: reference reconstruction with the released weights
The same photos, frame order, resolution (504) and settings will later be reconstructed with every
LoRA (arms, seeds, training snapshots) and rendered side by side. The fountain has no ground truth,
so it shows how the reconstruction *changes*, never whether it is better."""),
        code('''if RUN["fountain"] and FOUNTAIN:
    FOUT = f"{OUT}/fountain/released_{PRESET}"
    !python scripts/infer.py --images {FOUNTAIN} --out {FOUT} --weights official:{PRESET} --mesh {MESH}
    !python scripts/render_preview.py {FOUT}
    from IPython.display import Image, display
    display(Image(f"{FOUT}/preview.png", width=1000)); display(Image(f"{FOUT}/preview_3d.png", width=1000))'''),
        md("### Save: download `lora_part0.zip` (Output panel) and send me `lora_check.json` + `gonogo_<model>.md`"),
        code('''!cd /kaggle/working && rm -f lora_part0.zip && zip -qr lora_part0.zip lora && ls -la lora_part0.zip
!find {OUT} -maxdepth 2 -name "*.json" -o -maxdepth 2 -name "*.md" | sort'''),
    ],
    "09_compare_all": [
        md("""# 09 · Original DA3 vs all enhancements
Reads the standard result records (`da3.compare`) of every run: the committed ones in
`experiments/results/` plus any attached notebook outputs (`/kaggle/input/*/…/results`). Runs are
only compared when they share weights, datasets, scenes and frame sampling. CPU is enough
(**Accelerator = None**, **Internet = on** for the clone)."""),
        code(SETUP),
        code('''import glob
RESULT_DIRS = ["experiments/results"] + sorted(set(glob.glob("/kaggle/input/*/results") +
                                                  glob.glob("/kaggle/input/*/*/results")))
print(RESULT_DIRS)
dirs = " ".join(RESULT_DIRS)
!python scripts/compare_results.py --dirs {dirs} --out /kaggle/working/comparison.md > /dev/null'''),
        code('''from IPython.display import Markdown, display
display(Markdown(open("/kaggle/working/comparison.md").read()))'''),
        code('''import matplotlib.pyplot as plt
from da3.compare import load_records
from da3.compare.plots import sec6_delta_bars
recs = load_records(*RESULT_DIRS)
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, lab in zip(axes, ["AUC@3", "AUC@30", "F1 w/o p."]):
    sec6_delta_bars(recs, lab, ax=ax)
plt.tight_layout(); plt.show()'''),
    ],
}


def main():
    out = Path(__file__).parent
    for name, cells in NOTEBOOKS.items():
        meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}, **METADATA.get(name, {})}
        nb = {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}
        (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1))
        print("wrote", out / f"{name}.ipynb")


if __name__ == "__main__":
    main()
