# Running the examples on Kaggle (and using your own photos)

This guide covers `notebooks/04_demo_export.ipynb`. It runs the **released DA3 weights** (feed-forward,
no training) on example images and on your own photos. It produces:

- **Multi-view reconstruction**: from several photos of one scene, it estimates the camera poses
  and builds a coloured point cloud and a mesh. Everything comes packaged as a zip you open in
  Blender.
- **Depth map + ray map per image**: every photo goes through the model on its own (one view), like
  the depth and ray panels in Fig. 2 of the paper.

Everything runs on Kaggle; nothing needs to be installed on your computer except Blender (to look
at the result). If the Kaggle GPU queue is too long, the same demo runs on **Google Colab**
(`notebooks/04_demo_export_colab.ipynb`, §8).

---

## 1. One-time setup

1. **Kaggle account with phone verification.** Kaggle only allows GPUs and Internet access in
   notebooks for phone-verified accounts (Settings > Phone verification).
2. **Code on GitHub.** The notebooks `git clone` the repo at run time. Push your latest commit to the
   branch the notebook uses (`BRANCH = "main"` in the first code cell).
   - If the repository is **private**, create a GitHub personal access token (read-only access to
     the repo is enough). In the Kaggle notebook editor go to **Add-ons > Secrets**, add a secret
     named `GITHUB_TOKEN`, and tick it for this notebook.
3. **Import the notebook.** Kaggle > **Code > New Notebook**, then **File > Import Notebook**. Upload
   `notebooks/04_demo_export.ipynb`, or paste its GitHub URL.
4. **Notebook settings** (right-hand panel, *Session options*):
   - **Accelerator: GPU T4 x2** (or P100). Only one GPU is used.
   - **Internet: On.** It is needed to clone the repos and download the weights.

## 2. Run the examples only (no personal photos)

Leave `MY_MULTIVIEW = ""` and `MY_SINGLE = ""`, then **Run All**. The notebook:

1. clones this repo and the official DA3 repo (pinned commit), and installs the pinned requirements;
2. downloads **DA3-BASE** (Apache-2.0) from Hugging Face;
3. reconstructs two example sets: `SOH` (2 photos of the Sydney Opera House) and `robot_unitree`
   (12 frames extracted from the official example video);
4. shows a preview per set. Columns: RGB | depth (red = near, blue = far) | ray map | depth
   confidence (red = high, blue = low). It also shows a 3D
   view of the points with the cameras in red.

The results are in `/kaggle/working/demo/` (one folder plus one `.zip` per set).
The robot video has a moving subject, so its geometry is unreliable (see "Bad results" below).

## 3. Upload your own photos

### 3.1 Prepare the folders on your computer
```
my-photos/
├── multiview/             # photos for a 3D reconstruction
│   ├── kitchen/           # one sub-folder = one scene (or put the photos directly in multiview/)
│   │   ├── IMG_0001.jpg
│   │   └── ...
│   └── statue/
│       └── ...
└── single/                # independent photos: depth map + ray map for each one
    ├── photo1.jpg
    └── photo2.png
```
Supported formats: `.jpg .jpeg .png .bmp .webp .tif`. **Convert iPhone `.HEIC` files to `.jpg` first.**
Each multi-view sub-folder must contain **only one scene**. Never mix two places in one folder.

### 3.2 Upload them as a Kaggle Dataset
1. Kaggle > **Datasets > New Dataset**.
2. Drag the `my-photos` folder (or a `.zip` of it; Kaggle unpacks zips) and give it a title, e.g.
   `my-photos`. Keep it **Private**. Click **Create**.
3. Re-uploading new photos later: open the dataset > **New Version**.

### 3.3 Attach it to the notebook and run
1. In the notebook, right panel: **Add Input > Your Datasets > my-photos**.
2. In the *Input* panel, hover over the folder and click **Copy path**. Paste it into the third code
   cell, for example:
   ```python
   MY_MULTIVIEW = "/kaggle/input/my-photos/multiview"
   MY_SINGLE    = "/kaggle/input/my-photos/single"
   ```
   Always use the path Kaggle shows you. Don't type it from memory, because the mount path can
   differ between accounts.
3. **Run All** for an interactive look, or **Save Version > Save & Run All (Commit)** to keep the
   outputs permanently (see §5).

Each sub-folder of `multiview/` becomes one reconstruction named `my_<subfolder>`. `single/` produces
`/kaggle/working/demo/single/<photo name>/` for every photo.

### 3.4 Optional settings (in the same cell)
- `CAMERAS = "colmap"` (default) estimates the cameras of **your** photos with COLMAP (classical structure from
  motion, CPU, a few minutes for ~40 photos). DA3 then predicts the depth with those cameras as input
  (`infer.py --poses`).
  - **Why:** on a 360° walk around a nearly symmetric object (our fountain set), DA3's own cameras collapsed
    half of the walk onto one spot; the result was a "hedgehog" of streaks in Blender.
  - **Coverage:** photos COLMAP cannot register are skipped. If COLMAP fails altogether, DA3's own cameras are
    used.
  - **Alternatives:** `"da3"` uses DA3's own cameras; `"both"` makes one reconstruction of each (`<name>_colmap`,
    `<name>_da3`) for a side-by-side comparison. The built-in examples always use DA3's cameras.
- `EXTRA_ARGS = "--turntable-crop 1.0"` (default): the turntable video hides far background by default
  (`turntable.json` in the zip; the script's own default crop is 2.0). The confidence filter stays at the 40th
  percentile; `--conf-percentile 60` removes more sky, but also thin parts such as a statue on top.
- `OFFICIAL_REFERENCE = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"` also runs the **official** code with the online
  demo's model (1.4B parameters, CC BY-NC) on your photos (`scripts/official_reference.py`).
  - **Output:** `demo/official_<name>/scene.glb`, the same file the online demo gives.
  - **Reference only:** it is a different code path with different weights.
- **Fair comparisons:** released DA3 vs an enhancement always uses the same weights through our code for every
  arm (`da3-small`, `da3-base`, or `official:da3-large` = DA3-LARGE-1.1: CC BY-NC, not parity-checked by us).
  The online demo / `OFFICIAL_REFERENCE` model is never one arm of a comparison.
- `WEIGHTS = "official:da3-small"` is faster but less accurate. `"official:da3-base"` is the default.
- `WEIGHTS = "official:da3-large"` loads DA3-LARGE-1.1. **This is not parity-checked by us yet**, and
  its README license is CC BY-NC (non-commercial).
- `WEIGHTS = "/kaggle/input/<train output>/runs/reduced_vits_tartanair_vkitti2/ckpt/last.pt"` uses
  our own reduced-scale model. Expect much lower quality than the released weights.
- Extra command-line options of `scripts/infer.py` (edit the `!python scripts/infer.py ...` line):
  - `--max-views 40`: use at most 40 evenly spaced photos;
  - `--cameras-from ray`: build the point cloud from the ray-map cameras (paper Eq. 1-2) instead of
    the camera head;
  - `--conf-percentile 60`: drop more low-confidence points (cleaner, sparser point cloud);
  - `--edge-rtol 0.05` (default): drops the "flying pixels" at depth edges. At the border between
    an object and its background the predicted depth passes through in-between values, which
    become streaks of points along the camera rays. `--edge-rtol 0` keeps them. This only changes
    which pixels go into `points.ply` / `mesh.glb`; DA3's depth maps and cameras are untouched;
  - `--mesh depth` / `--mesh none`: cheaper meshing, or no mesh at all;
  - `--poses poses.json`: known cameras, if you have them. Format: `{"frames": [{"file": "IMG_0001.jpg",
    "K": 3x3 in pixels of the original image, "c2w": 4x4 OpenCV camera-to-world}, ...]}` (or `"w2c"`).
    The depth is then expressed in the scale of your poses. Photos without an entry are skipped.
    `scripts/colmap_reference.py --images <folder> --out <dir> --poses-out poses.json` writes this file from COLMAP
    (what `CAMERAS = "colmap"` does);
  - `--turntable-crop 1.0`: default clutter crop of `blender_turntable.py` for this reconstruction.

## 4. What you get

Per reconstruction folder (`/kaggle/working/demo/<name>/` and `<name>.zip`):

| File | Content |
|---|---|
| `preview.png` | RGB / depth / ray map / confidence for up to 6 views |
| `preview_3d.png` | 3D scatter of the point cloud with camera frusta |
| `depth/<img>.png`, `depth/<img>.npy` | colour-mapped depth, and raw depth (relative units) |
| `ray/<img>.png`, `ray/<img>.npy` | ray map: the PNG shows the unit ray direction as RGB; the `.npy` is (h, w, 6) = [origin, direction] |
| `conf/<img>.npy` | depth confidence |
| `cameras.json` | K and world-to-camera matrices from the camera head and from the ray map |
| `points.ply` | fused coloured point cloud (low-confidence pixels removed) |
| `scene.glb` | the same points as glTF POINTS + camera wireframes: the layout of the online demo's `scene.glb`, so both can be compared in the same viewer |
| `mesh.glb` | TSDF mesh with vertex colours |
| `transforms.json`, `colmap/` | cameras in NeRF / instant-ngp format and as a COLMAP text model |
| `images/` | the resized input photos (Blender uses them as camera backgrounds) |
| `blender_import.py` | Blender import script |
| `summary.json` | timings, number of points, agreement between head and ray-map cameras |

**Reading a ray map.** Each pixel stores the ray from the camera centre through that pixel. The PNG
shows the ray's *direction* (x → red, y → green, z → blue). A smooth colour gradient across the
image is expected; its spread reflects the predicted field of view. Noisy or blotchy ray maps mean
the model is unsure about the camera.

**Units.** Without `--poses` the scene has an arbitrary scale (relative depth). Camera 0 (the first
image in alphabetical order) defines the world frame.

## 5. Download and open in Blender

1. **Keep the outputs.** Files in `/kaggle/working` vanish when an interactive session ends.
   Either download the zips from the *Output* panel (right side) before closing, or use
   **Save Version > Save & Run All (Commit)**. The committed version then keeps
   `/kaggle/working` under its *Output* tab.
2. Unzip `<name>.zip` on your computer.
3. Open Blender (3.6 or newer; checked headless with 5.0.1; since 2026-11-27 `blender_import.py` reads `points.ply`
   itself, because Blender 5.3's PLY importer drew our points as streaks). Go to the **Scripting** workspace,
   **Open** `blender_import.py` from the unzipped folder, and click **Run Script**.
4. You get:
   - `DA3_points`: the point cloud, shown as coloured points through Geometry Nodes. Use *Material
     Preview* or *Rendered* shading to see the colours.
   - `DA3_mesh`: the mesh, hidden by default. Toggle the eye icon in the Outliner.
   - `DA3_cam_000 …`: one camera per photo, with the photo as background. Select a camera, press
     *Ctrl+Numpad 0* to make it active, then *Numpad 0* to look through it.
   - Camera 0 sits at the origin, looking along +Y, with +Z up.
5. You can also run it from a terminal: `blender --python blender_import.py -- --dir <unzipped folder>`.

### 5b. A rotating video for slides (`blender_turntable.py`)

The zip also contains `blender_turntable.py`. Zips made before 2026-11-24 don't have it: copy
`scripts/blender_turntable.py` from the repo into the unzipped folder. It renders a short looping
video, `turntable_<mode>.mp4`, into the same folder.

**The camera motion is chosen from your photos** (`mode = "auto"`):

| Your photos | Mode | Motion |
|---|---|---|
| All around the subject (you walked around a statue / a building) | `orbit` | full 360° turn |
| One side only (a façade, a palace front, a statue from 3–4 photos) | `arc` | smooth back-and-forth swing over the photographed side, so the empty back is never shown |
| From inside, facing outwards (a room, a pizzeria, a square from one spot) | `inside` | the camera stands where you stood and pans: 360° if the photos cover all around, otherwise a swing, with a wide lens |

**What is set automatically:**
- the centre of the point cloud, ignoring far sky points;
- points farther than `crop` × the subject size are hidden (sky shells, stray background; the
  files are not changed);
- the scene upright, using your photos' "up" direction;
- the start on the side you photographed from;
- the point size from the measured point spacing, at least about 1 pixel in the video (no gaps,
  no flicker);
- true colours, whatever the background.

The output is H.264 MP4, which plays in PowerPoint, Google Slides and browsers.

**Running it:**
- **In Blender:** after `blender_import.py`, **Scripting → Open** `blender_turntable.py` → **Run Script**.
  - Blender looks frozen while it renders, for a few minutes. The video appears in the folder.
  - The System Console (**Window → Toggle System Console**) prints the chosen mode and why.
  - Settings are in `CONFIG` at the top of the script:
    - `mode` (`"auto"`, `"orbit"`, `"arc"`, `"inside"`);
    - `seconds`, `resolution`, `background` (`"white"`, `"black"` or `"#RRGGBB"`);
    - `zoom` (larger = closer), `elevation`, `lens`;
    - `point_scale` (bigger points), `crop` (`None` keeps all points);
    - `show`: `"points"`, `"mesh"` (the filled TSDF surface, true colours) or `"both"`;
    - `transparent` (PNG frames with alpha).
  - Set `render = False` to preview first: **View → Cameras → Active Camera**, then press **Space**
    to play.
- **From a terminal, without opening Blender** (import + video in one command):
  `blender -b --python blender_import.py --python blender_turntable.py -- --dir <unzipped folder> --background white`.
  Add e.g. `--mode arc --seconds 10 --zoom 1.5` to override the defaults.
- **In the slides:** **Insert → Video** → `turntable_<mode>.mp4`, set to start automatically and loop.
  The loop is seamless.

**If it doesn't look right:**

| Problem | Fix |
|---|---|
| The subject is too small or too big | change `zoom` |
| Gaps between points / rows of dots | raise `point_scale` to 1.5–2, or `show = "mesh"` |
| Thin streaks fanning out from the edges | re-run the notebook: `--edge-rtol` (default 0.05) removes them; zips made before 2026-11-24 still contain them |
| Blobs | lower `point_scale` to 0.6 |
| Sky or background clutter | lower `crop` to 1.5 |
| Parts of the subject are cut off | raise `crop` to 3, or set it to `None` |
| Wrong kind of motion | set `mode` by hand |
| Interior seen from outside | `mode = "inside"` |

## 6. How to get good results

Multi-view photos:
- **Static scene.** Nothing in the scene should move between shots: people, cars, pets, leaves in
  wind. Moving objects break the reconstruction.
- **Overlap.** Walk around the object or along the room, and keep roughly 60-80% overlap between
  consecutive photos. 10-40 photos is a good range. More views help only if they add new viewpoints.
- **Same camera settings.** Use the same lens and zoom for all photos. Turn off "portrait / bokeh"
  modes, panoramas and fisheye or ultra-wide (0.5x) lenses. The model assumes a pinhole camera.
- **Sharp, well-lit photos.** Avoid motion blur, very dark shots and strong backlight.
- **Textured surfaces.** Avoid scenes dominated by plain white walls, mirrors, glass, water or other
  shiny or transparent things.
- **Little sky.** DA3-Small and DA3-Base have no sky head, so sky gets a finite depth and shows up as
  a floating layer. Keep sky out of frame, or raise `--conf-percentile`.
- **Same orientation.** Take all photos in landscape or all in portrait. Mixed sizes are centre-
  cropped to the smallest common size, and part of the image is lost.

Single images: any photo works; depth is relative (no metric scale).

## 7. What NOT to do (breaks things or gives bad results)

- ❌ **Don't** `pip install -e` the official Depth-Anything-3 repo in the notebook. It pins `numpy<2`
  and `xformers` and breaks the Kaggle environment. The notebooks only put it on `sys.path`.
- ❌ **Don't** ignore the Open3D self-test printed after the install cell. If `legacy_tsdf_ok` is
  `false`, the mesh is built with a simpler fallback method (`"mesher": "depth-grid"` in
  `summary.json`). The point cloud is not affected.
- ❌ **Don't** reinstall `torch`. Use the version shipped with the Kaggle image.
- ❌ **Don't** turn Internet off. The clone and the weight download need it.
- ❌ **Don't** write into `/kaggle/input` (read-only), and don't copy large data into
  `/kaggle/working` (20 GB limit, and it slows down *Save Version*).
- ❌ **Don't** put two different scenes, or a video of a moving subject, in one multi-view folder.
- ❌ **Don't** upload HEIC/RAW files or screenshots with UI overlays.
- ❌ **Don't** set `--process-res` far above 504. The models were trained around 504 px, and larger
  inputs mostly cost GPU memory. If you hit "CUDA out of memory", lower `--max-views` first.
- ❌ **Don't** expect metric units (metres) without `--poses`.
- ❌ **Don't** use DA3-LARGE / GIANT outputs commercially (CC BY-NC).
- ❌ **Don't** close an interactive session before downloading the zips. Use *Save & Run All* if
  unsure.

## 8. The same demo on Google Colab

`notebooks/04_demo_export_colab.ipynb` does what `04_demo_export` does (same scripts, same outputs,
same Blender zip), for Google Colab instead of Kaggle.

1. **Push the repo**: the notebook clones `main`, as on Kaggle.
2. **Open the notebook in Colab:** colab.research.google.com → **File → Upload notebook** → choose
   `notebooks/04_demo_export_colab.ipynb`. Or use **GitHub** and pick the repo; a private repo
   needs GitHub access granted to Colab.
3. **GPU:** **Runtime → Change runtime type → T4 GPU**. The notebook asks for it; check that it is
   selected. The first code cell stops with a clear message if there is no GPU.
4. **Private repo:** in the left bar, open the **key icon (Secrets)**, add `GITHUB_TOKEN` (the same
   token as on Kaggle) and switch on **Notebook access**.
5. **Your photos.** Choose one:
   - **Google Drive (recommended):** put the folder with the photos in your Drive, e.g.
     `MyDrive/DA3/building/`, and set `MY_MULTIVIEW = "/content/drive/MyDrive/DA3/building"`.
     When the notebook asks for Drive access, allow it.
   - **Upload a zip:** set `PHOTOS_FROM = "upload"`. Run the notebook, then choose the `.zip` of
     your photo folder when the upload button appears.
6. **Options**, in the same cell as on Kaggle:
   - `WEIGHTS`: Base by default; Small is faster.
   - `EXTRA_ARGS`: `--conf-percentile 60` by default, good for buildings with sky; add
     `--max-views 60` if you get "CUDA out of memory".
   - `RUN_EXAMPLES = True` also reconstructs the two built-in examples.
7. **Runtime → Run all.** The results are in `/content/demo/<name>/` and `<name>.zip`.
8. **Keep the results.** Colab deletes `/content` when the session ends. The last cell copies the
   zips to Drive (`SAVE_TO_DRIVE`, default `MyDrive/DA3/results`). With `SAVE_TO_DRIVE = ""` it
   downloads them instead.

Colab specifics:
- Free sessions disconnect after some idle time or after a few hours, and GPUs are not always
  available. Keep the tab open while it runs.
- Enhancement #1 also has a Colab version (`05_self_audit_colab`, see
  [ENHANCEMENTS_RUNBOOK.md](ENHANCEMENTS_RUNBOOK.md) §2b). The other benchmark notebooks (03, 06)
  download tens of GB and run for hours; they stay on Kaggle.
- Paths that start with `/content/drive/MyDrive/...` are Drive folders; check them in the
  **Files** panel on the left.
