# Running the examples on Kaggle (and using your own photos)

This guide covers `notebooks/04_demo_export.ipynb`. It runs the **released DA3 weights** (feed-forward,
no training) on example images and on your own photos. It produces:

- **Multi-view reconstruction**: from several photos of one scene, it estimates the camera poses
  and builds a coloured point cloud and a mesh. Everything comes packaged as a zip you open in
  Blender.
- **Depth map + ray map per image**: every photo goes through the model on its own (one view), like
  the depth and ray panels in Fig. 2 of the paper.

Everything runs on Kaggle; nothing needs to be installed on your computer except Blender (to look
at the result).

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
  - `--mesh depth` / `--mesh none`: cheaper meshing, or no mesh at all;
  - `--poses poses.json`: known cameras, if you have them. Format: `{"frames": [{"file": "IMG_0001.jpg",
    "K": 3x3 in pixels of the original image, "c2w": 4x4 OpenCV camera-to-world}, ...]}`.
    The depth is then expressed in the scale of your poses.

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
3. Open Blender (3.6 or newer; checked headless with 5.0.1). Go to the **Scripting** workspace,
   **Open** `blender_import.py` from the unzipped folder, and click **Run Script**.
4. You get:
   - `DA3_points`: the point cloud, shown as coloured points through Geometry Nodes. Use *Material
     Preview* or *Rendered* shading to see the colours.
   - `DA3_mesh`: the mesh, hidden by default. Toggle the eye icon in the Outliner.
   - `DA3_cam_000 …`: one camera per photo, with the photo as background. Select a camera, press
     *Ctrl+Numpad 0* to make it active, then *Numpad 0* to look through it.
   - Camera 0 sits at the origin, looking along +Y, with +Z up.
5. You can also run it from a terminal: `blender --python blender_import.py -- --dir <unzipped folder>`.

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
