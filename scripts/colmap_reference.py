"""COLMAP reference cameras for a photo set without ground truth (enhancement #4 photo demo).

  python scripts/colmap_reference.py --images /kaggle/input/trento/building --out /kaggle/working/lens_demo/colmap

Classical SfM (SIFT, exhaustive matching, incremental mapping, one camera per image so mixed lenses
are allowed) through pycolmap, on CPU. Its poses are a *pseudo ground truth*: accurate on a
well-textured building, but not GT. Results are labelled "vs COLMAP" everywhere.

The photos are first copied with their EXIF orientation applied (OpenCV, which DA3's pipeline uses,
rotates portrait photos by their EXIF tag; COLMAP does not), downscaled to --max-size and saved with
the remaining EXIF (focal length) so COLMAP gets its usual focal prior. Intrinsics are reported in
pixels of the *oriented original* image.

Output: <out>/colmap_cameras.json {"images": {name: {"K", "w2c", "hw"}}, "registered", "total",
"pycolmap"}, plus the COLMAP database and sparse model.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def oriented_copies(src: Path, dst: Path, max_size: int) -> dict:
    """EXIF-transposed, downscaled JPEG copies; returns {name: (scale, (h, w) oriented original)}."""
    from PIL import Image, ImageOps

    dst.mkdir(parents=True, exist_ok=True)
    info = {}
    for p in sorted(q for q in src.iterdir() if q.suffix.lower() in IMG_EXTS):
        with Image.open(p) as im:
            exif = im.getexif()
            im = ImageOps.exif_transpose(im).convert("RGB")
            W, H = im.size
            s = min(1.0, float(max_size) / max(W, H))
            if s < 1.0:
                im = im.resize((max(1, round(W * s)), max(1, round(H * s))), Image.LANCZOS)
            if 0x0112 in exif:
                del exif[0x0112]   # orientation is applied: drop the tag
            out = dst / (p.stem + ".jpg")
            try:
                im.save(out, quality=95, exif=exif.tobytes())
            except Exception:
                im.save(out, quality=95)
            info[p.name] = (im.size[0] / W, (H, W), out.name)
    return info


def _cam_from_world(img) -> np.ndarray:
    """World-to-camera (3, 4) of a pycolmap image, across pycolmap versions."""
    if hasattr(img, "cam_from_world"):
        cfw = img.cam_from_world
        cfw = cfw() if callable(cfw) else cfw
        return np.asarray(cfw.matrix(), np.float64)[:3, :4]
    R = np.asarray(img.rotation_matrix(), np.float64)       # pycolmap < 0.5
    return np.concatenate([R, np.asarray(img.tvec, np.float64)[:, None]], 1)


def run(images: Path, out: Path, max_size: int = 1600, matcher: str = "exhaustive") -> dict:
    import pycolmap

    work = out / "images"
    if work.exists():
        shutil.rmtree(work)
    info = oriented_copies(images, work, max_size)
    db = out / "database.db"
    if db.exists():
        db.unlink()
    sparse = out / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    pycolmap.extract_features(str(db), str(work), camera_mode=pycolmap.CameraMode.PER_IMAGE)
    if matcher == "sequential":
        pycolmap.match_sequential(str(db))
    else:
        pycolmap.match_exhaustive(str(db))
    maps = pycolmap.incremental_mapping(str(db), str(work), str(sparse))
    if not maps:
        raise SystemExit("COLMAP registered no model (too little overlap or texture?)")
    rec = max(maps.values(), key=lambda r: r.num_reg_images())
    by_copy = {v[2]: (k, v[0], v[1]) for k, v in info.items()}
    images_out = {}
    for _, img in rec.images.items():
        if img.name not in by_copy or getattr(img, "has_pose", True) is False or getattr(img, "registered", True) is False:
            continue
        name, s, hw = by_copy[img.name]
        K = np.asarray(rec.cameras[img.camera_id].calibration_matrix(), np.float64)
        K[:2] /= s   # copy pixels -> oriented original pixels
        images_out[name] = {"K": K.tolist(), "w2c": _cam_from_world(img).tolist(), "hw": list(hw)}
    res = {"images": images_out, "registered": len(images_out), "total": len(info),
           "pycolmap": getattr(pycolmap, "__version__", "?"), "max_size": max_size, "matcher": matcher,
           "note": "pseudo ground truth (classical SfM), not GT"}
    (out / "colmap_cameras.json").write_text(json.dumps(res, indent=1))
    print(f"[colmap] registered {len(images_out)} / {len(info)} images -> {out / 'colmap_cameras.json'}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-size", type=int, default=1600)
    ap.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run(Path(args.images), out, args.max_size, args.matcher)


if __name__ == "__main__":
    main()
