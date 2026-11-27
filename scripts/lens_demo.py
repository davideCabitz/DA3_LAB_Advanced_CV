"""Enhancement #4 photo demo: the same photos through the released DA3 and the lens-consistent DA3.

  python scripts/lens_demo.py --images /kaggle/input/trento/building --out /kaggle/working/lens_demo/da3-base \
      --weights official:da3-base [--colmap /kaggle/working/lens_demo/colmap/colmap_cameras.json]

Three reconstructions of one photo set (docs/LENS.md §8), each in the layout of scripts/infer.py
(depth/, conf/, ray/, cameras.json, points.ply, mesh.glb, transforms.json, colmap/, images/,
preview.png, blender_import.py) plus a zip for Blender:

  released  scripts/infer.py unchanged: every photo at 504 px, centre-cropped to one common size
            (portrait and landscape photos together lose their top / sides)
  orient    every photo at 504 px with its own aspect ratio (no crop), one pass (mixed-resolution
            routing of enhancement #2): the orientation half of P1
  lens      P1: every photo at the resolution that gives all photos the same angular resolution
            (pixels per degree), focal lengths from EXIF (FocalLengthIn35mmFilm), or from DA3's own
            `orient` pass when a photo has no EXIF; native aspect; one pass

Comparison in <out>/compare/ (no ground truth exists for these photos):
* reprojection consistency (scale-free, GT-free): how well every view's depth agrees with its
  neighbours' depth where they see the same surface, overall and per lens;
* with --colmap (scripts/colmap_reference.py, classical SfM = pseudo ground truth): pose AUC@3 /
  AUC@30 of every arm vs COLMAP on the registered photos, camera-centre error after a Sim(3)
  alignment and focal-length ratio, per lens;
* figures: depth and ray maps of the same photos side by side (depth normalised per arm by the
  median depth of the photos taken with the most common lens), top-down camera layout.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def load_model(weights: str, device):
    from da3.checkpoint_io import official_model
    from da3.model import DA3

    if weights.startswith("official:"):
        m, _ = official_model(weights.split(":", 1)[1])
        return m.eval().to(device), "official"
    state = torch.load(weights, map_location="cpu", weights_only=False)
    m = DA3(state["config"]["model"])
    m.load_state_dict(state["model"])
    return m.eval().to(device), "reduced-scale"


def colorize(d, lo=None, hi=None, high_is_red=False):
    v = d[np.isfinite(d) & (d > 0)]
    if lo is None or hi is None:
        lo, hi = (np.percentile(v, 2), np.percentile(v, 98)) if len(v) else (0.0, 1.0)
    x = np.clip((d - lo) / max(hi - lo, 1e-8), 0, 1)
    x = x if high_is_red else 1 - x
    img = cv2.cvtColor(cv2.applyColorMap((255 * x).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    img[~(np.isfinite(d) & (d > 0))] = 0
    return img


def ray_rgb(ray, hw):
    d = ray[..., 3:6]
    d = d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-8)
    return cv2.resize(((d + 1) / 2 * 255).clip(0, 255).astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)


def unproject(depth, K, w2c):
    h, w = depth.shape
    vv, uu = np.mgrid[0:h, 0:w]
    z = depth.astype(np.float64)
    X = np.stack([(uu + 0.5 - K[0, 2]) / K[0, 0] * z, (vv + 0.5 - K[1, 2]) / K[1, 1] * z, z], -1)
    R, t = np.asarray(w2c)[:3, :3], np.asarray(w2c)[:3, 3]
    return (X - t) @ R   # c2w: R^T (X - t)


def fit_height(img, h):
    return cv2.resize(img, (max(1, round(img.shape[1] * h / img.shape[0])), h), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------- one arm (any per-view sizes)
def write_arm(out: Path, names, imgs, res: dict, args, setting: str, note: str) -> dict:
    """infer.py-style folder from forward_views output on per-view processed images `imgs`."""
    from da3.export import depth_edge_mask, depth_mesh, tsdf_mesh, write_colmap_text, write_glb, write_ply, \
        write_points_glb, write_transforms_json

    for sub in ("depth", "conf", "ray", "images"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    n = len(names)
    hws = [tuple(int(t) for t in im.shape[:2]) for im in imgs]
    depth, conf, rays = res["depth"], res["conf"], res["ray"]
    K, w2c = np.asarray(res["K"], np.float64), np.asarray(res["w2c"], np.float64)
    c2w = []
    for i in range(n):
        T = np.eye(4)
        T[:3, :4] = w2c[i]
        c2w.append(np.linalg.inv(T))
    for i, nm in enumerate(names):
        stem = Path(nm).stem
        np.save(out / "depth" / f"{stem}.npy", depth[i])
        cv2.imwrite(str(out / "depth" / f"{stem}.png"), cv2.cvtColor(colorize(depth[i]), cv2.COLOR_RGB2BGR))
        np.save(out / "conf" / f"{stem}.npy", conf[i])
        if rays is not None:
            np.save(out / "ray" / f"{stem}.npy", rays[i])
            cv2.imwrite(str(out / "ray" / f"{stem}.png"), cv2.cvtColor(ray_rgb(rays[i], hws[i]), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "images" / nm), cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR))
    cams = {"image_size_hw": list(hws[0]), "setting": setting, "weights": args.weights,
            "cameras_used": "camera head D_C", "note": note,
            "frames": [{"file": nm, "hw": list(hws[i]),
                        "head": {"K": K[i].tolist(), "w2c": w2c[i].tolist()},
                        "used": {"K": K[i].tolist(), "w2c": w2c[i].tolist(), "c2w": c2w[i].tolist()}}
                       for i, nm in enumerate(names)]}
    (out / "cameras.json").write_text(json.dumps(cams, indent=2))

    edges = [depth_edge_mask(d, args.edge_rtol) if args.edge_rtol > 0 else np.zeros(d.shape, bool) for d in depth]
    thr = np.percentile(np.concatenate([c[~e].ravel() for c, e in zip(conf, edges)]), args.conf_percentile) \
        if args.conf_percentile > 0 else -np.inf
    keep = [(~e) & (c >= thr) & np.isfinite(d) & (d > 0) for c, e, d in zip(conf, edges, depth)]
    pts = [unproject(d, K[i], w2c[i]) for i, d in enumerate(depth)]
    P = np.concatenate([p[k] for p, k in zip(pts, keep)])
    C = np.concatenate([im[k] for im, k in zip(imgs, keep)])
    if len(P) > 2_000_000:
        sel = np.random.default_rng(0).choice(len(P), 2_000_000, replace=False)
        P, C = P[sel], C[sel]
    write_ply(out / "points.ply", P, C)
    write_points_glb(out / "scene.glb", P, C, w2c, K, hws)   # demo-style points + camera wireframes
    write_transforms_json(out / "transforms.json", w2c, K, hws, [f"images/{nm}" for nm in names],
                          {"cameras_used": "camera head D_C", "note": note})
    write_colmap_text(out / "colmap", w2c, K, hws, names)
    mesher, mesh_info = "none", None
    if args.mesh != "none":
        v = None
        if args.mesh == "tsdf":
            try:
                v, f, c = tsdf_mesh([np.where(k, d, 0.0) for d, k in zip(depth, keep)], imgs, K, w2c)
                mesher = "tsdf"
            except (ImportError, RuntimeError) as e:
                print(f"[lens_demo] TSDF meshing unavailable ({e}); depth-grid meshing instead")
        if v is None:
            v, f, c = depth_mesh(pts, imgs, conf, args.conf_percentile, mask=[~e for e in edges])
            mesher = "depth-grid"
        mesh_info = write_glb(out / "mesh.glb", v * np.array([1.0, -1.0, -1.0]), f, c)
    rows = []
    for i in range(min(n, 6)):
        parts = [imgs[i], colorize(depth[i]), ray_rgb(rays[i], hws[i]) if rays is not None else imgs[i] * 0,
                 colorize(np.log(np.maximum(conf[i], 1e-6)), high_is_red=True)]
        rows.append(np.concatenate([fit_height(p, 200) for p in parts], 1))
    if rows:
        W = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0))) for r in rows]
        cv2.imwrite(str(out / "preview.png"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    summary = {"views": n, "sizes_hw": [list(h) for h in hws], "forward_s": res["time_s"], "mixed_sizes": res["mixed"],
               "setting": setting, "weights": args.weights, "points": int(len(P)), "mesher": mesher,
               "mesh_vertices_faces": mesh_info, "note": note}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    shutil.copy(REPO / "scripts" / "blender_import.py", out / "blender_import.py")
    shutil.copy(REPO / "scripts" / "blender_turntable.py", out / "blender_turntable.py")
    print("package:", shutil.make_archive(str(out), "zip", root_dir=out))
    return summary


def load_arm(folder: Path) -> dict:
    cams = json.loads((folder / "cameras.json").read_text())
    names = [f["file"] for f in cams["frames"]]
    return {"names": names,
            "depth": [np.load(folder / "depth" / f"{Path(n).stem}.npy") for n in names],
            "conf": [np.load(folder / "conf" / f"{Path(n).stem}.npy") for n in names],
            "ray": [np.load(folder / "ray" / f"{Path(n).stem}.npy") if (folder / "ray" / f"{Path(n).stem}.npy").exists()
                    else None for n in names],
            "K": np.array([f["used"]["K"] for f in cams["frames"]], np.float64),
            "w2c": np.array([np.asarray(f["used"]["w2c"])[:3, :4] for f in cams["frames"]], np.float64),
            "hw": [tuple(f.get("hw", cams["image_size_hw"])) for f in cams["frames"]]}


# ----------------------------------------------------------------------------- comparison
def lens_groups(f35):
    """Lens label per photo from EXIF 35-mm-equivalent focal lengths (photos within 10% share a lens)."""
    labels, centres = [], []
    for f in f35:
        if f is None or not np.isfinite(f):
            labels.append("unknown")
            continue
        for c in centres:
            if abs(np.log(f / c)) < np.log(1.1):
                labels.append(f"{c:.0f}mm")
                break
        else:
            centres.append(f)
            labels.append(f"{f:.0f}mm")
    return labels


def camera_centres(w2c):
    return np.stack([-w[:3, :3].T @ w[:3, 3] for w in w2c])


def compare(arms: dict, names, groups, colmap: dict | None, orig_hws, res_by_arm, out: Path, args):
    from da3.enhancements.lens.metrics import reprojection_consistency
    from da3.eval.metrics import pose_metrics, umeyama
    from da3.io_utils import target_size

    out.mkdir(parents=True, exist_ok=True)
    report = {"arms": {}, "groups": sorted(set(groups)), "colmap": None}
    dominant = max(set(groups), key=groups.count)
    for arm, A in arms.items():
        rc = reprojection_consistency(A["depth"], A["K"], A["w2c"], stride=args.stride)
        pv = np.asarray(rc["per_view_median_pct"], np.float64)
        by_g = {g: float(np.nanmean(pv[[i for i in range(len(names)) if groups[i] == g]]))
                for g in sorted(set(groups))}
        report["arms"][arm] = {"reprojection": {k: v for k, v in rc.items() if k != "per_view_median_pct"},
                               "reprojection_per_lens_pct": by_g, "per_view_pct": pv.tolist()}
    if colmap:
        reg = [i for i, n in enumerate(names) if n in colmap["images"]]
        report["colmap"] = {"registered": len(reg), "total": len(names), "pycolmap": colmap.get("pycolmap")}
        if len(reg) >= 3:
            gt_w2c = np.array([np.asarray(colmap["images"][names[i]]["w2c"], np.float64) for i in reg])
            Cg = camera_centres(gt_w2c)
            scale = float(np.median(np.linalg.norm(Cg - Cg.mean(0), axis=1)))
            for arm, A in arms.items():
                pm = pose_metrics(A["w2c"][reg], gt_w2c)
                Cp = camera_centres(A["w2c"][reg])
                R, t, s = umeyama(Cp, Cg)
                err = np.linalg.norm((s * Cp @ R.T + t) - Cg, axis=1) / max(scale, 1e-9)
                # focal length in pixels of the oriented original photo
                fr = []
                for j, i in enumerate(reg):
                    Kc = np.asarray(colmap["images"][names[i]]["K"], np.float64)
                    H, W = orig_hws[i]
                    if arm == "released":
                        pre = target_size(H, W, args.res)      # resized size before the common crop
                        sx = W / pre[1]
                    else:
                        sx = W / A["hw"][i][1]
                    fr.append(A["K"][i][0, 0] * sx / Kc[0, 0])
                fr = np.asarray(fr)
                gidx = {g: [j for j, i in enumerate(reg) if groups[i] == g] for g in sorted(set(groups))}
                report["arms"][arm]["vs_colmap"] = {
                    "auc03": 100 * pm["auc03"], "auc30": 100 * pm["auc30"], "auc05": 100 * pm["auc05"],
                    "centre_err_rel_median": float(np.median(err)),
                    "centre_err_rel_per_lens": {g: float(np.median(err[ix])) for g, ix in gidx.items() if ix},
                    "focal_ratio_per_lens": {g: float(np.median(fr[ix])) for g, ix in gidx.items() if ix}}
    report["dominant_lens"] = dominant
    (out / "summary.json").write_text(json.dumps(report, indent=2, default=float))
    figures(arms, names, groups, dominant, colmap, out)
    md = [f"# Lens demo: {len(names)} photos · {args.weights}\n",
          f"Lenses (EXIF 35-mm equivalent): " + ", ".join(f"{g} × {groups.count(g)}" for g in sorted(set(groups))) +
          f"; dominant: {dominant}.\n",
          "| Arm | reprojection disagreement (median %, lower = more consistent) | within 2% | "
          + " | ".join(f"{g} views (%)" for g in sorted(set(groups))) + " |",
          "|---|---|---|" + "---|" * len(set(groups))]
    for arm in arms:
        r = report["arms"][arm]
        md.append(f"| {arm} | {r['reprojection']['median_pct']:.2f} | {100 * r['reprojection']['within_2pct']:.1f} | "
                  + " | ".join(f"{r['reprojection_per_lens_pct'][g]:.2f}" for g in sorted(set(groups))) + " |")
    if report["colmap"] and any("vs_colmap" in r for r in report["arms"].values()):
        md.append(f"\nVs COLMAP (pseudo ground truth; {report['colmap']['registered']} / {report['colmap']['total']} "
                  "photos registered):\n")
        md.append("| Arm | AUC@3 | AUC@30 | camera-centre error (median, × scene radius) | per lens | focal ratio per lens |")
        md.append("|---|---|---|---|---|---|")
        for arm in arms:
            v = report["arms"][arm].get("vs_colmap")
            if v:
                md.append(f"| {arm} | {v['auc03']:.1f} | {v['auc30']:.1f} | {v['centre_err_rel_median']:.3f} | "
                          + ", ".join(f"{g}: {e:.3f}" for g, e in v["centre_err_rel_per_lens"].items()) + " | "
                          + ", ".join(f"{g}: {e:.2f}" for g, e in v["focal_ratio_per_lens"].items()) + " |")
    md.append("\nPer-photo plan: " + ", ".join(f"{Path(n).stem}: {groups[i]} → {res_by_arm['lens'][i]} px"
                                                for i, n in enumerate(names)))
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return report


def figures(arms, names, groups, dominant, colmap, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(names)
    others = [i for i in range(n) if groups[i] != dominant]
    dom = [i for i in range(n) if groups[i] == dominant]
    pick = others[:6] + [dom[j] for j in np.unique(np.round(np.linspace(0, len(dom) - 1, min(len(dom), 8 - min(6, len(others))))).astype(int))] if dom else others[:8]
    pick = sorted(set(pick))[:8]
    arm_names = list(arms)
    # depth normalised per arm by the median depth of the dominant-lens views
    norm = {}
    for arm in arm_names:
        meds = [np.median(arms[arm]["depth"][i][arms[arm]["depth"][i] > 0]) for i in dom] or [1.0]
        norm[arm] = float(np.median(meds))
    allv = np.concatenate([(arms[a]["depth"][i] / norm[a]).ravel() for a in arm_names for i in pick])
    lo, hi = np.percentile(allv[allv > 0], [2, 98])
    fig, axes = plt.subplots(len(pick), 1 + len(arm_names), figsize=(3.2 * (1 + len(arm_names)), 2.6 * len(pick)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(pick):
        img = cv2.cvtColor(cv2.imread(str(out.parent / "lens" / "images" / names[i])), cv2.COLOR_BGR2RGB)
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"{Path(names[i]).stem} · {groups[i]}", fontsize=8)
        for c, arm in enumerate(arm_names):
            d = arms[arm]["depth"][i] / norm[arm]
            axes[r, c + 1].imshow(colorize(d, lo, hi))
            axes[r, c + 1].set_title(f"{arm}: median {np.median(d[d > 0]):.2f}", fontsize=8)
    for a in axes.ravel():
        a.axis("off")
    fig.suptitle("Depth on one colour scale (red = near); per arm normalised by the median depth of the "
                 f"{dominant} photos", fontsize=10)
    plt.tight_layout()
    fig.savefig(out / "depth_compare.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(len(pick), len(arm_names), figsize=(3.2 * len(arm_names), 2.6 * len(pick)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(pick):
        for c, arm in enumerate(arm_names):
            ray = arms[arm]["ray"][i]
            if ray is not None:
                axes[r, c].imshow(ray_rgb(ray, arms[arm]["depth"][i].shape))
            axes[r, c].set_title(f"{Path(names[i]).stem} · {arm}", fontsize=8)
            axes[r, c].axis("off")
    fig.suptitle("Ray maps (direction as RGB): the field of view each arm assigns to the photo", fontsize=10)
    plt.tight_layout()
    fig.savefig(out / "ray_compare.png", dpi=110)
    plt.close(fig)

    from da3.eval.metrics import umeyama

    ref_w2c = None
    if colmap:
        reg = [i for i, nm in enumerate(names) if nm in colmap["images"]]
        if len(reg) >= 3:
            ref_w2c = (reg, np.array([np.asarray(colmap["images"][names[i]]["w2c"]) for i in reg]))
    use_colmap = ref_w2c is not None
    if ref_w2c is None:
        ref_w2c = (list(range(n)), arms[arm_names[-1]]["w2c"])
    colmap = colmap if use_colmap else None
    reg, Wref = ref_w2c
    Cref = camera_centres(Wref)
    mu = Cref.mean(0)
    _, _, Vt = np.linalg.svd(Cref - mu)
    ax2 = Vt[:2]
    cols = {g: f"C{k}" for k, g in enumerate(sorted(set(groups)))}
    fig, axes = plt.subplots(1, len(arm_names) + (1 if colmap else 0), figsize=(4.5 * (len(arm_names) + 1), 4.5))
    axes = np.atleast_1d(axes)
    panels = [(a, arms[a]["w2c"]) for a in arm_names] + ([("COLMAP", None)] if colmap else [])
    for ax, (label, W) in zip(axes, panels):
        if W is None:
            C, Rw = Cref, np.stack([w[:3, :3] for w in Wref])
            idx = reg
        else:
            Cp = camera_centres(W[reg])
            R, t, s = umeyama(Cp, Cref)
            C = s * Cp @ R.T + t
            Rw = np.stack([w[:3, :3] @ R.T for w in W[reg]])   # world-to-camera rotations in the reference frame
            idx = reg
        P = (C - mu) @ ax2.T
        F = np.stack([r[2] for r in Rw]) @ ax2.T              # viewing direction (camera z in world)
        for j, i in enumerate(idx):
            ax.arrow(P[j, 0], P[j, 1], 0.15 * F[j, 0], 0.15 * F[j, 1], color=cols[groups[i]], head_width=0.03)
            ax.plot(P[j, 0], P[j, 1], "o", color=cols[groups[i]], ms=4)
        ax.set_title(label)
        ax.set_aspect("equal")
    for g, c in cols.items():
        axes[0].plot([], [], "o", color=c, label=g)
    axes[0].legend(fontsize=7)
    fig.suptitle("Cameras seen from above (Sim(3)-aligned to " + ("COLMAP" if colmap else arm_names[-1]) + "), colour = lens")
    plt.tight_layout()
    fig.savefig(out / "cameras_topdown.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    from da3.enhancements.lens import canonical_resolutions, exif_focal_px, reference_kappa
    from da3.enhancements.lens.inference import forward_views
    from da3.io_utils import list_images, preprocess_like_official, read_rgb

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="official:da3-small")
    ap.add_argument("--colmap", default="", help="colmap_cameras.json of scripts/colmap_reference.py (optional)")
    ap.add_argument("--focal-source", choices=["auto", "exif", "da3"], default="auto",
                    help="auto: EXIF when every photo has it, else DA3's own orient pass")
    ap.add_argument("--res", type=int, default=504)
    ap.add_argument("--deadband", type=float, default=0.15)
    ap.add_argument("--max-res", type=int, default=1008)
    ap.add_argument("--min-res", type=int, default=112)
    ap.add_argument("--max-views", type=int, default=0)
    ap.add_argument("--conf-percentile", type=float, default=40.0)
    ap.add_argument("--edge-rtol", type=float, default=0.05)
    ap.add_argument("--mesh", choices=["tsdf", "depth", "none"], default="tsdf")
    ap.add_argument("--stride", type=int, default=4, help="pixel stride of the reprojection-consistency metric")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import infer as infer_mod   # scripts/infer.py: the released arm is exactly its reconstruct()

    paths = list_images(args.images)
    if args.max_views:
        paths = paths[:: max(1, len(paths) // args.max_views)][: args.max_views]
    if len(paths) < 2:
        raise SystemExit(f"need >= 2 photos in {args.images}")
    names = [p.name for p in paths]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, setting = load_model(args.weights, args.device)
    ref_strategy = "saddle_balanced" if model.cfg.ray_head == "released" else "first"
    originals = [read_rgb(p) for p in paths]             # EXIF orientation applied by OpenCV
    orig_hws = [tuple(int(t) for t in im.shape[:2]) for im in originals]
    f_exif = [exif_focal_px(p, hw) for p, hw in zip(paths, orig_hws)]
    f35 = [None if f is None else f * 43.2666 / float(np.hypot(*hw)) for f, hw in zip(f_exif, orig_hws)]
    groups = lens_groups(f35)
    t0 = time.time()

    # ---- released: scripts/infer.py as is
    ns = Namespace(process_res=args.res, poses=None, cameras_from="head", conf_percentile=args.conf_percentile,
                   edge_rtol=args.edge_rtol, mesh=args.mesh, device=args.device, weights=args.weights)
    infer_mod.reconstruct(paths, out / "released", ns, model, setting)

    # ---- orient: native aspect ratio at the released resolution, one pass
    imgs_o = [preprocess_like_official(im, args.res) for im in originals]
    res_o = forward_views(model, imgs_o, args.device, ref_strategy=ref_strategy, keep_ray=True)
    write_arm(out / "orient", names, imgs_o, res_o, args, setting, "native aspect ratio, every photo at "
              f"{args.res} px (no centre-crop)")

    # ---- lens: equal angular resolution
    if args.focal_source == "exif" or (args.focal_source == "auto" and all(f is not None for f in f_exif)):
        if any(f is None for f in f_exif):
            raise SystemExit("--focal-source exif: some photos have no EXIF focal length")
        kappa = np.array([f / max(hw) for f, hw in zip(f_exif, orig_hws)])
        source = "EXIF FocalLengthIn35mmFilm"
    else:
        kappa = np.array([0.5 * (K[0, 0] + K[1, 1]) / max(hw) for K, hw in zip(res_o["K"], res_o["hw"])])
        source = "DA3 camera head (orient pass)"
    res_list = canonical_resolutions(kappa, args.res, "median", args.deadband, args.min_res, args.max_res)
    z_rel = (kappa / reference_kappa(kappa)).tolist()
    if all(r == args.res for r in res_list):
        imgs_l, res_l, note = imgs_o, res_o, "all photos already share one angular resolution: identical to orient"
    else:
        imgs_l = [preprocess_like_official(im, r) for im, r in zip(originals, res_list)]
        res_l = forward_views(model, imgs_l, args.device, ref_strategy=ref_strategy, keep_ray=True)
        note = f"equal angular resolution (focal from {source}), native aspect ratio"
    write_arm(out / "lens", names, imgs_l, res_l, args, setting, note)
    plan = [{"file": n, "hw": list(orig_hws[i]), "exif_f35mm": f35[i], "lens": groups[i], "relative_zoom": z_rel[i],
             "res_lens": res_list[i], "processed_hw_lens": list(imgs_l[i].shape[:2]),
             "processed_hw_orient": list(imgs_o[i].shape[:2])} for i, n in enumerate(names)]
    (out / "plan.json").write_text(json.dumps({"focal_source": source, "deadband": args.deadband, "views": plan},
                                              indent=2, default=float))

    colmap = json.loads(Path(args.colmap).read_text()) if args.colmap and Path(args.colmap).exists() else None
    arms = {a: load_arm(out / a) for a in ("released", "orient", "lens")}
    compare(arms, names, groups, colmap, orig_hws, {"lens": res_list}, out / "compare", args)
    print(f"[lens_demo] done in {time.time() - t0:.0f} s; focal source: {source}; outputs in {out}")


if __name__ == "__main__":
    main()
