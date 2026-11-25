"""Same photos, two (or more) models: compare infer.py reconstructions (docs/LORA.md Phase 5).

  python scripts/compare_reconstructions.py --ref out/fountain/released_da3-small \
      --test e3_s0=out/fountain/e3_s0 po_s0=out/fountain/pose_only_s0 --out out/fountain/compare

Every --test folder must come from the same images, frame order and resolution as --ref (checked).
Without ground truth these numbers say how much the reconstruction CHANGES, not whether it is better.

1. Alignment: Sim(3) (Umeyama) from the test cameras' centres to the reference cameras' centres; the
   test points and cameras are mapped into the reference frame and scale.
2. Metrics per test model: per-view depth difference (after the Sim(3) scale), rotation / translation
   differences of the cameras, Chamfer distance and F1 between the two point clouds (threshold = 1% of
   the reference scene size), confidence statistics, number of points.
3. Pictures: per-view relative depth-difference maps; a side-by-side orbit video (identical virtual
   camera path, framing from the reference, simple z-buffered point splatting) and aligned PLYs
   (`aligned/<name>.ply`) for Blender.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from da3.eval.metrics import umeyama


def read_ply(path) -> tuple[np.ndarray, np.ndarray]:
    """Binary little-endian xyz + rgb PLY as written by da3.export.write_ply."""
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr = np.frombuffer(data[end:], dtype=dt)
    return np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float64), np.stack([arr["red"], arr["green"], arr["blue"]], 1)


def load(folder: Path) -> dict:
    cams = json.loads((folder / "cameras.json").read_text())
    names = [Path(f["file"]).stem for f in cams["frames"]]
    w2c = np.array([np.r_[np.asarray(f["used"]["w2c"])[:3], [[0, 0, 0, 1]]] for f in cams["frames"]])
    pts, rgb = read_ply(folder / "points.ply")
    return {"names": [f["file"] for f in cams["frames"]], "hw": cams["image_size_hw"], "w2c": w2c, "c2w": np.linalg.inv(w2c),
            "depth": np.stack([np.load(folder / "depth" / f"{n}.npy") for n in names]),
            "conf": np.stack([np.load(folder / "conf" / f"{n}.npy") for n in names]), "points": pts, "rgb": rgb,
            "setting": cams.get("setting"), "weights": cams.get("weights")}


def rot_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def compare(ref: dict, test: dict) -> tuple[dict, dict]:
    if ref["names"] != test["names"] or ref["hw"] != test["hw"]:
        raise SystemExit("reference and test were not run on the same images / order / resolution")
    R, t, s = umeyama(test["c2w"][:, :3, 3], ref["c2w"][:, :3, 3])
    c2w = test["c2w"].copy()
    c2w[:, :3, :3] = R @ c2w[:, :3, :3]
    c2w[:, :3, 3] = (s * (R @ test["c2w"][:, :3, 3].T)).T + t
    pts = (s * (R @ test["points"].T)).T + t
    lo, hi = np.percentile(ref["points"], [2, 98], axis=0)
    size = float(np.linalg.norm(hi - lo))
    tau = 0.01 * size
    rng = np.random.default_rng(0)   # fixed subsample (<= 300k points each) keeps the KD-tree queries fast
    pick = lambda P: P[rng.choice(len(P), size=min(len(P), 300_000), replace=False)]
    P_ref, P_test = pick(ref["points"]), pick(pts)
    d_rt = cKDTree(P_test).query(P_ref, k=1)[0]
    d_tr = cKDTree(P_ref).query(P_test, k=1)[0]
    prec, rec = float(np.mean(d_tr < tau)), float(np.mean(d_rt < tau))
    dref, dtest = ref["depth"], test["depth"] * s
    rel = np.abs(dtest - dref) / np.maximum(dref, 1e-6)
    cam_extent = float(np.median(np.linalg.norm(ref["c2w"][:, :3, 3] - ref["c2w"][:, :3, 3].mean(0), axis=1))) or 1.0
    m = {
        "sim3_scale": float(s),
        "scene_size_ref": size, "f1_threshold": tau,
        "chamfer_rel_to_size": float((d_rt.mean() + d_tr.mean()) / 2 / size),
        "precision": prec, "recall": rec, "f1": float(2 * prec * rec / max(prec + rec, 1e-12)),
        "depth_reldiff_median": float(np.median(rel)), "depth_reldiff_p90": float(np.percentile(rel, 90)),
        "depth_reldiff_per_view_median": [float(np.median(r)) for r in rel],
        "rot_diff_deg_mean": float(np.mean([rot_deg(a[:3, :3], b[:3, :3]) for a, b in zip(ref["c2w"], c2w)])),
        "rot_diff_deg_max": float(np.max([rot_deg(a[:3, :3], b[:3, :3]) for a, b in zip(ref["c2w"], c2w)])),
        "center_diff_rel_to_camera_spread": float(np.mean(np.linalg.norm(c2w[:, :3, 3] - ref["c2w"][:, :3, 3], axis=1)) / cam_extent),
        "points_ref": int(len(ref["points"])), "points_test": int(len(pts)),
        "log_conf_mean_ref": float(np.mean(np.log(ref["conf"]))), "log_conf_mean_test": float(np.mean(np.log(test["conf"]))),
    }
    return m, {"points": pts, "rgb": test["rgb"], "c2w": c2w, "rel": rel}


def diff_strip(ref: dict, rel: np.ndarray, k: int = 6, vmax: float = 0.2) -> np.ndarray:
    idx = np.linspace(0, len(rel) - 1, min(k, len(rel))).astype(int)
    rows = []
    for i in idx:
        d = np.clip(rel[i] / vmax, 0, 1)
        heat = cv2.applyColorMap((255 * d).astype(np.uint8), cv2.COLORMAP_INFERNO)
        img = cv2.imread(str(ref["folder"] / "images" / ref["names"][i]))
        img = cv2.resize(img, (heat.shape[1], heat.shape[0])) if img is not None else np.zeros_like(heat)
        rows.append(np.concatenate([img, heat], 1))
    return np.concatenate(rows, 0)


# ----------------------------------------------------------------------------- orbit video (no Blender needed)
def look_at(eye, target, up):
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    w2c = np.eye(4)
    w2c[:3, :3] = np.stack([r, -u, f])            # OpenCV: x right, y down, z forward
    w2c[:3, 3] = -w2c[:3, :3] @ eye
    return w2c


def splat(points, rgb, w2c, K, hw, radius=1):
    H, W = hw
    pc = points @ w2c[:3, :3].T + w2c[:3, 3]
    ok = pc[:, 2] > 1e-6
    pc, col = pc[ok], rgb[ok]
    uv = pc[:, :2] / pc[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    img = np.full((H, W, 3), 255, np.uint8)
    zbuf = np.full((H, W), np.inf)
    order = np.argsort(-pc[:, 2])                # far first; near points overwrite
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            uu, vv = u[order] + du, v[order] + dv
            m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            img[vv[m], uu[m]] = col[order][m][:, ::-1]   # RGB -> BGR
            zbuf[vv[m], uu[m]] = pc[order][m][:, 2]
    return img


def orbit_video(path, clouds: dict, ref: dict, frames=90, hw=(360, 480), arc_deg=60.0, fps=30):
    """Side by side, identical camera path: a swing of +-arc_deg/2 around the reference scene centre,
    starting from the mean photo direction (the photographed side), at the photos' mean height."""
    lo, hi = np.percentile(ref["points"], [3, 97], axis=0)
    center = (lo + hi) / 2
    cams = ref["c2w"][:, :3, 3]
    up = -ref["c2w"][:, :3, 1].mean(0)
    up /= np.linalg.norm(up)
    start = cams.mean(0) - center
    radius = 1.1 * np.linalg.norm(start) if np.linalg.norm(start) > 1e-6 else np.linalg.norm(hi - lo)
    start = start / max(np.linalg.norm(start), 1e-9)
    K = np.array([[0.9 * hw[1], 0, hw[1] / 2], [0, 0.9 * hw[1], hw[0] / 2], [0, 0, 1]])
    names = list(clouds)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (hw[1] * len(names), hw[0]))
    for f in range(frames):
        ang = np.radians(arc_deg / 2 * np.sin(2 * np.pi * f / frames))
        rot = cv2.Rodrigues((up * ang).reshape(3, 1))[0]
        eye = center + radius * (rot @ start)
        w2c = look_at(eye, center, up)
        tiles = []
        for n in names:
            img = splat(clouds[n]["points"], clouds[n]["rgb"], w2c, K, hw)
            cv2.putText(img, n, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            tiles.append(img)
        writer.write(np.concatenate(tiles, 1))
    writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="infer.py output of the released model")
    ap.add_argument("--test", nargs="+", required=True, metavar="NAME=DIR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-points", type=int, default=400_000, help="per cloud, for the video only")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    from da3.export import write_ply

    out = Path(args.out)
    (out / "aligned").mkdir(parents=True, exist_ok=True)
    ref = load(Path(args.ref))
    ref["folder"] = Path(args.ref)
    write_ply(out / "aligned" / "released.ply", ref["points"], ref["rgb"])
    rng = np.random.default_rng(0)
    sub = lambda n: rng.choice(n, size=min(n, args.max_points), replace=False)
    i = sub(len(ref["points"]))
    clouds = {"released": {"points": ref["points"][i], "rgb": ref["rgb"][i]}}
    report = {"ref": args.ref, "tests": {}}
    for spec in args.test:
        name, folder = spec.split("=", 1)
        m, al = compare(ref, load(Path(folder)))
        report["tests"][name] = {"dir": folder, **m}
        write_ply(out / "aligned" / f"{name}.ply", al["points"], al["rgb"])
        cv2.imwrite(str(out / f"depth_diff_{name}.png"), diff_strip(ref, al["rel"]))
        j = sub(len(al["points"]))
        clouds[name] = {"points": al["points"][j], "rgb": al["rgb"][j]}
        print(f"{name}: depth rel. diff median {m['depth_reldiff_median']:.3f} | rot diff {m['rot_diff_deg_mean']:.2f} deg "
              f"| Chamfer {100 * m['chamfer_rel_to_size']:.2f}% of scene | F1@1% {m['f1']:.3f}", flush=True)
    (out / "compare.json").write_text(json.dumps(report, indent=2))
    md = ["| Model | depth rel. diff (median / p90) | rotation diff (mean / max °) | centre diff (rel.) | Chamfer (% scene) | F1@1% | points |",
          "|---|---|---|---|---|---|---|"]
    md += [f"| {n} | {m['depth_reldiff_median']:.3f} / {m['depth_reldiff_p90']:.3f} | {m['rot_diff_deg_mean']:.2f} / {m['rot_diff_deg_max']:.2f} "
           f"| {m['center_diff_rel_to_camera_spread']:.3f} | {100 * m['chamfer_rel_to_size']:.2f} | {m['f1']:.3f} | {m['points_test']} |"
           for n, m in report["tests"].items()]
    (out / "compare.md").write_text("Differences to the released model on the same photos (no ground truth: "
                                    "change, not accuracy).\n\n" + "\n".join(md) + "\n", encoding="utf-8")
    if not args.no_video:
        orbit_video(out / "side_by_side.mp4", clouds, ref)
        print("video:", out / "side_by_side.mp4")
    print("\n".join(md))


if __name__ == "__main__":
    main()
