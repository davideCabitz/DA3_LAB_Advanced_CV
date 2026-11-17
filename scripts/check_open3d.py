"""Self-test of Open3D TSDF fusion (the benchmark's reconstruction metrics and the demo mesh rely on it).

Fuses a synthetic fronto-parallel wall seen by one camera with the legacy ScalableTSDFVolume (the API
used by the released benchmark harness) and, for information, with the tensor VoxelBlockGrid API.
A working setup yields a non-empty mesh. Exit code 1 if the legacy API is broken.

  python scripts/check_open3d.py [--out report.json]
"""

import argparse
import json
import sys

import numpy as np


def legacy_tsdf(o3d, depth_dtype: str) -> int:
    H, W = 120, 160
    depth = np.full((H, W), 1.0, np.float32)
    color = np.full((H, W, 3), 128, np.uint8)
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01, sdf_trunc=0.04, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    if depth_dtype == "uint16_mm":
        d_img, scale = o3d.geometry.Image((depth * 1000).astype(np.uint16)), 1000.0
    else:
        d_img, scale = o3d.geometry.Image(depth), 1.0
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color), d_img, depth_scale=scale, depth_trunc=3.0, convert_rgb_to_intensity=False)
    vol.integrate(rgbd, o3d.camera.PinholeCameraIntrinsic(W, H, 100.0, 100.0, W / 2, H / 2), np.eye(4))
    return len(vol.extract_triangle_mesh().vertices)


def tensor_tsdf(o3d) -> int:
    H, W = 120, 160
    depth = o3d.t.geometry.Image(np.full((H, W), 1.0, np.float32))
    K = o3d.core.Tensor([[100.0, 0, W / 2], [0, 100.0, H / 2], [0, 0, 1]], o3d.core.Dtype.Float64)
    vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=("tsdf", "weight"), attr_dtypes=(o3d.core.float32, o3d.core.float32),
                                        attr_channels=((1), (1)), voxel_size=0.01, block_resolution=16,
                                        block_count=10000, device=o3d.core.Device("CPU:0"))
    ext = o3d.core.Tensor(np.eye(4), o3d.core.Dtype.Float64)
    blocks = vbg.compute_unique_block_coordinates(depth, K, ext, depth_scale=1.0, depth_max=3.0)
    vbg.integrate(blocks, depth, K, ext, depth_scale=1.0, depth_max=3.0)
    return len(vbg.extract_triangle_mesh().vertex.positions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import open3d as o3d

    rep = {"open3d": o3d.__version__, "python": sys.version.split()[0]}
    for kind in ("float32", "uint16_mm"):
        try:
            rep[f"legacy_vertices_{kind}"] = legacy_tsdf(o3d, kind)
        except Exception as e:  # noqa: BLE001
            rep[f"legacy_vertices_{kind}"] = f"error: {type(e).__name__}: {e}"
    try:
        rep["tensor_vertices"] = tensor_tsdf(o3d)
    except Exception as e:  # noqa: BLE001
        rep["tensor_vertices"] = f"error: {type(e).__name__}: {e}"
    ok = all(isinstance(rep[f"legacy_vertices_{k}"], int) and rep[f"legacy_vertices_{k}"] > 0
             for k in ("float32", "uint16_mm"))
    rep["legacy_tsdf_ok"] = ok
    print(json.dumps(rep, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)
    if not ok:
        print("\n*** Open3D legacy TSDF fusion is BROKEN in this environment: reconstruction metrics (03_eval) "
              "would be wrong. The demo (04) falls back to depth-grid meshing. Send this output to fix it. ***")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
