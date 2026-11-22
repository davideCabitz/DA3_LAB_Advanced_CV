"""Extra per-scene outputs the audit needs, written next to the harness export during the GPU run.

`AuditDumper` is plugged into `OursAsOfficialAPI(dumper=...)`. After each UNPOSED prediction it
writes `<export_dir>/exports/audit_dump.npz` with:

* head_w2c, head_K: camera-head poses (the API default, identical to results.npz extrinsics);
* ray_dlt_w2c, ray_dlt_K: ray-map poses with our plain Eq. 1 + Eq. 2 (DLT + RQ);
* ray_off_w2c, ray_off_K: ray-map poses with the official confidence-weighted RANSAC routine, when a
  `ray_solver` is given (scripts/eval_bench.py injects the pinned official function; src/da3 never
  imports the official package);
* ray_ds, ray_conf_ds: ray map [origin, direction] and its confidence, 4x4 average-pooled (pooling
  a ray map keeps it consistent with pixel centres of the coarser grid);
* depth_ds, depth_conf_ds: pass-1 depth / confidence pooled to (H/4, W/4);
* repass_*: the E_c fixed-point re-pass, DA3 conditioned on its own head cameras. Its outputs are
  read directly from the network (D_C and ray map), NOT the input poses the API would return.

Full-resolution depth / confidence are not duplicated: they are in the harness' results.npz.
Writing the dump does not change results.npz (baseline numbers are unaffected).
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from da3.camera import to_homogeneous
from da3.model import DA3


def _pool_ray(ray: torch.Tensor, k: int) -> torch.Tensor:
    """(S, h, w, C) -> (S, h/k, w/k, C) by k x k averaging (h, w divisible by k)."""
    x = ray.permute(0, 3, 1, 2)
    h, w = x.shape[-2:]
    if h % k or w % k:
        x = F.adaptive_avg_pool2d(x, (h // k, w // k))
    else:
        x = F.avg_pool2d(x, k)
    return x.permute(0, 2, 3, 1)


def _pool_map(m: torch.Tensor, k: int) -> torch.Tensor:
    """(S, H, W) -> (S, H/k, W/k) adaptive average pooling."""
    return F.adaptive_avg_pool2d(m[:, None], (m.shape[-2] // k, m.shape[-1] // k))[:, 0]


def _np16(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy().astype(np.float16)


def _np32(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy().astype(np.float32)


class AuditDumper:
    def __init__(self, ray_solver=None, repass: bool = True, pool: int = 4):
        self.ray_solver = ray_solver
        self.repass = repass
        self.pool = pool
        self.failures = []

    @torch.no_grad()
    def __call__(self, api, x: torch.Tensor, out: dict, export_dir: str, ref_strategy: str = "first") -> str:
        H, W = x.shape[-2:]
        k = self.pool
        dump = {"image_hw": np.array([H, W], np.int32), "pool": np.int32(k)}
        head_w2c, head_K = DA3.cameras_from_head(out["pose_enc"].float(), (H, W))
        dump["head_w2c"], dump["head_K"] = _np32(head_w2c[0]), _np32(head_K[0])
        ray = out["ray"].float()
        ray_conf = out.get("ray_conf")
        # plain DLT in chunks of views (the DLT system is large at full ray-map resolution)
        parts = [DA3.cameras_from_rays({"ray": ray[:, s:s + 16]}, (H, W)) for s in range(0, ray.shape[1], 16)]
        dump["ray_dlt_w2c"] = _np32(torch.cat([p[0] for p in parts], dim=1)[0])
        dump["ray_dlt_K"] = _np32(torch.cat([p[1] for p in parts], dim=1)[0])
        if self.ray_solver is not None and ray_conf is not None:
            try:
                w2c, K = self.ray_solver(ray, ray_conf.float(), (H, W))
                dump["ray_off_w2c"], dump["ray_off_K"] = _np32(w2c[0]), _np32(K[0])
            except Exception as e:  # keep the run going; the analysis falls back to the DLT poses
                self.failures.append(f"{export_dir}: official ray solver failed: {e!r}")
        dump["ray_ds"] = _np16(_pool_ray(ray[0], k))
        if ray_conf is not None:
            dump["ray_conf_ds"] = _np16(_pool_map(ray_conf[0].float(), k))
        dump["depth_ds"] = _np16(_pool_map(out["depth"][0].float(), k))
        dump["depth_conf_ds"] = _np16(_pool_map(out["depth_conf"][0].float(), k))

        if self.repass:
            t0 = time.time()
            with torch.autocast(api.device.type, dtype=torch.float16, enabled=api.amp):
                out2 = api.model(x, w2c=to_homogeneous(head_w2c.float()), K=head_K.float(), ref_strategy=ref_strategy)
            w2c2, K2 = DA3.cameras_from_head(out2["pose_enc"].float(), (H, W))
            dump["repass_w2c"], dump["repass_K"] = _np32(w2c2[0]), _np32(K2[0])
            dump["repass_depth_ds"] = _np16(_pool_map(out2["depth"][0].float(), k))
            dump["repass_depth_conf_ds"] = _np16(_pool_map(out2["depth_conf"][0].float(), k))
            dump["repass_ray_ds"] = _np16(_pool_ray(out2["ray"][0].float(), k))
            if api.device.type == "cuda":
                torch.cuda.synchronize()
            dump["time_repass_s"] = np.float32(time.time() - t0)
            del out2

        path = os.path.join(export_dir, "exports", "audit_dump.npz")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **dump)
        return path
