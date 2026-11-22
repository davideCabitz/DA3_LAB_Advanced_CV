"""Benchmark adapters for enhancement #2: Foveated DA3 on the released Sec. 6 harness.

* `MeteredAPI`: the original adapter (`OursAsOfficialAPI`, any uniform `process_res`) plus a
  per-scene efficiency record. Predictions are unchanged.
* `FoveatedAPI`: focus views at `spec.focus_res`, context views at `spec.context_res`, ONE forward
  pass (routing.foveated_forward). The harness expects one stacked depth array per scene, so the
  export brings every view to the focus views' size: context depth / confidence are bilinearly
  upsampled and their intrinsics rescaled per axis (the harness then resizes depth to the original
  image size and rescales the intrinsics, exactly as for a uniform run).

Both write `<export_dir>/exports/foveated_meta.npz` next to the harness' results.npz: focus mask,
per-view processed size, patch tokens, backbone GFLOPs, forward time and peak GPU memory, reference
view. With `spec.fraction = 1` FoveatedAPI is bit-identical to the original adapter.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from da3.camera import to_homogeneous
from da3.eval.bench_adapter import OursAsOfficialAPI
from da3.eval.metrics import align_sim3_ransac
from da3.io_utils import load_views, scale_intrinsics, to_model_input

from .plan import FoveationSpec, backbone_cost, model_dims, select_focus
from .routing import cameras_from_head, cameras_from_rays, foveated_forward, resize_maps, scale_K

META_FILE = "foveated_meta.npz"


def load_mixed_views(image_files, focus: np.ndarray, focus_res: int, context_res: int):
    """Released preprocessing of every view at its own resolution (focus / context), with the
    released center-crop to a common size inside each resolution. Returns (views (3, H, W) list,
    original sizes, processed sizes)."""
    n = len(image_files)
    views, sizes, hws = [None] * n, [None] * n, [None] * n
    for res, mask in ((focus_res, focus), (context_res, ~focus)):
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        rgb, sz = load_views([image_files[i] for i in idx], res)
        x = to_model_input(rgb)[0]
        for j, i in enumerate(idx):
            views[i], sizes[i], hws[i] = x[j], sz[j], tuple(int(t) for t in x.shape[-2:])
    return views, sizes, hws


class MeteredAPI(OursAsOfficialAPI):
    """Original DA3 adapter + per-scene efficiency meta (predictions unchanged)."""

    config_name = "uniform"

    def __init__(self, model, device="cuda", process_res: int = 504, use_ray_pose: bool = False, amp: bool = True):
        super().__init__(model, device, use_ray_pose=use_ray_pose, process_res=process_res, amp=amp)
        self.config_name = f"u{process_res}"
        self.dims = model_dims(self.model)
        self.stats = []
        self._meta = None

    def _reset_peak(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def _peak_gb(self) -> float:
        return torch.cuda.max_memory_allocated(self.device) / 1e9 if self.device.type == "cuda" else float("nan")

    def _record(self, focus, hws, t: float, ref):
        cost = backbone_cost(hws, **self.dims)
        self._meta = {"config": self.config_name, "n_views": len(hws), "focus": np.asarray(focus, bool),
                      "view_hw": np.asarray(hws, np.int32), "tokens": cost["tokens"], "gflops": cost["gflops"],
                      "time_s": float(t), "peak_mem_gb": self._peak_gb(), "ref_index": -1 if ref is None else int(ref)}
        self.stats.append({k: v for k, v in self._meta.items() if k not in ("focus", "view_hw")})

    def predict(self, image_files, extrinsics=None, intrinsics=None, ref_view_strategy="first"):
        self._reset_peak()
        pred = super().predict(image_files, extrinsics, intrinsics, ref_view_strategy)
        n, t = self.timings[-1]
        hw = tuple(pred["depth"].shape[1:3])
        self._record(np.ones(n, bool), [hw] * n, t, None)
        return pred

    def inference(self, image, extrinsics=None, intrinsics=None, export_dir=None, **kw):
        pred = super().inference(image, extrinsics, intrinsics, export_dir=export_dir, **kw)
        if export_dir is not None and self._meta is not None:
            path = os.path.join(export_dir, "exports", META_FILE)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(path, **{k: np.asarray(v) for k, v in self._meta.items()})
        self._meta = None
        return pred

    def summary(self) -> dict:
        """Means over the scenes run so far (peak memory: max)."""
        if not self.stats:
            return {}
        out = {k: float(np.mean([s[k] for s in self.stats])) for k in ("n_views", "tokens", "gflops", "time_s")}
        out["peak_mem_gb"] = float(np.nanmax([s["peak_mem_gb"] for s in self.stats])) if self.device.type == "cuda" else float("nan")
        out["scenes"] = len(self.stats)
        return out


class FoveatedAPI(MeteredAPI):
    """Mixed per-view resolutions in one pass (focus views at full resolution)."""

    def __init__(self, model, device="cuda", spec: FoveationSpec | None = None, use_ray_pose: bool = False,
                 amp: bool = True):
        spec = spec or FoveationSpec()
        super().__init__(model, device, process_res=spec.focus_res, use_ray_pose=use_ray_pose, amp=amp)
        self.spec = spec
        self.config_name = spec.name

    @torch.no_grad()
    def predict(self, image_files, extrinsics=None, intrinsics=None, ref_view_strategy="first"):
        spec = self.spec
        n = len(image_files)
        focus = select_focus(n, spec.fraction, strategy=spec.strategy, seed=spec.seed)
        views, sizes, hws = load_mixed_views(image_files, focus, spec.focus_res, spec.context_res)
        export_hw = hws[int(np.flatnonzero(focus)[0])]
        w2c_in = K_view = None
        if extrinsics is not None:
            w2c_in = to_homogeneous(torch.from_numpy(np.asarray(extrinsics, np.float32))).numpy()
            K_view = torch.from_numpy(np.stack([scale_intrinsics(K, sz, hw) for K, sz, hw in
                                                zip(intrinsics, sizes, hws)])).float()
        views = [v.to(self.device) for v in views]
        self._reset_peak()
        t0 = time.time()
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
            out = foveated_forward(
                self.model, views, w2c=None if w2c_in is None else torch.from_numpy(w2c_in), K=K_view,
                ref_strategy=ref_view_strategy, balance=spec.balance,
                ref_candidates=list(np.flatnonzero(focus)) if spec.ref == "focus" else None)
        if self.use_ray_pose:
            w2c, K = cameras_from_rays(out["ray"], hws)
        else:
            w2c, K = cameras_from_head(out["pose_enc"], hws)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t = time.time() - t0
        self.timings.append((n, t))
        K_exp = torch.stack([scale_K(K[s], hws[s], export_hw) for s in range(n)])
        pred = {
            "depth": resize_maps(out["depth"], export_hw).cpu().numpy(),
            "conf": resize_maps(out["depth_conf"], export_hw).cpu().numpy(),
            "extrinsics": w2c.float().cpu().numpy(),
            "intrinsics": K_exp.float().cpu().numpy(),
            "images": None,
        }
        if w2c_in is not None:  # align the prediction to the given poses (released API behaviour)
            K_in = np.stack([scale_intrinsics(K0, sz, export_hw) for K0, sz in zip(intrinsics, sizes)])
            _, _, scale, _ = align_sim3_ransac(pred["extrinsics"], w2c_in, ransac=n >= 10, seed=42)
            pred["depth"] = pred["depth"] / scale
            pred["extrinsics"] = w2c_in[:, :3, :].astype(np.float32)
            pred["intrinsics"] = K_in.astype(np.float32)
        self._record(focus, hws, t, out["ref_index"])
        return pred
