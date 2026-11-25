"""Adapter that lets the released benchmark harness (depth_anything_3.bench.Evaluator) run OUR model.

The harness calls `api.inference(image_files, [extrinsics, intrinsics], export_dir=...,
export_format="mini_npz", ref_view_strategy=...)` and later reads
`<export_dir>/exports/mini_npz/results.npz` (depth, conf, extrinsics, intrinsics). This class
reproduces the released API's pre/post-processing with our modules:

* preprocessing: longest side -> 504, sides rounded to multiples of 14 (io_utils);
* forward: fp16 autocast on CUDA (the API uses bf16 when supported; T4/P100 do not support it);
* cameras from the camera head D_C (API default `use_ray_pose=False`) or from the ray map;
* with input poses: E_c conditioning on normalized poses, then Umeyama alignment of the prediction
  to the input poses (RANSAC when >= 10 views, seed 42), depth divided by that scale, and the
  input extrinsics / (resized) intrinsics returned, exactly like `_align_to_input_extrinsics_intrinsics`.

Optional `dumper`: a callable run after each UNPOSED prediction, with access to the raw network
outputs (e.g. the Self-Audit dump, da3.enhancements.self_audit.AuditDumper). It never changes the
exported results; without a dumper the adapter behaves exactly as before.

Optional `skip_existing` (resume after an interrupted run): a scene whose export (and dump, when a
dumper is set) already exists and loads is not run again; its saved prediction is returned. Exports
are written atomically (temporary file, then rename), so a saved file is always complete.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from da3.camera import to_homogeneous
from da3.eval.metrics import align_sim3_ransac
from da3.io_utils import load_views, scale_intrinsics, to_model_input
from da3.model import DA3


def savez_atomic(path: str, **arrays) -> None:
    """np.savez_compressed to a temporary file, then rename: readers never see a partial file."""
    tmp = path[:-4] + ".partial.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _loads(path: str) -> dict | None:
    """All arrays of an .npz, or None if it is missing or unreadable (e.g. a truncated copy)."""
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    except Exception:
        return None


class OursAsOfficialAPI:
    def __init__(self, model: DA3, device: str | torch.device = "cuda", use_ray_pose: bool = False,
                 process_res: int = 504, amp: bool = True, dumper=None, skip_existing: bool = False):
        self.model = model.eval().to(device)
        self.device = torch.device(device)
        self.use_ray_pose = use_ray_pose
        self.process_res = process_res
        self.amp = amp and self.device.type == "cuda"
        self.timings = []
        self.dumper = dumper
        self.skip_existing = skip_existing
        self.skipped = 0
        self._last = None  # raw outputs of the last prediction, kept only when a dumper is set

    def _existing(self, image_files, export_dir: str, posed: bool) -> dict | None:
        res = _loads(os.path.join(export_dir, "exports", "mini_npz", "results.npz"))
        if res is None or not {"depth", "conf", "extrinsics", "intrinsics"} <= set(res):
            return None
        if self.dumper is not None and not posed and \
                _loads(os.path.join(export_dir, "exports", "audit_dump.npz")) is None:
            return None
        return {**res, "images": load_views(image_files, self.process_res)[0]}

    @torch.no_grad()
    def predict(self, image_files, extrinsics=None, intrinsics=None, ref_view_strategy="first"):
        rgb, orig_sizes = load_views(image_files, self.process_res)
        H, W = rgb.shape[1:3]
        x = to_model_input(rgb).to(self.device)
        w2c_in = K_in = None
        if extrinsics is not None:
            K_in = np.stack([scale_intrinsics(K, sz, (H, W)) for K, sz in zip(intrinsics, orig_sizes)])
            w2c_in = to_homogeneous(torch.from_numpy(np.asarray(extrinsics, np.float32))).numpy()
        t0 = time.time()
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
            out = self.model(
                x,
                w2c=None if w2c_in is None else torch.from_numpy(w2c_in)[None].to(self.device),
                K=None if K_in is None else torch.from_numpy(K_in).float()[None].to(self.device),
                ref_strategy=ref_view_strategy,
            )
        if self.use_ray_pose:
            w2c, K = DA3.cameras_from_rays(out, (H, W))
        else:
            w2c, K = DA3.cameras_from_head(out["pose_enc"], (H, W))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.timings.append((len(image_files), time.time() - t0))
        if self.dumper is not None and w2c_in is None:
            self._last = {"x": x, "out": out, "ref": ref_view_strategy}
        pred = {
            "depth": out["depth"][0].float().cpu().numpy(),
            "conf": out["depth_conf"][0].float().cpu().numpy(),
            "extrinsics": w2c[0].float().cpu().numpy(),
            "intrinsics": K[0].float().cpu().numpy(),
            "images": rgb,
        }
        if w2c_in is not None:  # align prediction to the given poses (released API behaviour)
            _, _, scale, _ = align_sim3_ransac(pred["extrinsics"], w2c_in, ransac=len(w2c_in) >= 10, seed=42)
            pred["depth"] = pred["depth"] / scale
            pred["extrinsics"] = w2c_in[:, :3, :].astype(np.float32)
            pred["intrinsics"] = K_in.astype(np.float32)
        return pred

    # -- the method the benchmark harness calls
    def inference(self, image, extrinsics=None, intrinsics=None, export_dir=None,
                  export_format="mini_npz", ref_view_strategy="first", **_):
        if self.skip_existing and export_dir is not None:
            pred = self._existing(image, export_dir, posed=extrinsics is not None)
            if pred is not None:
                self.skipped += 1
                return pred
        pred = self.predict(image, extrinsics, intrinsics, ref_view_strategy)
        if export_dir is not None:
            path = os.path.join(export_dir, "exports", "mini_npz", "results.npz")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            savez_atomic(path, depth=np.round(pred["depth"], 8), conf=np.round(pred["conf"], 2),
                         extrinsics=pred["extrinsics"], intrinsics=pred["intrinsics"])
            if self._last is not None:
                self.dumper(self, self._last["x"], self._last["out"], export_dir, self._last["ref"])
        self._last = None
        return pred
