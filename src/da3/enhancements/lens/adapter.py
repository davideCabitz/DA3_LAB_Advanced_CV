"""Benchmark adapter of enhancement #4 (lens-consistent DA3) on the released Sec. 6 harness.

The harness hands the adapter the dataset image files and scores what comes back against the
unchanged ground truth (GT poses, GT meshes / points). `LensAPI` turns the scene into a mixed-lens
photo set *inside* the adapter (plan.LensSpec: which views are zoomed and by how much), runs the arm,
and maps every prediction back to the frame of the dataset image, exactly:

* depth / confidence: warped onto the export grid (the dataset image at the released processing
  size) with nearest neighbours; a zoomed view covers only the central 1/z of its source image, the
  rest is 0 (= invalid, skipped by the harness' TSDF fusion);
* intrinsics: transformed by the same affine map, so a zoomed view whose zoom DA3 understood
  correctly gets back the source camera's focal length;
* extrinsics: unchanged (cropping and resizing do not move a camera).

With `fraction = 0` (no zoomed view) every arm is bit-identical to the original adapter
(tests/test_lens.py, `scripts/lens_eval.py parity`). Unposed only (pose, recon_unposed): the posed
mode would need the zoomed GT intrinsics, and the protocol does not use it.

Per scene, `<export_dir>/exports/lens_meta.npz` stores the view kinds, zoom factors, processing
sizes, the zoom estimates of canon_self, the number of passes, tokens and time.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from da3.eval.bench_adapter import OursAsOfficialAPI, _loads
from da3.io_utils import read_rgb, target_size

from .geometry import KIND_NORMAL, AxisAffine, transform_K, warp_to
from .inference import canonical_views, common_canvas, forward_views, padded_views, released_batch
from .plan import LensSpec, assign_kinds, canonical_resolutions, captures_for, kappa_from_captures, reference_kappa

META_FILE = "lens_meta.npz"


def kappa_from_prediction(K_proc: np.ndarray, affs, photo_hws) -> np.ndarray:
    """kappa_i = predicted focal length in photo pixels / photo longest side (from the processed-frame
    intrinsics of a pass and its photo -> processed maps)."""
    out = []
    for K, a, hw in zip(K_proc, affs, photo_hws):
        Kp = transform_K(K, a.inverse())
        out.append(0.5 * (Kp[0, 0] + Kp[1, 1]) / max(hw))
    return np.asarray(out, np.float64)


def patch_tokens(hws, patch: int = 14) -> int:
    return int(sum((h // patch) * (w // patch) for h, w in hws))


class LensAPI(OursAsOfficialAPI):
    """The original adapter with the mixed-lens protocol and the arms of plan.LensSpec."""

    def __init__(self, model, device="cuda", spec: LensSpec | None = None, use_ray_pose: bool = False,
                 amp: bool = True, skip_existing: bool = False):
        spec = spec or LensSpec()
        super().__init__(model, device, use_ray_pose=use_ray_pose, process_res=spec.res, amp=amp,
                         skip_existing=skip_existing)
        self.spec = spec
        self.stats = []
        self._meta = None

    def _existing(self, image_files, export_dir: str, posed: bool):
        if _loads(os.path.join(export_dir, "exports", META_FILE)) is None:
            return None
        return super()._existing(image_files, export_dir, posed)

    # -- one scene
    @torch.no_grad()
    def run(self, originals, ref_view_strategy: str = "first") -> dict:
        """The arm on in-memory dataset images (uint8 RGB). Returns the export-frame prediction and
        the per-scene meta (used by predict and by scripts/lens_eval.py parity)."""
        spec = self.spec
        n = len(originals)
        orig_hws = [tuple(int(t) for t in im.shape[:2]) for im in originals]
        kinds = assign_kinds(n, spec.fraction, spec.portrait)
        caps = captures_for(spec, orig_hws, kinds)
        photos = [c.apply(im) for c, im in zip(caps, originals)]
        photo_hws = [tuple(int(t) for t in p.shape[:2]) for p in photos]
        export_hw = target_size(orig_hws[0][0], orig_hws[0][1], spec.res)
        kw = dict(ref_strategy=ref_view_strategy, amp=self.amp, use_ray_pose=self.use_ray_pose)
        contents = [None] * n
        z_est = np.full(n, np.nan)
        passes, res_list = 1, [spec.res] * n
        time_s = 0.0

        if spec.arm in ("unzoomed", "native"):
            batch, affs = released_batch(photos, spec.res)
            out = forward_views(self.model, batch, self.device, **kw)
        elif spec.arm in ("canon_oracle", "canonpad_oracle"):
            res_list = canonical_resolutions(kappa_from_captures(caps), spec.res, spec.ref, 0.0,
                                             spec.min_res, spec.max_res)
            if spec.arm == "canon_oracle":
                views, affs = canonical_views(photos, res_list)
                out = forward_views(self.model, views, self.device, **kw)
            else:
                batch, affs, contents = padded_views(photos, res_list, common_canvas(photo_hws[0], spec.res))
                out = forward_views(self.model, batch, self.device, **kw)
        else:   # canon_self: zoom estimated from DA3's own first pass
            batch, affs = released_batch(photos, spec.res)
            out = forward_views(self.model, batch, self.device, **kw)
            kappa = kappa_from_prediction(out["K"], affs, photo_hws)
            if np.isfinite(kappa).all() and (kappa > 0).all():
                z_est = kappa / reference_kappa(kappa, spec.ref)
                res_list = canonical_resolutions(kappa, spec.res, spec.ref, spec.deadband, spec.min_res, spec.max_res)
            # else: no usable focal lengths, nothing to plan: the released pass is kept
            if any(r != spec.res for r in res_list):
                time_s += out["time_s"]
                views, affs = canonical_views(photos, res_list)
                out = forward_views(self.model, views, self.device, **kw)
                passes = 2
        time_s += out["time_s"]

        # ---- back to the frame of the dataset image, at the export size
        depth, conf, Ks = [], [], []
        for s in range(n):
            A = AxisAffine.resize(export_hw, orig_hws[s]).then(caps[s].affine()).then(affs[s])   # export -> processed
            depth.append(warp_to(out["depth"][s], A, export_hw, contents[s]))
            conf.append(warp_to(out["conf"][s], A, export_hw, contents[s]))
            Ks.append(out["K"][s] if A.is_identity() else transform_K(out["K"][s], A.inverse()))
        pred = {"depth": np.stack(depth).astype(np.float32, copy=False),
                "conf": np.stack(conf).astype(np.float32, copy=False),
                "extrinsics": np.asarray(out["w2c"], np.float32),
                "intrinsics": np.stack(Ks).astype(np.float32),
                "images": None}
        meta = {"arm": spec.arm, "spec": spec.to_text(), "protocol": spec.protocol_input,
                "kinds": np.asarray(kinds, np.int32), "zoom": np.array([c.zoom for c in caps], np.float64),
                "orig_hw": np.asarray(orig_hws, np.int32), "photo_hw": np.asarray(photo_hws, np.int32),
                "view_hw": np.asarray(out["hw"], np.int32), "res": np.asarray(res_list, np.int32),
                "export_hw": np.asarray(export_hw, np.int32), "z_est": z_est, "passes": passes,
                "tokens": patch_tokens(out["hw"]), "time_s": float(time_s),
                "ref_index": -1 if out["ref_index"] is None else int(out["ref_index"]),
                "n_zoom": int((np.asarray(kinds) != KIND_NORMAL).sum())}
        return {"pred": pred, "meta": meta}

    def predict(self, image_files, extrinsics=None, intrinsics=None, ref_view_strategy="first"):
        if extrinsics is not None:
            raise NotImplementedError("the mixed-lens protocol is unposed only (modes pose, recon_unposed)")
        r = self.run([read_rgb(f) for f in image_files], ref_view_strategy)
        self.timings.append((len(image_files), r["meta"]["time_s"]))
        self._meta = r["meta"]
        self.stats.append({k: r["meta"][k] for k in ("passes", "tokens", "time_s", "n_zoom")})
        return r["pred"]

    def inference(self, image, extrinsics=None, intrinsics=None, export_dir=None, **kw):
        pred = super().inference(image, extrinsics, intrinsics, export_dir=export_dir, **kw)
        if export_dir is not None and self._meta is not None:
            path = os.path.join(export_dir, "exports", META_FILE)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(path, **{k: np.asarray(v) for k, v in self._meta.items()})
        self._meta = None
        return pred

    def summary(self) -> dict:
        if not self.stats:
            return {}
        out = {k: float(np.mean([s[k] for s in self.stats])) for k in ("passes", "tokens", "time_s", "n_zoom")}
        out["scenes"] = len(self.stats)
        return out
