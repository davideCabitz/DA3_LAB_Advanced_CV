"""Preprocessing and forward passes of enhancement #4 (lens-consistent DA3), on in-memory photos.

* `released_batch`: the released preprocessing (longest side -> res, sides to multiples of 14, then a
  centre-crop of every view to the smallest common size), exactly as `io_utils.load_views` does it
  for files, plus the affine photo -> processed frame of every view.
* `canonical_views`: every photo at its own longest side r_i, no common crop (native aspect ratio).
* `padded_views`: every photo at r_i, pasted into the centre of one common canvas (diagnostic arm).
* `forward_views`: one DA3 pass over views of one size (the original `DA3.forward`, bit-identical to
  the benchmark adapter) or of several sizes (the mixed-resolution routing of enhancement #2), with
  camera-head (or ray-map) cameras in every view's own processed pixels.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from da3.io_utils import preprocess_like_official, target_size, to_model_input

from .geometry import MEAN_RGB, AxisAffine


def released_batch(photos, res: int):
    """(uint8 (N, H, W, 3), [photo -> processed AxisAffine]) with the released preprocessing."""
    proc = [preprocess_like_official(p, res) for p in photos]
    mh, mw = min(o.shape[0] for o in proc), min(o.shape[1] for o in proc)
    out, affs = [], []
    for p, o in zip(photos, proc):
        y0, x0 = int(round((o.shape[0] - mh) / 2.0)), int(round((o.shape[1] - mw) / 2.0))
        out.append(o[y0:][:mh, x0:][:, :mw])      # same slicing as io_utils.load_views
        affs.append(AxisAffine.resize(p.shape[:2], o.shape[:2]).then(AxisAffine.crop(x0, y0)))
    return np.stack(out), affs


def canonical_views(photos, res_list):
    """([uint8 (h_i, w_i, 3)], [photo -> processed AxisAffine]): photo i at longest side res_list[i]."""
    out, affs = [], []
    for p, r in zip(photos, res_list):
        o = preprocess_like_official(p, int(r))
        out.append(o)
        affs.append(AxisAffine.resize(p.shape[:2], o.shape[:2]))
    return out, affs


def padded_views(photos, res_list, canvas_hw):
    """Every photo at res_list[i], centred in a canvas of size canvas_hw filled with the ImageNet mean.
    Returns (uint8 (N, Hc, Wc, 3), [photo -> canvas AxisAffine], [content mask (Hc, Wc), or None when
    the view fills the canvas])."""
    Hc, Wc = int(canvas_hw[0]), int(canvas_hw[1])
    out, affs, masks = [], [], []
    for p, r in zip(photos, res_list):
        o = preprocess_like_official(p, int(r))
        h, w = o.shape[:2]
        if h > Hc or w > Wc:
            raise ValueError(f"padded arm: a view at {int(r)} px ({h}x{w}) does not fit the {Hc}x{Wc} canvas "
                             "(only defined when every canonical resolution is <= res)")
        py, px = (Hc - h) // 2, (Wc - w) // 2
        canvas = np.empty((Hc, Wc, 3), np.uint8)
        canvas[:] = np.asarray(MEAN_RGB, np.uint8)
        canvas[py:py + h, px:px + w] = o
        if (h, w) == (Hc, Wc):
            m = None                       # fills the canvas: no padding, the export is an exact copy
        else:
            m = np.zeros((Hc, Wc), np.float32)
            m[py:py + h, px:px + w] = 1.0
        out.append(canvas)
        affs.append(AxisAffine.resize(p.shape[:2], (h, w)).then(AxisAffine.pad(px, py)))
        masks.append(m)
    return np.stack(out), affs, masks


def common_canvas(photo_hw, res: int):
    return target_size(int(photo_hw[0]), int(photo_hw[1]), int(res))


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def forward_views(model, imgs, device, ref_strategy: str = "first", amp: bool = True, use_ray_pose: bool = False,
                  keep_ray: bool = False) -> dict:
    """One DA3 pass over uint8 RGB views (a stacked (N, H, W, 3) array or a list of (h_i, w_i, 3)).

    Returns {"depth": [N maps], "conf": [N maps], "ray": [N maps] | None, "w2c": (N, 3, 4),
    "K": (N, 3, 3) in every view's own processed pixels, "hw": [(h_i, w_i)], "ref_index", "time_s",
    "mixed": bool}. Equal sizes run `model(x)` on the stacked batch, the code path of the benchmark
    adapter (bit-identical outputs); mixed sizes run `foveated_forward` (enhancement #2)."""
    from da3.model import DA3

    device = torch.device(device)
    hws = [tuple(int(t) for t in im.shape[:2]) for im in imgs]
    amp = amp and device.type == "cuda"
    keep = ("depth", "depth_conf", "ray", "ray_conf") if (keep_ray or use_ray_pose) else ("depth", "depth_conf")
    t0 = time.time()
    if len(set(hws)) == 1:
        x = to_model_input(np.stack(list(imgs)) if not isinstance(imgs, np.ndarray) else imgs).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out = model(x, ref_strategy=ref_strategy)
        hw = hws[0]
        if use_ray_pose:
            w2c, K = DA3.cameras_from_rays(out, hw)
        else:
            w2c, K = DA3.cameras_from_head(out["pose_enc"], hw)
        _sync(device)
        depth = out["depth"][0].float().cpu().numpy()
        conf = out["depth_conf"][0].float().cpu().numpy()
        ray = out["ray"][0].float().cpu().numpy() if "ray" in keep else None
        ref = out.get("ref_index")
        res = {"depth": [depth[i] for i in range(len(hws))], "conf": [conf[i] for i in range(len(hws))],
               "ray": None if ray is None else [ray[i] for i in range(len(hws))],
               "w2c": w2c[0].float().cpu().numpy(), "K": K[0].float().cpu().numpy(),
               "ref_index": None if ref is None else int(ref[0]), "mixed": False}
    else:
        from da3.enhancements.foveated import foveated_forward
        from da3.enhancements.foveated.routing import cameras_from_head, cameras_from_rays

        views = [to_model_input(np.asarray(im)[None])[0, 0].to(device) for im in imgs]
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out = foveated_forward(model, views, ref_strategy=ref_strategy, keep=keep)
        if use_ray_pose:
            w2c, K = cameras_from_rays(out["ray"], hws)
        else:
            w2c, K = cameras_from_head(out["pose_enc"], hws)
        _sync(device)
        res = {"depth": [d.float().cpu().numpy() for d in out["depth"]],
               "conf": [c.float().cpu().numpy() for c in out["depth_conf"]],
               "ray": [r.float().cpu().numpy() for r in out["ray"]] if "ray" in out else None,
               "w2c": w2c.float().cpu().numpy(), "K": K.float().cpu().numpy(),
               "ref_index": out.get("ref_index"), "mixed": True}
    res["hw"] = hws
    res["time_s"] = time.time() - t0
    return res
