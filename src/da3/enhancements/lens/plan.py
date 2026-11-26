"""Protocol, arms and the angular-resolution plan of enhancement #4 (lens-consistent DA3).

docs/LENS.md. The *mixed-lens protocol* turns a Sec. 6 scene into a photo set taken with two lenses:
a share `fraction` of the views (never view 0, the reference of the paper protocol) is zoomed by `zoom`
(centred crop of 1/z of each side resized back to the full size, i.e. focal length x z). Arms:

* `unzoomed`        no zoom at all: the ceiling (the same scenes and views as the original DA3);
* `native`          the zoomed set as the released DA3 processes it: every view at `res` on one grid,
                    so zoomed views have z x finer angular resolution (pixels per degree);
* `canon_oracle`    P1 with the true zoom: every view at the resolution that gives all views the same
                    angular resolution (zoomed views on a smaller grid), one pass (mixed-resolution
                    routing of enhancement #2, bit-identical at equal resolution);
* `canonpad_oracle` diagnostic: like canon_oracle, but every view padded back to the common grid
                    (same grid AND same angular resolution): separates the two hypotheses of docs/LENS.md;
* `canon_self`      P1 deployable: the zoom of every view is estimated from DA3's own first pass
                    (camera-head focal length); views within the dead band keep `res`. If no view is
                    outside the dead band, the first pass is returned (bit-identical to `native`).

Angular resolution of a view processed at longest side r: f_photo * r / L_photo pixels per radian at
the image centre (f_photo = focal length in photo pixels, L_photo = photo longest side). With
kappa = f_photo / L_photo, equal angular resolution means r_i = res * kappa_ref / kappa_i.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .geometry import KIND_NORMAL, KIND_PORTRAIT, KIND_ZOOM, identity_capture, portrait_capture, zoom_capture

ARMS = ("unzoomed", "native", "canon_oracle", "canonpad_oracle", "canon_self")
REFS = ("median", "min", "max")


@dataclass(frozen=True)
class LensSpec:
    """One arm of one mixed-lens protocol (parsed from 'arm=native,zoom=2,fraction=0.3')."""

    arm: str = "native"
    zoom: float = 2.0          # focal multiplier of the zoomed views
    fraction: float = 0.3      # share of zoomed views (view 0 never)
    portrait: float = 0.0      # share of portrait-cropped views (orientation; not used by default)
    res: int = 504             # longest side of the released preprocessing (the paper's 504)
    deadband: float = 0.15     # canon_self: views within +-15% of the reference focal keep `res`
    min_res: int = 112         # canonical resolutions are clipped to [min_res, max_res]
    max_res: int = 1008
    ref: str = "median"        # reference view for the angular resolution: lower median of kappa

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {self.arm!r}")
        if self.zoom <= 1.0:
            raise ValueError(f"zoom must be > 1, got {self.zoom}")
        if not 0.0 <= self.fraction < 1.0 or not 0.0 <= self.portrait < 1.0 or self.fraction + self.portrait >= 1.0:
            raise ValueError("fraction and portrait must be in [0, 1) with fraction + portrait < 1")
        if self.ref not in REFS:
            raise ValueError(f"ref must be one of {REFS}, got {self.ref!r}")

    # ---- names
    @property
    def protocol_id(self) -> str:
        """File-safe name of the protocol (all arms of one protocol share it)."""
        s = f"z{self.zoom:g}_f{int(round(100 * self.fraction))}".replace(".", "p")
        if self.portrait:
            s += f"_p{int(round(100 * self.portrait))}"
        return s

    @property
    def protocol_input(self) -> str:
        """Value of the result records' protocol 'input' field: records of one protocol are compared
        with each other only (da3.compare.protocol_key)."""
        s = f"lens:zoom={self.zoom:g},fraction={self.fraction:g}"
        if self.portrait:
            s += f",portrait={self.portrait:g}"
        if self.res != 504:
            s += f",res={self.res}"
        return s

    @property
    def name(self) -> str:
        return f"{self.protocol_id}__{self.arm}"

    @classmethod
    def parse(cls, text: str) -> "LensSpec":
        types = {f.name: f.type for f in fields(cls)}
        kw = {}
        for item in filter(None, (t.strip() for t in text.split(","))):
            key, sep, val = item.partition("=")
            key, val = key.strip(), val.strip()
            if not sep or key not in types:
                raise ValueError(f"bad lens spec item {item!r} in {text!r}; fields: {sorted(types)}")
            if key in ("zoom", "fraction", "portrait", "deadband"):
                kw[key] = float(val)
            elif key in ("res", "min_res", "max_res"):
                kw[key] = int(val)
            else:
                kw[key] = val
        return cls(**kw)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_text(self) -> str:
        return ",".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.to_dict().items())


# ----------------------------------------------------------------------------- which views are zoomed
def _evenly(cands: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or len(cands) == 0:
        return np.zeros(0, int)
    k = min(k, len(cands))
    return cands[np.unique(np.round(np.linspace(0, len(cands) - 1, k)).astype(int))]


def assign_kinds(n: int, fraction: float, portrait: float = 0.0) -> np.ndarray:
    """Kind of every view (KIND_NORMAL / ZOOM / PORTRAIT): round(fraction * n) zoomed views evenly
    spaced among views 1..n-1 (view 0 = the reference stays normal), then round(portrait * n)
    portrait views evenly spaced among the remaining ones. Deterministic."""
    kinds = np.full(n, KIND_NORMAL, int)
    if n < 2:
        return kinds
    zi = _evenly(np.arange(1, n), int(round(fraction * n)))
    kinds[zi] = KIND_ZOOM
    if portrait > 0:
        rest = np.array([i for i in range(1, n) if kinds[i] == KIND_NORMAL], int)
        kinds[_evenly(rest, int(round(portrait * n)))] = KIND_PORTRAIT
    return kinds


def captures_for(spec: LensSpec, hws, kinds=None) -> list:
    """Capture of every view (identity for the `unzoomed` arm)."""
    kinds = assign_kinds(len(hws), spec.fraction, spec.portrait) if kinds is None else kinds
    caps = []
    for hw, k in zip(hws, kinds):
        if spec.arm == "unzoomed" or k == KIND_NORMAL:
            caps.append(identity_capture(hw))
        elif k == KIND_ZOOM:
            caps.append(zoom_capture(hw, spec.zoom))
        else:
            caps.append(portrait_capture(hw))
    return caps


# ----------------------------------------------------------------------------- angular-resolution plan
def reference_kappa(kappa, ref: str = "median") -> float:
    k = np.sort(np.asarray(kappa, np.float64))
    if ref == "min":
        return float(k[0])
    if ref == "max":
        return float(k[-1])
    return float(k[(len(k) - 1) // 2])      # lower median: always the value of an actual view


def canonical_resolutions(kappa, res: int = 504, ref: str = "median", deadband: float = 0.0,
                          min_res: int = 112, max_res: int = 1008) -> list[int]:
    """Longest-side processing resolution of every view so that all views share the angular
    resolution of the reference view processed at `res`: r_i = res * kappa_ref / kappa_i.
    kappa_i = focal length / longest side of photo i (any common unit). Views within +-deadband of the
    reference keep `res` exactly; the others are clipped to [min_res, max_res]."""
    k = np.asarray(kappa, np.float64)
    if not np.isfinite(k).all() or (k <= 0).any():
        raise ValueError(f"focal lengths must be finite and positive, got {k}")
    k_ref = reference_kappa(k, ref)
    r = res * k_ref / k
    within = np.abs(np.log(k / k_ref)) <= np.log1p(deadband) if deadband > 0 else np.isclose(k, k_ref, rtol=1e-9)
    r = np.where(within, float(res), np.clip(np.round(r), min_res, max_res))
    return [int(v) for v in r]


def kappa_from_captures(caps) -> np.ndarray:
    """Oracle kappa of every photo: the source camera's focal length is common to all views of a scene
    (one camera), so kappa_i is proportional to focal_scale_i / longest side of photo i."""
    return np.array([c.focal_scale / max(c.size) for c in caps], np.float64)


# ----------------------------------------------------------------------------- EXIF (real photos)
FULL_FRAME_DIAGONAL_MM = 43.2666   # 36 x 24 mm


def exif_focal_px(path, hw) -> float | None:
    """Focal length in pixels of an image of oriented size hw = (h, w) from its EXIF
    FocalLengthIn35mmFilm (phones write it for every lens): f_px = f35 / 43.27 mm * image diagonal.
    None when the tag is missing (e.g. photos sent through messaging apps lose their EXIF)."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            exif = im.getexif()
            f35 = exif.get_ifd(0x8769).get(41989) or exif.get(41989)
    except Exception:
        return None
    try:
        f35 = float(f35)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f35) or f35 <= 0:
        return None
    return f35 / FULL_FRAME_DIAGONAL_MM * float(np.hypot(hw[0], hw[1]))
