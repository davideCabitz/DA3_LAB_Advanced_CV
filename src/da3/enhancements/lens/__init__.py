"""Enhancement #4: lens-consistent DA3 (docs/LENS.md).

DA3 judges distance partly from how large things look in patch units (finding of enhancement #2,
reports/enhancement_2_foveated.md §4.7). In a photo set taken with different lenses (zoom, phone
telephoto / ultrawide) that cue differs between views, so their depth scales may disagree. The fix,
training-free: give every view the same angular resolution (pixels per degree) by processing it at
its own resolution, in one pass, with the mixed-resolution routing of enhancement #2; mixed
orientations keep their native aspect ratio (no centre-crop to a common size).
"""

from .detail import apply_scale_shift, fit_scale_shift
from .geometry import (
    KIND_NAMES,
    KIND_NORMAL,
    KIND_PORTRAIT,
    KIND_ZOOM,
    AxisAffine,
    Capture,
    identity_capture,
    portrait_capture,
    transform_K,
    warp_to,
    zoom_capture,
)
from .plan import (
    ARMS,
    LensSpec,
    assign_kinds,
    canonical_resolutions,
    captures_for,
    exif_focal_px,
    kappa_from_captures,
    reference_kappa,
)

__all__ = [
    "ARMS", "KIND_NAMES", "KIND_NORMAL", "KIND_PORTRAIT", "KIND_ZOOM", "AxisAffine", "Capture", "LensSpec",
    "apply_scale_shift", "assign_kinds", "canonical_resolutions", "captures_for", "exif_focal_px", "fit_scale_shift",
    "identity_capture",
    "kappa_from_captures", "portrait_capture", "reference_kappa", "transform_K", "warp_to", "zoom_capture",
]
