"""Enhancement #2: Foveated DA3, mixed per-view resolutions in one pass (docs/da3_enhancement_report.md §4.3).

Training-free, on the frozen released weights:
* routing.py: one DA3 forward pass in which every view keeps its own resolution (focus views at
  504 px, context views at 252 px, ...); bit-identical to DA3.forward when all views share one
  resolution; optional attention-mass balancing of the cross-view blocks;
* plan.py: focus-view selection, per-view resolutions, token / FLOP budgets, equal-budget resolution;
* adapter.py: the benchmark adapter for the released harness (mixed-resolution runs on the Sec. 6
  protocol, plus per-scene efficiency meta for every run);
* zoom.py: region level (H4): exact virtual zoom cameras (pure rotation) warped from the
  full-resolution photo, given to DA3 through E_c together with the scene views;
* metrics.py: depth errors, pose AUC on view subsets (focus / context), region metrics.
The experiments that tie these together are in scripts/foveated_eval.py (notebook 06_foveated).
"""

from .plan import FoveationSpec, backbone_cost, equal_budget_res, processed_hw, select_focus, view_resolutions
from .routing import cameras_from_head, cameras_from_rays, foveated_forward, resize_maps, scale_K

__all__ = [
    "FoveationSpec", "backbone_cost", "equal_budget_res", "processed_hw", "select_focus", "view_resolutions",
    "cameras_from_head", "cameras_from_rays", "foveated_forward", "resize_maps", "scale_K",
]
