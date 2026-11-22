"""Enhancement #1: DA3 Self-Audit (docs/da3_enhancement_report.md §4.2).

Training-free reliability for a frozen DA3, from its redundant outputs:
* signals.py: per-view signals (head-vs-ray gap, ray-origin spread, pinhole residual, confidence,
  cross-view reprojection, E_c fixed-point residual) and per-view GT error labels;
* models.py: small ridge model + leave-one-dataset-out protocol (fitted on other datasets only);
* pose_sources.py: per-view choice / fusion of camera-head and ray-map poses;
* triple_collocation.py: per-scene, GT-free calibration of the depth confidence (sigma(c));
* metrics.py: AUROC, Spearman, risk-coverage, interval coverage, NLL, AUSE;
* dump.py: extra outputs saved during the GPU run (plugged into the benchmark adapter).
The analysis that ties these together is scripts/self_audit_eval.py (notebook 05_self_audit).
"""

from .dump import AuditDumper
from .signals import BASELINE_SCORE, EC_FEATURES, FEATURES, compute_signals, per_view_pose_error

__all__ = ["AuditDumper", "BASELINE_SCORE", "EC_FEATURES", "FEATURES", "compute_signals", "per_view_pose_error"]
