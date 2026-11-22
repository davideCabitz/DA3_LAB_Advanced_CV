"""Evaluation metrics for the audit: failure detection (A) and depth uncertainty (C).

Conventions: a reliability *score* is larger for LESS reliable views / pixels; errors are
non-negative (degrees for poses, |log-depth residual| for depth).
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
from scipy.stats import rankdata, spearmanr


def auroc(score, label) -> float:
    """Area under the ROC curve of `score` for the positive class `label` (Mann–Whitney, ties
    averaged). NaN when only one class is present."""
    score, label = np.asarray(score, np.float64), np.asarray(label, bool)
    keep = np.isfinite(score)
    score, label = score[keep], label[keep]
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def spearman(score, err) -> float:
    score, err = np.asarray(score, np.float64), np.asarray(err, np.float64)
    keep = np.isfinite(score) & np.isfinite(err)
    if keep.sum() < 3 or np.ptp(score[keep]) == 0 or np.ptp(err[keep]) == 0:
        return float("nan")
    return float(spearmanr(score[keep], err[keep]).statistic)


def risk_coverage(score, err, clip: float = 30.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Keep the most reliable views first (lowest score). Risk = mean clipped error of the kept
    views. Returns (coverage, risk, AURC = mean risk over all coverages)."""
    score, err = np.asarray(score, np.float64), np.minimum(np.asarray(err, np.float64), clip)
    keep = np.isfinite(score) & np.isfinite(err)
    score, err = score[keep], err[keep]
    if len(err) == 0:
        return np.zeros(0), np.zeros(0), float("nan")
    order = np.argsort(score, kind="stable")
    risk = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    cov = np.arange(1, len(err) + 1) / len(err)
    return cov, risk, float(risk.mean())


def coverage(e, sigma, p: float) -> float:
    """Fraction of residuals inside the central p-interval of N(0, sigma²)."""
    z = NormalDist().inv_cdf((1 + p) / 2)
    e, sigma = np.abs(np.asarray(e, np.float64)), np.asarray(sigma, np.float64)
    return float(np.mean(e <= z * sigma))


def gaussian_nll(e, sigma) -> float:
    e, sigma = np.asarray(e, np.float64), np.clip(np.asarray(sigma, np.float64), 1e-6, None)
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma**2) + e**2 / (2 * sigma**2)))


def ause(err, conf, steps: int = 20) -> float:
    """Area under the sparsification error, relative to the mean error: remove the least confident
    pixels first and compare with removing the largest errors first (oracle). Lower is better; it
    only depends on the ranking, so any monotone recalibration of the confidence has the same AUSE."""
    err, conf = np.abs(np.asarray(err, np.float64)), np.asarray(conf, np.float64)
    if len(err) < steps or err.mean() <= 0:
        return float("nan")
    by_conf = err[np.argsort(-conf, kind="stable")]       # most confident first
    by_err = np.sort(err)                                  # smallest error first (oracle)
    n = len(err)
    fracs = np.linspace(0, 0.95, steps)
    curve = [by_conf[: max(1, int(round(n * (1 - f))))].mean() for f in fracs]
    oracle = [by_err[: max(1, int(round(n * (1 - f))))].mean() for f in fracs]
    return float(np.mean(np.subtract(curve, oracle)) / err.mean())


def reliability_curve(e, sigma, ps=np.linspace(0.1, 0.9, 9)) -> tuple[np.ndarray, np.ndarray]:
    """Nominal vs empirical coverage of central Gaussian intervals."""
    return np.asarray(ps), np.array([coverage(e, sigma, p) for p in ps])
