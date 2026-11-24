"""Statistics for enhancement #2: scene-level uncertainty and pre-registered three-way decisions.

Design (docs/FOVEATED_ANALYSIS_PLAN.md, constants in configs/foveated_analysis.yaml):

* Unit of analysis = the scene. Views (and view subsets) of one scene are correlated, so every
  quantity is first reduced to one value per scene; uncertainty comes from a percentile bootstrap
  that resamples scenes (cluster bootstrap). Comparisons are paired: both arms on the same scenes
  and the same views.
* Decisions are three-way and read off the confidence interval (CI) of the effect:
  PASS (the CI is entirely on the good side of the threshold), FAIL (entirely on the bad side),
  INCONCLUSIVE (the CI contains the threshold). A two-sided 95% CI bound = a one-sided test at 2.5%.
* Threshold sensitivity: for a CI [lo, hi], the decision at EVERY margin follows from two numbers,
  which are reported with each gate ("PASS for margins >= x, FAIL for margins < y"), so no
  threshold has to be chosen after looking at the result.
"""

from __future__ import annotations

import numpy as np

PASS, FAIL, INCONCLUSIVE, NA = "PASS", "FAIL", "INCONCLUSIVE", "n.a."


# ----------------------------------------------------------------------------- bootstrap
def bootstrap(values, stat, n: int = 4000, seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """Percentile bootstrap over the first axis (clusters = scenes).
    values: (C,) or (C, k) array of per-cluster data (rows with a non-finite entry are dropped);
    stat: function of a (C', ...) array -> scalar. Returns (estimate, lo, hi); NaN if < 2 clusters."""
    v = np.asarray(values, np.float64)
    keep = np.isfinite(v).all(axis=tuple(range(1, v.ndim))) if v.ndim > 1 else np.isfinite(v)
    v = v[keep]
    if len(v) < 2:
        est = float(stat(v)) if len(v) else float("nan")
        return est, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    boots = np.array([stat(v[i]) for i in idx], np.float64)
    boots = boots[np.isfinite(boots)]
    a = 100.0 * (1.0 - level) / 2.0
    lo, hi = np.percentile(boots, [a, 100.0 - a])
    return float(stat(v)), float(lo), float(hi)


def ratio_of_means(test, ref, **kw) -> tuple[float, float, float]:
    """mean(test) / mean(ref) over paired clusters, with bootstrap CI (clusters resampled jointly)."""
    return bootstrap(np.stack([np.asarray(test, float), np.asarray(ref, float)], 1),
                     lambda x: x[:, 0].mean() / x[:, 1].mean(), **kw)


def mean_difference(test, ref, **kw) -> tuple[float, float, float]:
    """mean(test - ref) over paired clusters, with bootstrap CI."""
    return bootstrap(np.asarray(test, float) - np.asarray(ref, float), np.mean, **kw)


def median_ci(values_by_cluster, **kw) -> tuple[float, float, float]:
    """Median of pooled per-item values, resampling clusters (list of 1-D arrays)."""
    groups = [np.asarray(g, float)[np.isfinite(g)] for g in values_by_cluster]
    groups = [g for g in groups if len(g)]
    if not groups:
        return float("nan"), float("nan"), float("nan")
    ids = np.arange(len(groups), dtype=float)[:, None]
    return bootstrap(ids, lambda i: float(np.median(np.concatenate([groups[int(j)] for j in i[:, 0]]))), **kw)


def wilcoxon_p(test, ref) -> float:
    """Two-sided Wilcoxon signed-rank p-value of paired per-scene values (NaN if not computable)."""
    from scipy.stats import wilcoxon

    d = np.asarray(test, float) - np.asarray(ref, float)
    d = d[np.isfinite(d)]
    if len(d) < 6 or np.all(d == 0):
        return float("nan")
    return float(wilcoxon(d).pvalue)


def bonferroni_level(level: float, k: int) -> float:
    """CI level giving family-wise coverage `level` over k simultaneous intervals."""
    return 1.0 - (1.0 - level) / max(1, k)


# ----------------------------------------------------------------------------- decisions
def at_most(ci, threshold: float) -> str:
    """A lower-is-better quantity must be <= threshold (e.g. AbsRel ratio <= 1 + margin)."""
    lo, hi = ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return NA
    return PASS if hi <= threshold else (FAIL if lo > threshold else INCONCLUSIVE)


def at_least(ci, threshold: float) -> str:
    """A higher-is-better quantity must be >= threshold (e.g. AUC ratio >= 1 - margin)."""
    lo, hi = ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return NA
    return PASS if lo >= threshold else (FAIL if hi < threshold else INCONCLUSIVE)


def above(ci, threshold: float) -> str:
    """A quantity must be significantly larger than threshold (PASS), else shown not to be (FAIL)."""
    lo, hi = ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return NA
    return PASS if lo > threshold else (FAIL if hi <= threshold else INCONCLUSIVE)


def combine_and(*results) -> str:
    """Intersection of criteria: PASS only if all pass; FAIL if any fails."""
    if any(r == FAIL for r in results):
        return FAIL
    if results and all(r == PASS for r in results):
        return PASS
    if any(r == NA for r in results):
        return NA
    return INCONCLUSIVE


# Critical margins: the decision of every criterion as a function of the margin m. Ratios are
# test / reference; "pass_from" = PASS for every m >= this value, "fail_below" = FAIL for every
# m < this value, INCONCLUSIVE in between.
def critical_ratio_at_most(ci) -> dict:
    """Ratio must be <= 1 + m (lower is better, non-inferiority with margin m)."""
    lo, hi = ci
    return {"pass_from": hi - 1.0, "fail_below": lo - 1.0}


def critical_ratio_at_least(ci) -> dict:
    """Ratio must be >= 1 - m (higher is better, non-inferiority with margin m)."""
    lo, hi = ci
    return {"pass_from": 1.0 - lo, "fail_below": 1.0 - hi}


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values (NaN entries are left out)."""
    items = sorted(((k, p) for k, p in pvals.items() if np.isfinite(p)), key=lambda t: t[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return {k: out.get(k, float("nan")) for k in pvals}


# ----------------------------------------------------------------------------- splits
def stratified_split(keys, strata, dev_fraction: float = 0.5, seed: int = 0) -> tuple[list, list]:
    """Deterministic split of cluster keys into (dev, test), stratified by `strata` (one label per
    key, e.g. the dataset). Each stratum keeps at least one key in test."""
    keys, strata = list(keys), list(strata)
    rng = np.random.default_rng(seed)
    dev, test = [], []
    for s in sorted(set(strata)):
        ks = sorted(k for k, t in zip(keys, strata) if t == s)
        ks = [ks[i] for i in rng.permutation(len(ks))]
        n_dev = min(int(np.floor(dev_fraction * len(ks))), len(ks) - 1)
        dev += ks[:n_dev]
        test += ks[n_dev:]
    return sorted(dev), sorted(test)


def fmt_ci(est, lo, hi, digits=3) -> str:
    return f"{est:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
