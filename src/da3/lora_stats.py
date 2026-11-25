"""Statistics of the pre-registered LoRA analysis (docs/LORA_ANALYSIS_PLAN.md, configs/lora_analysis.yaml).

* Unit = scene / drive; paired arms; per-unit effect averaged over training seeds first.
* Estimation: percentile cluster bootstrap (stratified by dataset when pooled), 10k resamples, seed 0.
* Tests: paired sign-flip permutation test on the estimand (exact up to 2^16 patterns, else Monte
  Carlo); ratio estimands are linearised per unit: mean(B)/mean(A) < c  <=>  mean(B - c A) < 0.
* Decisions: superiority = PASS / FAIL / NULL / INCONCLUSIVE; non-inferiority = PASS / FAIL / INCONCLUSIVE.
* Multiplicity: Holm over a family; the intersection-union test needs no correction.
"""

from __future__ import annotations

import itertools

import numpy as np

PASS, FAIL, NULL, INCONCLUSIVE, NOT_EVALUATED = "PASS", "FAIL", "NULL", "INCONCLUSIVE", "NOT EVALUATED"


# ----------------------------------------------------------------------------- weights / estimands
def unit_weights(groups) -> np.ndarray:
    """Equal weight per group (dataset), equal weight per unit inside it; sums to 1."""
    groups = np.asarray(groups)
    w = np.zeros(len(groups))
    names = list(dict.fromkeys(groups.tolist()))
    for g in names:
        m = groups == g
        w[m] = 1.0 / (len(names) * m.sum())
    return w


def _strata_resample(groups, n, rng):
    groups = np.asarray(groups)
    idx = []
    for g in dict.fromkeys(groups.tolist()):
        members = np.flatnonzero(groups == g)
        idx.append(members[rng.integers(0, len(members), size=(n, len(members)))])
    return np.concatenate(idx, axis=1)


def bootstrap_ci(stat, data: dict, groups, n=10000, seed=0, level=0.95):
    """stat(dict of arrays indexed by unit, groups) -> scalar; resampling units within groups."""
    groups = np.asarray(groups)
    est = float(stat({k: v for k, v in data.items()}, groups))
    if len(groups) < 2:
        return est, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = _strata_resample(groups, n, rng)
    boots = np.array([stat({k: v[i] for k, v in data.items()}, groups[i]) for i in idx])
    boots = boots[np.isfinite(boots)]
    a = 100 * (1 - level) / 2
    lo, hi = np.percentile(boots, [a, 100 - a])
    return est, float(lo), float(hi)


def weighted_mean(x, groups):
    return float(np.sum(unit_weights(groups) * np.asarray(x)))


def ratio_stat(d, g):
    w = unit_weights(g)
    return float(np.sum(w * d["B"]) / np.sum(w * d["A"]))


def diff_stat(d, g):
    return weighted_mean(d["B"] - d["A"], g)


# ----------------------------------------------------------------------------- permutation test
def sign_flip_p(e, groups=None, alternative="less", exact_max_n=16, mc_n=100000, seed=0) -> float:
    """One-sided paired sign-flip test of H0: E[e] = 0 (symmetric) with statistic sum(w e)."""
    e = np.asarray(e, float)
    w = unit_weights(groups if groups is not None else np.zeros(len(e)))
    obs = float(np.sum(w * e))
    n = len(e)
    if n == 0:
        return float("nan")
    if n <= exact_max_n:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
    else:
        signs = np.random.default_rng(seed).choice([1.0, -1.0], size=(mc_n, n))
    null = signs @ (w * e)
    p = np.mean(null <= obs + 1e-12) if alternative == "less" else np.mean(null >= obs - 1e-12)
    return float(p)


def holm(pvals: dict, alpha: float) -> dict:
    """Holm step-down; returns {name: (adjusted p, rejected)}."""
    items = sorted((p, k) for k, p in pvals.items() if np.isfinite(p))
    m, out, running, stop = len(items), {}, 0.0, False
    for i, (p, k) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        rej = (not stop) and p <= alpha / (m - i)
        stop = stop or not rej
        out[k] = (adj, rej)
    for k, p in pvals.items():
        out.setdefault(k, (float("nan"), False))
    return out


# ----------------------------------------------------------------------------- decisions
def superiority_ratio(A, B, groups=None, alpha=0.025, eq_margin=0.03, lower_is_better=True, n_boot=10000, seed=0) -> dict:
    """mean(B)/mean(A) with paired units. PASS: test p < alpha and CI on the good side of 1; FAIL: CI on the
    bad side; NULL: CI inside [1 - eq, 1 + eq]; else INCONCLUSIVE. Decision fields are filled by `decide`."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    ok = np.isfinite(A) & np.isfinite(B)
    A, B = A[ok], B[ok]
    g = np.zeros(len(A)) if groups is None else np.asarray(groups)[ok]
    est, lo, hi = bootstrap_ci(ratio_stat, {"A": A, "B": B}, g, n=n_boot, seed=seed)
    e = B - A
    p = sign_flip_p(e, g, "less" if lower_is_better else "greater", seed=seed)
    d = B - A
    return {"n": int(len(A)), "estimate": est, "ci_lo": lo, "ci_hi": hi, "p": p,
            "p_wilcoxon": _wilcoxon(d, lower_is_better), "mean_diff": float(np.mean(d)) if len(d) else float("nan"),
            "d_z": float(np.mean(d) / np.std(d, ddof=1)) if len(d) > 1 and np.std(d, ddof=1) > 0 else float("nan"),
            "frac_improved": float(np.mean(d < 0 if lower_is_better else d > 0)) if len(d) else float("nan"),
            "lower_is_better": lower_is_better, "eq_margin": eq_margin}


def decide_superiority(r: dict, alpha: float) -> str:
    if r["n"] < 2 or not np.isfinite(r["ci_lo"]):
        return INCONCLUSIVE
    good_lo, good_hi = (r["ci_hi"] < 1.0, r["ci_lo"] > 1.0) if r["lower_is_better"] else (r["ci_lo"] > 1.0, r["ci_hi"] < 1.0)
    if r["p"] < alpha and good_lo:
        return PASS
    if good_hi:
        return FAIL
    if 1 - r["eq_margin"] <= r["ci_lo"] and r["ci_hi"] <= 1 + r["eq_margin"]:
        return NULL
    return INCONCLUSIVE


def non_inferiority(A, B, groups, margin_abs=None, margin_rel=None, higher_is_better=True, alpha=0.025,
                    n_boot=10000, seed=0) -> dict:
    """Absolute margin (points, higher better): H0 Δ <= -m, test on d + m. Relative margin (lower better):
    H0 mean(B)/mean(A) >= 1 + m, test on B - (1 + m) A."""
    A, B, groups = np.asarray(A, float), np.asarray(B, float), np.asarray(groups)
    ok = np.isfinite(A) & np.isfinite(B)
    A, B, groups = A[ok], B[ok], groups[ok]
    if len(A) == 0:
        return {"decision": NOT_EVALUATED, "n": 0}
    if margin_abs is not None:
        est, lo, hi = bootstrap_ci(diff_stat, {"A": A, "B": B}, groups, n=n_boot, seed=seed)
        sgn = 1 if higher_is_better else -1
        p = sign_flip_p(sgn * (B - A) + margin_abs, groups, "greater", seed=seed)
        bound_ok = (lo > -margin_abs) if higher_is_better else (hi < margin_abs)
        bound_bad = (hi < -margin_abs) if higher_is_better else (lo > margin_abs)
        kind, margin = "difference", margin_abs
    else:
        est, lo, hi = bootstrap_ci(ratio_stat, {"A": A, "B": B}, groups, n=n_boot, seed=seed)
        p = sign_flip_p(B - (1 + margin_rel) * A, groups, "less", seed=seed)
        bound_ok, bound_bad = hi < 1 + margin_rel, lo > 1 + margin_rel
        kind, margin = "ratio", margin_rel
    dec = PASS if (p < alpha and bound_ok) else FAIL if bound_bad else INCONCLUSIVE
    return {"decision": dec, "n": int(len(A)), "datasets": sorted(set(groups.tolist())), "kind": kind,
            "margin": margin, "estimate": est, "ci_lo": lo, "ci_hi": hi, "p": p}


def _wilcoxon(d, lower_is_better):
    try:
        from scipy.stats import wilcoxon

        d = np.asarray(d)[np.asarray(d) != 0]
        if len(d) < 2:
            return float("nan")
        return float(wilcoxon(d, alternative="less" if lower_is_better else "greater").pvalue)
    except Exception:
        return float("nan")
