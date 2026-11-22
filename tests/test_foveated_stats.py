"""Enhancement #2: statistics and the pre-registered decision logic, on synthetic data with known answers."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from da3.enhancements.foveated import stats as S
from da3.enhancements.foveated.analysis import load_plan, margins, probe_gates, zoom_gates

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "configs" / "foveated_analysis.yaml"


def test_bootstrap_ci_has_nominal_coverage():
    rng = np.random.default_rng(0)
    hits = 0
    for k in range(150):
        x = rng.normal(1.0, 0.2, 30)
        _, lo, hi = S.bootstrap(x, np.mean, n=400, seed=k)
        hits += lo <= 1.0 <= hi
    assert 0.86 <= hits / 150 <= 0.99


def test_paired_ratio_and_difference():
    b = np.array([1.0, 2.0, 3.0, 4.0])
    est, lo, hi = S.ratio_of_means(1.05 * b, b, n=200)
    assert np.isclose(est, 1.05) and np.isclose(lo, 1.05) and np.isclose(hi, 1.05)
    est, lo, hi = S.mean_difference(b + 2, b, n=200)
    assert np.isclose(est, 2) and np.isclose(lo, 2) and np.isclose(hi, 2)
    assert np.isnan(S.ratio_of_means([1.0], [1.0])[1])          # one cluster: no CI


def test_three_way_decisions():
    assert S.at_most((0.98, 1.02), 1.03) == S.PASS
    assert S.at_most((1.01, 1.05), 1.03) == S.INCONCLUSIVE
    assert S.at_most((1.04, 1.06), 1.03) == S.FAIL
    assert S.at_least((0.98, 1.10), 0.97) == S.PASS
    assert S.at_least((0.90, 0.99), 0.97) == S.INCONCLUSIVE
    assert S.at_least((0.80, 0.95), 0.97) == S.FAIL
    assert S.above((1.10, 1.30), 1.03) == S.PASS and S.above((0.99, 1.02), 1.03) == S.FAIL
    assert S.at_most((np.nan, 1.0), 1.03) == S.NA
    assert S.combine_and(S.PASS, S.PASS) == S.PASS
    assert S.combine_and(S.PASS, S.INCONCLUSIVE) == S.INCONCLUSIVE
    assert S.combine_and(S.INCONCLUSIVE, S.FAIL) == S.FAIL


def test_critical_margins_reproduce_the_decision():
    ci = (1.012, 1.041)
    c = S.critical_ratio_at_most(ci)
    for m in np.linspace(0, 0.08, 33):
        expected = S.at_most(ci, 1 + m)
        got = S.PASS if m >= c["pass_from"] else (S.FAIL if m < c["fail_below"] else S.INCONCLUSIVE)
        assert got == expected, m


def test_holm_and_bonferroni():
    adj = S.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert np.isclose(adj["a"], 0.03) and np.isclose(adj["c"], 0.06) and np.isclose(adj["b"], 0.06)
    assert np.isclose(S.bonferroni_level(0.95, 2), 0.975)


def test_stratified_split_is_deterministic_and_disjoint():
    keys = [("hiroom", f"s{i}") for i in range(10)] + [("7scenes", f"t{i}") for i in range(3)]
    strata = [k[0] for k in keys]
    d1, t1 = S.stratified_split(keys, strata, 0.5, seed=0)
    d2, t2 = S.stratified_split(keys, strata, 0.5, seed=0)
    assert d1 == d2 and t1 == t2 and not set(d1) & set(t1) and len(d1) + len(t1) == 13
    assert sum(k[0] == "7scenes" for k in t1) >= 1 and sum(k[0] == "hiroom" for k in d1) == 5


# ----------------------------------------------------------------------------- pre-registered gates
def _const_pairs(n, focus, ff, cc, fc):
    E = np.full((n, n), float(cc))
    f = np.asarray(focus, bool)
    E[np.ix_(f, f)] = ff
    E[np.ix_(f, ~f)] = E[np.ix_(~f, f)] = fc
    np.fill_diagonal(E, np.nan)
    return E


def _synthetic_rows(fov_focus_factor=1.01, n_scenes=12, n=8, seed=0):
    """u252 is 20% worse than u504; the foveated focus views are `fov_focus_factor` x u504, its
    context views like u252; the focus views alone are 10% worse than with context; A/A ~ u504."""
    rng = np.random.default_rng(seed)
    focus = np.zeros(n, bool)
    focus[::2] = True
    rows = []
    for s in range(n_scenes):
        data = "hiroom" if s < 8 else "7scenes"
        for sub in range(2):
            base = rng.uniform(0.05, 0.10, n)
            jit = lambda: 1 + rng.normal(0, 0.003, n)   # noqa: E731
            cfg = {"u504": (base * jit(), _const_pairs(n, focus, 5, 5, 5), np.ones(n, bool)),
                   "u504_aa": (base * jit(), _const_pairs(n, focus, 5, 5, 5), np.ones(n, bool)),
                   "u252": (1.2 * base * jit(), _const_pairs(n, focus, 10, 10, 10), np.ones(n, bool)),
                   "u280": (1.15 * base * jit(), _const_pairs(n, focus, 9, 9, 9), np.ones(n, bool)),
                   "focus_only": (np.where(focus, 1.1 * fov_focus_factor * base, np.nan) * jit(),
                                  _const_pairs(n, focus, 6, np.nan, np.nan), focus.copy())}
            for name in ("fov50_504-252", "fov50_504-252_bal", "fov50_504-280"):
                cfg[name] = (np.where(focus, fov_focus_factor * base, 1.18 * base) * jit(),
                             _const_pairs(n, focus, 5, 9, 7), np.ones(n, bool))
            for name, (ar, E, members) in cfg.items():
                rows.append({"data": data, "scene": f"sc{s}", "subset": sub, "config": name, "n_views": n,
                             "focus": focus.tolist(), "members": members.tolist(), "abs_rel": ar.tolist(),
                             "delta1": [1.0] * n, "delta105": [1.0] * n, "abs_rel_scene": ar.tolist(),
                             "pair_err": E.tolist(), "tokens": 100})
    return rows


def test_probe_gates_pass_when_the_foveated_views_keep_full_quality():
    plan = load_plan(PLAN)
    plan["bootstrap"]["n"] = 500
    out = probe_gates(_synthetic_rows(1.01), plan)
    g = out["gates"]
    assert out["margins"]["depth"] == pytest.approx(0.03) and out["margins"]["pose"] == pytest.approx(0.03)
    assert g["G1 uniform-resolution curve"]["result"] == S.PASS
    assert g["G3 cross-scale probe"]["result"] == S.PASS
    assert g["G3 cross-scale probe"]["C1"]["estimate"] == pytest.approx(1.01, abs=0.005)
    assert "G3 rescue (dev selection, test confirmation)" not in g       # only when the primary does not pass
    assert g["G3 secondary (fov50_504-280)"]["C1"]["level"] == pytest.approx(0.975)


def test_probe_gates_fail_and_rescue_when_the_focus_views_degrade():
    plan = load_plan(PLAN)
    plan["bootstrap"]["n"] = 500
    g = probe_gates(_synthetic_rows(1.10), plan)["gates"]
    assert g["G3 cross-scale probe"]["result"] == S.FAIL                 # 10% worse > 3% margin, tight CI
    assert g["G3 cross-scale probe"]["C1"]["result"] == S.FAIL
    rescue = g["G3 rescue (dev selection, test confirmation)"]
    assert rescue["role"] == "rescue" and rescue["result"] == S.FAIL     # the mitigations degrade too


def test_margin_is_capped_and_flagged():
    plan = load_plan(PLAN)
    nf = {"depth": {"floor": 0.10}, "pose": {"floor": 0.01}}
    m = margins(plan, nf)
    assert m["depth"] == pytest.approx(0.06) and nf["depth"]["above_cap"]
    assert m["pose"] == pytest.approx(0.03) and not nf["pose"]["above_cap"]


def test_zoom_gates():
    plan = load_plan(PLAN)
    plan["bootstrap"]["n"] = 500
    rng = np.random.default_rng(0)
    rows = []
    for s in range(6):
        for v in range(2):
            for sel in ("oracle", "random"):
                rows.append({"scene": f"sc{s}", "view": v, "box": [0, 0, 1, 1], "selector": sel, "methods": {
                    "zoom_posed_da3cams": {"abs_rel": 0.08 * (1 + rng.normal(0, 0.01)), "rv_err_head_deg": 0.4},
                    "global_896": {"abs_rel": 0.10 * (1 + rng.normal(0, 0.01))}}})
    g = zoom_gates(rows, plan)
    assert g["G2 region zoom (oracle)"]["result"] == S.PASS and g["G2 region zoom (oracle)"]["role"] == "primary"
    assert g["G2 region zoom (random)"]["role"] == "secondary"
    assert g["G2 region zoom (texture)"]["result"] == S.NA


def test_plan_is_consistent_with_the_probe_defaults():
    """The plan's configuration names must be the ones foveated_eval.py probe produces by default."""
    spec = importlib.util.spec_from_file_location("foveated_eval", REPO / "scripts" / "foveated_eval.py")
    fe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fe)
    args = fe.argparse.Namespace(uniform=[504, 378, 336, 280, 252], fraction=0.5, focus_seeds=[1, 2, 3],
                                 foveated=["context_res=252", "context_res=280", "context_res=252,balance=1"])
    names = {c[0] for c in fe.probe_configs(args)}
    p = load_plan(PLAN)["probe"]
    assert {p["hi"], p["lo"], p["aa"], p["focus_only"], p["primary"], *p["mitigations"], *p["focus_seeds"]} <= names
    subs = fe.subset_indices(1000, 16, 3)
    assert len(subs) == 3 and all(len(s) == 16 for s in subs) and not set(subs[0]) & set(subs[1])
    assert len(fe.subset_indices(20, 16, 3)) >= 1 and len(fe.subset_indices(10, 16, 3)[0]) == 10
