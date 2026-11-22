"""Comparison layer: standard records, protocol grouping, differences to the original DA3."""

import pytest

from da3.compare import extra_markdown, load_records, make_record, protocol, save_record, sec6_markdown, sec6_table


def _rec(enh, variant, tag, auc3, weights="official:da3-small", scenes="all", extra=None):
    return make_record(enhancement=enh, variant=variant, tag=tag, setting="official", weights=weights,
                       protocol=protocol(["hiroom", "dtu64"], scenes, 100, ["pose"], "first"),
                       sec6={"hiroom_pose": {"auc03": auc3, "auc30": 0.75},
                             "hiroom_recon_unposed": {"fscore": 0.2, "overall": 0.1},
                             "dtu64_pose": {"auc03": 0.3, "auc30": 0.9}},
                       extra=extra)


def test_records_roundtrip_and_deltas(tmp_path):
    recs = [_rec("baseline", "head", "base", 0.10), _rec("self_audit", "audit_fuse", "fuse", 0.12),
            _rec("baseline", "head", "other_scenes", 0.5, scenes=["s1"])]
    for r in recs:
        save_record(r, tmp_path)
    loaded = load_records(tmp_path)
    assert len(loaded) == 3
    tables = sec6_table(loaded)
    assert len(tables) == 2                       # different scene sets are never compared
    t = next(t for t in tables if t["scenes"] == "all")
    assert t["reference"] == "base"
    fuse = next(r for r in t["rows"] if r["variant"] == "audit_fuse")
    assert fuse["deltas"][("hiroom", "AUC@3")] == pytest.approx(2.0)
    assert ("dtu64", "F1 w/o p.") not in fuse["values"]          # DTU-64 is pose-only
    md = sec6_markdown(loaded)
    assert "+2.0 ▲" in md


def test_extra_tables_render():
    r = _rec("self_audit", "audit_metrics", "m", 0.1,
             extra={"gates": {"G1": {"result": "PASS", "value": "2/3"}},
                    "failure_detection": {"HiRoom · Self-Audit": {"AUROC>15°": 0.8, "views": 300}}})
    md = extra_markdown([r], "self_audit", "failure_detection")
    assert "0.800" in md and "HiRoom · Self-Audit" in md
    assert "PASS" in extra_markdown([r], "self_audit", "gates")


def test_invalid_record_is_rejected(tmp_path):
    r = _rec("baseline", "head", "x", 0.1)
    del r["weights"]
    with pytest.raises(ValueError):
        save_record(r, tmp_path)
