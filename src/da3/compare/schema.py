"""Standard result record shared by the baseline and every enhancement.

Every run (original DA3 or an enhancement variant) is saved as one JSON file

    experiments/results/<enhancement>/<tag>.json

so that one comparison layer (da3.compare.tables / plots, notebook 09_compare_all) can put the
original DA3 next to all enhancements. Fields:

* identity: `enhancement` ("baseline", "self_audit", "foveated", "pose_distill", ...), `variant`
  (e.g. "head", "ray_official", "audit_select"), `tag` (unique run name), `setting` tag
  ([official] / [reduced-scale]), `weights`;
* protocol: datasets, scenes, max_frames, modes, ref_view_strategy, and optionally `input` (how the
  input images were changed before the model saw them, e.g. "lens:zoom=2,fraction=0.3" for the
  mixed-lens protocol of enhancement #4; absent = the dataset images as they are). Two records are
  only compared when their protocol key and weights match (same data, same scenes, same frame
  sampling, same input);
* `sec6`: means of the released harness, keyed "<data>_<mode>" (e.g. "hiroom_pose" ->
  {"auc03": ..., "auc30": ...}, "hiroom_recon_unposed" -> {"fscore": ..., "overall": ...});
* `sec6_per_scene`: same, per scene (optional);
* `extra`: enhancement-specific metric groups, {group: {dataset_or_"all": {metric: value}}};
* `baseline_tag`: tag of the original-DA3 run this record is compared with (optional);
* `adapter`: for LoRA runs (setting "official+LoRA"), the adapter file, its SHA-256, config, seed, arm,
  step and training drives; `weights` stays "official:<preset>" so the record is paired with the
  released run (optional);
* bookkeeping: commit, env, time, notes, files (figures etc.).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

SCHEMA = "da3-result/1"
REQUIRED = ("schema", "enhancement", "variant", "tag", "setting", "weights", "protocol")


def protocol(datasets, scenes="all", max_frames=100, modes=(), ref_view_strategy="first",
             input: str | None = None) -> dict:
    p = {"datasets": sorted(datasets), "scenes": scenes if scenes == "all" else sorted(scenes),
         "max_frames": int(max_frames), "modes": sorted(modes), "ref_view_strategy": ref_view_strategy}
    if input:
        p["input"] = str(input)
    return p


def make_record(*, enhancement: str, variant: str, tag: str, setting: str, weights: str, protocol: dict,
                sec6: dict | None = None, sec6_per_scene: dict | None = None, extra: dict | None = None,
                baseline_tag: str | None = None, commit: str = "unknown", env: dict | None = None,
                notes: str = "", files: list[str] | None = None, adapter: dict | None = None) -> dict:
    rec = {
        "schema": SCHEMA, "enhancement": enhancement, "variant": variant, "tag": tag,
        "setting": setting, "weights": weights, "protocol": protocol,
        "sec6": sec6 or {}, "sec6_per_scene": sec6_per_scene or {}, "extra": extra or {},
        "baseline_tag": baseline_tag, "commit": commit, "env": env or {},
        "time": time.strftime("%Y-%m-%d %H:%M:%S"), "notes": notes, "files": files or [],
    }
    if adapter:
        rec["adapter"] = adapter
    return rec


def validate(record: dict) -> dict:
    missing = [k for k in REQUIRED if k not in record]
    if missing:
        raise ValueError(f"result record {record.get('tag')!r} misses {missing}")
    if record["schema"] != SCHEMA:
        raise ValueError(f"unknown schema {record['schema']!r} (expected {SCHEMA})")
    return record


def save_record(record: dict, root: str | Path) -> Path:
    validate(record)
    path = Path(root) / record["enhancement"] / f"{record['tag']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")
    return path


def load_records(*roots: str | Path) -> list[dict]:
    """All valid records under the given folders (recursively); later duplicates of a tag win."""
    by_tag = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(rec, dict) and rec.get("schema") == SCHEMA:
                rec["_path"] = str(p)
                by_tag[(rec["enhancement"], rec["tag"])] = validate(rec)
    return list(by_tag.values())


def protocol_key(record: dict) -> tuple:
    p = record["protocol"]
    scenes = p["scenes"] if p["scenes"] == "all" else tuple(p["scenes"])
    return (record["weights"], tuple(p["datasets"]), scenes, p["max_frames"], p.get("ref_view_strategy"),
            p.get("input"))
