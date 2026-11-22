"""Helpers around the released benchmark harness' on-disk layout (no import of the official package).

The harness resolves dataset paths relative to the CWD (`workspace/benchmark_dataset/<data>`), and
writes one export folder per scene:

    <work_root>/workspace/evaluation/<tag>/model_results/<data>/<scene>/<unposed|posed>/exports/
        mini_npz/results.npz   (depth, conf, extrinsics, intrinsics)
        gt_meta.npz            (sampled GT extrinsics, intrinsics, image_files)

Scene names can contain slashes (HiRoom), so scenes are recovered from the folder structure.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


def prepare_workspace(work_root: str | Path, bench_root: str | Path) -> Path:
    """Create <work_root>/workspace/benchmark_dataset -> <bench_root> (symlink, or a directory
    junction on Windows without symlink rights) and return the resolved work_root."""
    work_root = Path(work_root).resolve()
    (work_root / "workspace").mkdir(parents=True, exist_ok=True)
    link = work_root / "workspace" / "benchmark_dataset"
    if not link.exists():
        try:
            os.symlink(Path(bench_root).resolve(), link, target_is_directory=True)
        except OSError:
            subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(Path(bench_root).resolve())], check=True)
    return work_root


def evaluation_dir(work_root: str | Path, tag: str) -> Path:
    return Path(work_root) / "workspace" / "evaluation" / tag


def unposed_scenes(work_dir: str | Path, data: str, require: str = "mini_npz/results.npz") -> dict[str, Path]:
    """{scene: <.../unposed/exports>} for every scene of `data` whose exports contain `require`."""
    root = Path(work_dir) / "model_results" / data
    out = {}
    if not root.exists():
        return out
    for exports in sorted(root.glob("**/unposed/exports")):
        if (exports / require).exists():
            scene = exports.parent.parent.relative_to(root).as_posix()
            out[scene] = exports
    return out


def load_scene_exports(exports: str | Path) -> dict:
    """Read results.npz + gt_meta.npz (+ audit_dump.npz when present) of one scene."""
    exports = Path(exports)
    res = dict(np.load(exports / "mini_npz" / "results.npz"))
    meta = np.load(exports / "gt_meta.npz", allow_pickle=True)
    out = {"results": res, "gt_extrinsics": meta["extrinsics"], "gt_intrinsics": meta["intrinsics"],
           "image_files": [str(p) for p in meta["image_files"]]}
    dump = exports / "audit_dump.npz"
    if dump.exists():
        out["dump"] = dict(np.load(dump, allow_pickle=True))
    return out


def write_variant_exports(src_exports: str | Path, dst_work_dir: str | Path, data: str, scene: str,
                          extrinsics: np.ndarray, intrinsics: np.ndarray) -> Path:
    """Copy one scene's baseline exports into another evaluation folder with new cameras.

    Depth and confidence are copied unchanged (bit-identical to the baseline run); only the
    extrinsics / intrinsics change, so a harness `--eval-only` run on `dst_work_dir` measures the
    effect of the pose source alone."""
    src_exports = Path(src_exports)
    dst = Path(dst_work_dir) / "model_results" / data / scene / "unposed" / "exports"
    (dst / "mini_npz").mkdir(parents=True, exist_ok=True)
    res = dict(np.load(src_exports / "mini_npz" / "results.npz"))
    res["extrinsics"] = np.asarray(extrinsics, np.float32)[:, :3, :4]
    res["intrinsics"] = np.asarray(intrinsics, np.float32)
    np.savez_compressed(dst / "mini_npz" / "results.npz", **res)
    shutil.copy2(src_exports / "gt_meta.npz", dst / "gt_meta.npz")
    return dst
