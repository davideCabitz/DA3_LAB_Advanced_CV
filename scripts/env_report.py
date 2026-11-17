"""Report the compute environment (Phase 0 check on Kaggle): GPUs, RAM, disk, versions.

  python scripts/env_report.py --out /kaggle/working/env_report.json
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<failed: {e}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import torch

    rep = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpus": [{"name": torch.cuda.get_device_name(i),
                  "mem_GiB": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2),
                  "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
                  "bf16_supported": torch.cuda.is_bf16_supported() if i == 0 else None}
                 for i in range(torch.cuda.device_count())],
        "nvidia_smi": run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv"),
        "free_h": run("free -h"),
        "disk": {p: {k: round(v / 2**30, 1) for k, v in shutil.disk_usage(p)._asdict().items()}
                 for p in ("/kaggle/working", "/kaggle/tmp", "/tmp", ".") if Path(p).exists()},
        "kaggle_inputs": sorted(os.listdir("/kaggle/input")) if Path("/kaggle/input").exists() else None,
        "internet": run("curl -sI --max-time 10 https://huggingface.co | head -1"),
    }
    try:
        import torch.nn.functional as F

        q = torch.randn(1, 6, 4096, 64, device="cuda", dtype=torch.float16)
        F.scaled_dot_product_attention(q, q, q)
        rep["sdpa_fp16_ok"] = True
    except Exception as e:  # noqa: BLE001
        rep["sdpa_fp16_ok"] = f"{type(e).__name__}: {e}"
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
