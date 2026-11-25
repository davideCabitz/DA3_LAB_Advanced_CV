"""LoRA platform check on the GPU with the RELEASED weights (notebook 07, part 0).

1. Determinism floor: the released model run twice on the same input (fp16 autocast, as in the
   benchmark adapter) must give identical outputs; otherwise bit-identity cannot be tested here.
2. LoRA off = released: same input through released DA3 and through DA3 + injected LoRA switched off.
3. B = 0 at init = released: the LoRA switched on right after injection.
4. Memory / time probe: one training step (forward + backward into A/B only, backbone gradient
   checkpointing, fp16 autocast) per configuration; replaces the estimates in docs/LORA.md.

  python scripts/lora_check.py --presets da3-small da3-base --out /kaggle/working/lora/lora_check.json
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from da3.lora import LoRAConfig, count_lora_parameters, inject_lora, lora_parameters, set_lora_enabled

KEYS = ("depth", "depth_conf", "ray", "ray_conf", "pose_enc")


def outputs(model, x, amp=True):
    with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.float16, enabled=amp and x.is_cuda):
        out = model(x, ref_strategy="first")
    return {k: out[k].float().cpu() for k in KEYS if k in out}


def compare(a, b) -> dict:
    return {k: {"equal": bool(torch.equal(a[k], b[k])), "max_abs": float((a[k] - b[k]).abs().max())} for k in a}


def bit_checks(preset, device, views, hw, cfg) -> dict:
    from da3.checkpoint_io import official_model

    ref, _ = official_model(preset)
    ref = ref.to(device).eval()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, views, 3, *hw, generator=g).to(device)
    a1, a2 = outputs(ref, x), outputs(ref, x)
    lora = copy.deepcopy(ref)
    inject_lora(lora, cfg, seed=0)
    b0 = outputs(lora, x)                              # enabled, B = 0
    set_lora_enabled(lora, False)
    off = outputs(lora, x)
    with torch.no_grad():                              # a non-trivial adapter must change the outputs
        set_lora_enabled(lora, True)
        for n, p in lora.named_parameters():
            if n.endswith("lora_B"):
                p.normal_(0, 0.01, generator=torch.Generator(device=p.device).manual_seed(1))
    on = outputs(lora, x)
    rep = {"repeat_released": compare(a1, a2), "lora_off_vs_released": compare(a1, off),
           "B0_vs_released": compare(a1, b0), "lora_on_changes_depth": not torch.equal(a1["depth"], on["depth"]),
           "lora_params": count_lora_parameters(lora)}
    det = all(v["equal"] for v in rep["repeat_released"].values())
    rep["deterministic"] = det
    rep["pass"] = (all(v["equal"] for v in rep["lora_off_vs_released"].values())
                   and all(v["equal"] for v in rep["B0_vs_released"].values()) and rep["lora_on_changes_depth"]) if det else None
    del ref, lora
    torch.cuda.empty_cache()
    return rep


def probe(preset, device, views, hw, cfg, steps=3) -> dict:
    from da3.checkpoint_io import official_model

    model, _ = official_model(preset, grad_checkpointing=True)
    model = model.to(device)
    inject_lora(model, cfg, seed=0)
    model.train()
    opt = torch.optim.AdamW(lora_parameters(model), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(1, views, 3, *hw, device=device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    times = []
    try:
        for _ in range(steps):
            t0 = time.time()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x, ref_strategy="first")
            loss = out["depth"].float().mean() + out["ray"].float().abs().mean() + out["pose_enc"].float().abs().mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        res = {"peak_gb": torch.cuda.max_memory_allocated() / 2**30, "s_per_step": min(times[1:] or times), "ok": True}
    except torch.cuda.OutOfMemoryError:
        res = {"peak_gb": None, "s_per_step": None, "ok": False, "error": "OOM"}
    del model, opt
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", nargs="+", default=["da3-small", "da3-base"])
    ap.add_argument("--probe", nargs="+", default=["8x280x504", "16x280x504", "8x504x504"],
                    help="views x H x W configurations for the memory probe")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "this check needs a GPU"
    device = torch.device("cuda")
    cfg = LoRAConfig()
    from da3.train_utils import environment_info, git_commit

    report = {"lora_config": cfg.to_dict(), "env": environment_info(), "commit": git_commit(Path(__file__).parents[1]),
              "bit": {}, "probe": {}}
    for preset in args.presets:
        report["bit"][preset] = bit_checks(preset, device, 4, (280, 504), cfg)
        print(preset, "bit checks:", json.dumps({k: report["bit"][preset][k] for k in ("deterministic", "pass", "lora_params")}),
              flush=True)
        for spec in args.probe:
            v, h, w = (int(s) for s in spec.split("x"))
            report["probe"][f"{preset}/{spec}"] = r = probe(preset, device, v, (h, w), cfg)
            print(f"{preset} {spec}: " + (f"peak {r['peak_gb']:.2f} GB, {r['s_per_step']:.2f} s/step" if r["ok"] else "OUT OF MEMORY"),
                  flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    failed = [p for p, r in report["bit"].items() if r["pass"] is False]
    print("wrote", args.out)
    if failed:
        raise SystemExit(f"LoRA bit-identity FAILED for {failed}: do not train, send {args.out}")


if __name__ == "__main__":
    main()
