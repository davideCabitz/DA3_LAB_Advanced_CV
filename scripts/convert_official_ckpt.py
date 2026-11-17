"""Convert a released DA3 checkpoint (HF safetensors) to our module names and verify a strict load.

  python scripts/convert_official_ckpt.py --preset da3-small --out weights/da3-small_ours.safetensors

The mapping rules are in src/da3/checkpoint_io.py (_RULES). Dropped keys are the ray-branch output
layers of pyramid levels 0-2, which the released forward pass never uses (only level 3 is used;
its LayerNorm is shared with level 0 and stored under level 0's key).
"""

import argparse
import json

from safetensors.torch import save_file

from da3.checkpoint_io import convert_official_state_dict, download_official, load_safetensors, official_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="da3-small", choices=["da3-small", "da3-base", "da3-large"])
    ap.add_argument("--weights", default=None, help="local model.safetensors (default: download from HF)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    path = args.weights or download_official(args.preset)
    sd, dropped = convert_official_state_dict(load_safetensors(path))
    _, info = official_model(args.preset, path)  # strict load into our DA3 with the released preset
    save_file({k: v.contiguous() for k, v in sd.items()}, args.out)
    print(json.dumps({"source": str(path), "out": args.out, "tensors": len(sd), "dropped_unused": dropped,
                      "strict_load_missing": info["missing"], "strict_load_unexpected": info["unexpected"]}, indent=2))


if __name__ == "__main__":
    main()
