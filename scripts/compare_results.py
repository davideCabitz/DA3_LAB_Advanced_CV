"""Original DA3 vs every enhancement, from the standard result records (da3.compare).

Reads every record under the given folders (default: experiments/results), groups runs that share
the same protocol (weights, datasets, scenes, frame sampling) and writes one Markdown report:
Sec. 6 tables with differences to the original DA3, then each enhancement's own metrics.

  python scripts/compare_results.py --dirs experiments/results /kaggle/input/<old-run>/results \
      --out reports/comparison.md
"""

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=[str(REPO / "experiments" / "results")])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from da3.compare import load_records, summary_markdown

    records = load_records(*args.dirs)
    head = [f"# Original DA3 vs enhancements\n",
            f"{len(records)} result records from: " + ", ".join(f"`{d}`" for d in args.dirs) + "\n",
            "| Enhancement | Variant | Model | Tag | Setting |", "|---|---|---|---|---|"]
    head += [f"| {r['enhancement']} | {r['variant']} | {r['weights']} | {r['tag']} | {r['setting']} |"
             for r in sorted(records, key=lambda r: (r["enhancement"], r["weights"], r["variant"]))]
    md = "\n".join(head) + "\n\n" + summary_markdown(records)
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md, encoding="utf-8")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
