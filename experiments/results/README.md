# Standard result records

One JSON file per run, original DA3 or enhancement:

```
experiments/results/<enhancement>/<tag>.json      (schema "da3-result/1", see src/da3/compare/schema.py)
```

- **Where they come from.** The Kaggle notebooks write them to `/kaggle/working/<…>/results/`.
  Copy those folders here, keeping the sub-folders (`baseline/`, `self_audit/`, `foveated/`, …), and commit
  them.
- **Who reads them.** `notebooks/09_compare_all` and `scripts/compare_results.py` read every
  record in this folder and compare the original DA3 with all enhancements. Runs are only compared
  when they share weights, datasets, scenes, frame sampling and input (`protocol.input`, e.g.
  "lens:zoom=2,fraction=0.3" for the mixed-lens protocol of enhancement #4, `lens/`).
- How to run and what not to do: [docs/ENHANCEMENTS_RUNBOOK.md](../../docs/ENHANCEMENTS_RUNBOOK.md).
