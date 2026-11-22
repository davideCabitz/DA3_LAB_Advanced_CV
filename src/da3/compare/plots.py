"""Plots for the comparison notebooks (matplotlib; imported lazily so the package works without it)."""

from __future__ import annotations

from .tables import DATA_NAMES, model_name, sec6_table


def sec6_delta_bars(records: list[dict], label: str = "AUC@3", ax=None):
    """Bar chart: difference to the original DA3 (per dataset) for every non-reference variant."""
    import matplotlib.pyplot as plt
    import numpy as np

    tables = sec6_table(records)
    series, datasets = {}, []
    for t in tables:
        for row in t["rows"]:
            for (d, lab), delta in row["deltas"].items():
                if lab != label or delta is None:
                    continue
                if d not in datasets:
                    datasets.append(d)
                series.setdefault(f"{model_name(t['weights'])} · {row['enhancement']}:{row['variant']}", {})[d] = delta
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 1.6 * len(datasets) + 2), 3.6))
    if not series:
        ax.text(0.5, 0.5, f"no {label} differences to plot", ha="center", va="center")
        ax.set_axis_off()
        return ax
    x = np.arange(len(datasets))
    w = 0.8 / len(series)
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + i * w - 0.4 + w / 2, [vals.get(d, np.nan) for d in datasets], w, label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [DATA_NAMES.get(d, d) for d in datasets])
    ax.set_ylabel(f"Δ {label} vs original DA3")
    ax.legend(fontsize=7, loc="best")
    ax.grid(axis="y", alpha=0.3)
    return ax


def curves(named: dict, xlabel: str, ylabel: str, title: str = "", ax=None, diagonal: bool = False):
    """Line plot of {name: (x, y)} (risk–coverage curves, reliability diagrams, ...)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 3.6))
    for name, (x, y) in named.items():
        ax.plot(x, y, label=name, lw=1.6)
    if diagonal:
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="ideal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    return ax
