from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

from .core import ScopeReport


@dataclass
class ComparisonResult:
    reports: Dict[str, ScopeReport]

    def hidden_norm_growth_table(self, layer_name: str) -> Dict[str, Optional[float]]:
        return {
            name: report.hidden_norm_growth_ratio(layer_name)
            for name, report in self.reports.items()
        }

    def final_grad_norm_table(self, layer_name: str) -> Dict[str, Optional[float]]:
        out = {}
        for name, report in self.reports.items():
            traj = report.grad_norm_trajectory(layer_name)
            out[name] = traj[-1] if traj else None
        return out

    def summary_table(self, layer_name: str) -> List[Dict[str, object]]:
        """One row per config — same structure as Tables 3/5 in the paper."""
        rows = []
        for name, report in self.reports.items():
            traj = report.grad_norm_trajectory(layer_name)
            rows.append({
                "config": name,
                "final_grad_norm": traj[-1] if traj else None,
                "hidden_norm_growth_ratio": report.hidden_norm_growth_ratio(layer_name),
                "dead_neuron_frac_final": report.dead_neuron_fraction_final(layer_name),
            })
        return rows


def compare(named_reports: Dict[str, ScopeReport]) -> ComparisonResult:
    return ComparisonResult(reports=named_reports)


def plot_comparison_bar(
    comparison: ComparisonResult,
    layer_name: str,
    metric: str = "hidden_norm_growth_ratio",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 5))

    table = comparison.summary_table(layer_name)
    names = [row["config"] for row in table]
    values = [row.get(metric) for row in table]

    default_colors = ["#1565C0", "#2E7D32", "#C62828", "#757575", "#6A1B9A", "#EF6C00"]
    bar_colors = [
        (colors or {}).get(n, default_colors[i % len(default_colors)])
        for i, n in enumerate(names)
    ]

    plot_values = [v if v is not None else 0 for v in values]
    bars = ax.bar(range(len(names)), plot_values, color=bar_colors, width=0.6,
                  edgecolor="white", zorder=3)
    for bar, v in zip(bars, values):
        if v is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(plot_values) * 0.02,
                f"{v:.3f}", ha="center", fontsize=9, fontweight="bold"
            )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title or f"{metric.replace('_', ' ')} — layer: {layer_name}", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if own_fig:
        return fig, ax
    return ax
