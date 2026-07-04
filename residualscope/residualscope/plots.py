from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .core import ScopeReport


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def plot_gradient_norms(
    report: ScopeReport,
    layer_names: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Gradient norm by layer",
):
    """
    Gradient L2 norm across training steps per layer.

    In the original experiments, all partial-residual configurations
    (AttnOnly, FFNOnly, NoResidual) showed Layer 0 grad norm collapsing
    to 0.000 from step 300 onward. FullResidual sustained ~0.114.
    """
    _style()
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 5))

    for name in (layer_names or report.layer_names):
        traj = report.grad_norm_trajectory(name)
        if traj:
            ax.plot(report.steps[:len(traj)], traj, marker="o", ms=3, lw=1.6, label=name)

    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient L2 norm")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)

    if own_fig:
        return fig, ax
    return ax


def plot_hidden_norm_growth(
    report: ScopeReport,
    layer_names: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Hidden-state norm growth (final / init)",
):
    """
    Final/initial hidden-state norm ratio per layer.

    From the 10M experiments: FullResidual 1.38×, AttnOnly 14.03×,
    FFNOnly 5.37×, NoResidual 8.4×. The divergence between AttnOnly
    and FullResidual — despite similar losses at that scale — was the
    first mechanistic signal of structural difference between configs.
    """
    _style()
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 5))

    ratios, labels = [], []
    for name in (layer_names or report.layer_names):
        r = report.hidden_norm_growth_ratio(name)
        if r is not None:
            ratios.append(r)
            labels.append(name)

    if ratios:
        bars = ax.bar(range(len(ratios)), ratios, color="#3F51B5", alpha=0.85, zorder=3)
        for bar, v in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, v + max(ratios) * 0.02,
                    f"{v:.1f}×", ha="center", fontsize=8, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.axhline(1.0, color="#333", lw=1, ls="--", alpha=0.4)

    ax.set_ylabel("Growth ratio")
    ax.set_title(title, fontweight="bold")

    if own_fig:
        return fig, ax
    return ax


def plot_dead_neurons(
    report: ScopeReport,
    layer_names: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Dead neuron fraction (final step)",
):
    _style()
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 5))

    fracs, labels = [], []
    for name in (layer_names or report.layer_names):
        f = report.dead_neuron_fraction_final(name)
        if f is not None:
            fracs.append(f * 100)
            labels.append(name)

    if fracs:
        ax.bar(range(len(fracs)), fracs, color="#E53935", alpha=0.85, zorder=3)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_ylim(0, 100)

    ax.set_ylabel("Dead neuron %")
    ax.set_title(title, fontweight="bold")

    if own_fig:
        return fig, ax
    return ax


def plot_summary_dashboard(report: ScopeReport, suptitle: str = "ResidualScope diagnostics"):
    """Three-panel dashboard: gradient norms, hidden-norm growth, dead neurons."""
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plot_gradient_norms(report, ax=axes[0])
    plot_hidden_norm_growth(report, ax=axes[1])
    plot_dead_neurons(report, ax=axes[2])
    fig.suptitle(suptitle, fontweight="bold", fontsize=13)
    fig.tight_layout()
    return fig
