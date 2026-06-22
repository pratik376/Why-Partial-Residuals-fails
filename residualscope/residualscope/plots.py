"""
residualscope.plots
====================

Plotting helpers that reproduce the diagnostic figures used throughout
the residual-pathway research (gradient starvation, hidden-norm drift,
dead-neuron collapse) for an arbitrary ScopeReport.

These are intentionally close to the exact panels used in the paper
"Why Partial Residuals Fail" — this module is the generalized,
re-usable version of the one-off matplotlib code from that project's
notebooks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .core import ScopeReport


def _default_style():
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
    Plot gradient L2 norm trajectories across training steps for the
    given layers (default: all tracked layers).

    This is the panel that reveals gradient starvation: a layer whose
    residual connection has been removed will typically show its
    gradient norm collapse to ~0 within the first few hundred steps,
    while a layer with an intact residual connection sustains a
    non-trivial gradient norm throughout training.
    """
    _default_style()
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 5))

    names = layer_names or report.layer_names
    for name in names:
        traj = report.grad_norm_trajectory(name)
        steps = report.steps[: len(traj)]
        if traj:
            ax.plot(steps, traj, marker="o", ms=3, lw=1.6, label=name)

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
    title: str = "Hidden-state norm growth ratio (final / init)",
):
    """
    Bar chart of the final/initial hidden-state norm ratio per layer.

    A ratio near 1.0 indicates a controlled, stable representation.
    Large ratios (5-20x or more, as observed for partial-residual
    configurations in the source research) indicate uncontrolled
    representation drift -- the hallmark of a sublayer that lost its
    identity path and is no longer anchored to its initial scale.
    """
    _default_style()
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 5))

    names = layer_names or report.layer_names
    ratios = []
    labels = []
    for name in names:
        r = report.hidden_norm_growth_ratio(name)
        if r is not None:
            ratios.append(r)
            labels.append(name)

    if ratios:
        bars = ax.bar(range(len(ratios)), ratios, color="#3F51B5", alpha=0.85, zorder=3)
        for bar, v in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, v + max(ratios) * 0.02,
                    f"{v:.1f}\u00d7", ha="center", fontsize=8, fontweight="bold")
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
    title: str = "Dead neuron fraction by layer (final step)",
):
    """
    Bar chart of the final dead-neuron fraction per layer.

    Caution interpreting this in isolation: a healthy, well-trained
    model can have a HIGH dead-neuron fraction because it has learned
    sparse, selective features. A collapsed model can have a LOW
    dead-neuron fraction because it never learned any structure at all
    (activations stay close to their random-initialization statistics).
    Always read this panel together with the loss curve and the hidden
    norm growth panel, not as a standalone health signal.
    """
    _default_style()
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 5))

    names = layer_names or report.layer_names
    fracs = []
    labels = []
    for name in names:
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


def plot_summary_dashboard(report: ScopeReport, suptitle: str = "ResidualScope diagnostic summary"):
    """
    Three-panel dashboard combining gradient norms, hidden-norm growth,
    and dead-neuron fractions -- the same three-panel layout used to
    diagnose the asymmetric residual-pathway failure in the source
    research (gradient starvation + representation drift + outcome).

    Returns the matplotlib Figure; caller is responsible for saving or
    displaying it.
    """
    _default_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plot_gradient_norms(report, ax=axes[0])
    plot_hidden_norm_growth(report, ax=axes[1])
    plot_dead_neurons(report, ax=axes[2])
    fig.suptitle(suptitle, fontweight="bold", fontsize=13)
    fig.tight_layout()
    return fig
