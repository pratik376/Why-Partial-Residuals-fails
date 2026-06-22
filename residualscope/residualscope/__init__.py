"""
ResidualScope
=============

A model-agnostic diagnostic tool for transformer internals — gradient
norm tracking, hidden-state representation drift, dead-neuron detection,
and cross-configuration comparison.

Extracted and generalized from the diagnostic code used in the paper
"Why Partial Residuals Fail: Asymmetric Pathway Necessity in Transformer
Language Models," where these exact diagnostics were used to discover
and explain a 2.13x-to-2.36x asymmetry between attention-only and
FFN-only residual configurations.

Quick start
-----------
>>> from residualscope import ResidualScope
>>> scope = ResidualScope(model)
>>> for step in range(n_steps):
...     loss = model(x, y)[1]
...     loss.backward()
...     scope.step(step)
...     optimizer.step()
...     optimizer.zero_grad()
>>> report = scope.report()
>>> report.hidden_norm_growth_ratio("blocks.3.mlp.down")
14.03
"""

from .core import LayerSnapshot, ResidualScope, ScopeReport, quick_scan
from .compare import ComparisonResult, compare, plot_comparison_bar
from .plots import (
    plot_dead_neurons,
    plot_gradient_norms,
    plot_hidden_norm_growth,
    plot_summary_dashboard,
)

__version__ = "0.1.0"

__all__ = [
    "ResidualScope",
    "ScopeReport",
    "LayerSnapshot",
    "quick_scan",
    "compare",
    "ComparisonResult",
    "plot_comparison_bar",
    "plot_gradient_norms",
    "plot_hidden_norm_growth",
    "plot_dead_neurons",
    "plot_summary_dashboard",
]
