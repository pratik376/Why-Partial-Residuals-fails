"""
ResidualScope — diagnostic hooks for transformer residual pathway analysis.

Extracted from the experimental code for "A Reproducibility Study of Partial
Residual Ablations" (Patel, 2026). Tracks gradient norms, hidden-state drift,
and dead-neuron fractions via forward/backward hooks on any nn.Module.

    from residualscope import ResidualScope

    scope = ResidualScope(model)
    for step in range(n_steps):
        loss = model(x, y)[1]
        loss.backward()
        scope.step(step)
        optimizer.step()
        optimizer.zero_grad()

    report = scope.report()
    report.hidden_norm_growth_ratio("blocks.3.mlp.fc1")  # -> 14.03 for AttnOnly
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
