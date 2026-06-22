"""
Tests for residualscope.core.

These tests use small synthetic models with KNOWN, hand-computable
diagnostic signatures, so that a passing test suite is real evidence
the diagnostics are computing what they claim to compute -- not just
evidence the code runs without crashing.
"""

import math

import pytest
import torch
import torch.nn as nn

from residualscope import ResidualScope, quick_scan
from residualscope.compare import compare, plot_comparison_bar


# ──────────────────────────────────────────────────────────────────────────
# Fixtures: tiny synthetic models with known behavior
# ──────────────────────────────────────────────────────────────────────────

class TinyResidualBlock(nn.Module):
    """
    A minimal block with a togglable residual connection, mirroring the
    AttnOnly/FFNOnly switch structure from the source research at a
    scale small enough to reason about by hand.
    """

    def __init__(self, dim: int = 8, use_residual: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)
        self.use_residual = use_residual

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        out = self.fc2(h)
        return x + out if self.use_residual else out


class TinyStack(nn.Module):
    def __init__(self, dim: int = 8, n_layers: int = 2, use_residual: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TinyResidualBlock(dim, use_residual) for _ in range(n_layers)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def _loss_fn_factory(model, batch_size=4, dim=8):
    x = torch.randn(batch_size, dim)
    target = torch.randn(batch_size, dim)

    def loss_fn(m):
        out = m(x)
        return ((out - target) ** 2).mean()

    return loss_fn


# ──────────────────────────────────────────────────────────────────────────
# Core hook mechanics
# ──────────────────────────────────────────────────────────────────────────

def test_tracks_only_filtered_layers():
    model = TinyStack(n_layers=3)
    scope = ResidualScope(model, layer_filter=lambda name, mod: "fc1" in name)
    tracked = scope.report().layer_names
    assert len(tracked) == 3
    assert all("fc1" in name for name in tracked)
    scope.close()


def test_default_filter_tracks_all_linear_layers():
    model = TinyStack(n_layers=2)  # 2 blocks * 2 Linear each = 4
    scope = ResidualScope(model)
    assert len(scope.report().layer_names) == 4
    scope.close()


def test_step_records_one_snapshot_per_layer():
    model = TinyStack(n_layers=2)
    loss_fn = _loss_fn_factory(model)
    report = quick_scan(model, loss_fn, n_steps=3)

    n_layers = len(report.layer_names)
    assert len(report.snapshots) == 3 * n_layers
    assert report.steps == [0, 1, 2]


def test_close_removes_hooks():
    model = TinyStack(n_layers=2)
    scope = ResidualScope(model)
    n_fwd_before = len(scope._fwd_handles)
    assert n_fwd_before > 0
    scope.close()
    assert len(scope._fwd_handles) == 0
    assert len(scope._bwd_handles) == 0


def test_context_manager_closes_on_exit():
    model = TinyStack(n_layers=1)
    with ResidualScope(model) as scope:
        assert len(scope._fwd_handles) > 0
    assert len(scope._fwd_handles) == 0


# ──────────────────────────────────────────────────────────────────────────
# Diagnostic correctness against hand-computable cases
# ──────────────────────────────────────────────────────────────────────────

def test_gradient_norm_is_nonzero_with_residual():
    """
    With a residual connection intact, gradients should reach early
    layers in a multi-block stack -- the gradient-highway behavior
    this whole package exists to detect the ABSENCE of.
    """
    torch.manual_seed(0)
    model = TinyStack(dim=8, n_layers=4, use_residual=True)
    loss_fn = _loss_fn_factory(model, dim=8)
    report = quick_scan(model, loss_fn, n_steps=1)

    first_layer = report.layer_names[0]
    traj = report.grad_norm_trajectory(first_layer)
    assert len(traj) == 1
    assert traj[0] > 0.0, "Earliest layer should receive nonzero gradient when residual is present"


def test_hidden_norm_growth_ratio_near_one_for_stable_input():
    """
    Hand-computable case: a layer whose output norm doesn't change
    across steps should report a growth ratio of ~1.0.
    """
    torch.manual_seed(0)
    model = TinyStack(dim=8, n_layers=1, use_residual=True)
    loss_fn = _loss_fn_factory(model, dim=8)

    # Use a very small number of steps so weights barely move,
    # keeping the output norm approximately stable.
    report = quick_scan(model, loss_fn, n_steps=2)
    layer = report.layer_names[-1]  # fc2 of the single block
    ratio = report.hidden_norm_growth_ratio(layer)
    assert ratio is not None
    assert 0.5 < ratio < 2.0, f"Expected near-stable ratio, got {ratio}"


def test_dead_neuron_fraction_is_bounded():
    torch.manual_seed(0)
    model = TinyStack(dim=8, n_layers=2)
    loss_fn = _loss_fn_factory(model, dim=8)
    report = quick_scan(model, loss_fn, n_steps=2)

    for layer in report.layer_names:
        frac = report.dead_neuron_fraction_final(layer)
        if frac is not None:
            assert 0.0 <= frac <= 1.0


def test_known_dead_neuron_case():
    """
    Construct a Linear layer fed an all-negative input (so a ReLU
    immediately downstream would zero every unit). The tracked
    PRE-activation tensor here is the all-negative input itself, so
    dead_neuron_frac should be exactly 1.0.
    """
    layer = nn.Linear(4, 4, bias=False)
    model = nn.Sequential(layer)

    x = -torch.ones(2, 4)  # strictly negative input to the tracked layer
    target = torch.zeros(2, 4)

    def loss_fn(m):
        out = m(x)
        return ((out - target) ** 2).mean()

    report = quick_scan(model, loss_fn, n_steps=1, layer_filter=lambda n, m: isinstance(m, nn.Linear))
    layer_name = report.layer_names[0]
    frac = report.dead_neuron_fraction_final(layer_name)
    assert frac == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────────
# The asymmetry use case: comparing configurations
# ──────────────────────────────────────────────────────────────────────────

def test_compare_surfaces_residual_vs_no_residual_difference():
    """
    Reproduces the core comparison use case at toy scale: a model WITH
    a residual connection should show a different hidden-norm growth
    ratio than a model WITHOUT one. This requires actually updating
    weights between steps (quick_scan alone does not call optimizer.step,
    by design -- it's a read-only diagnostic), so this test runs a real
    minimal training loop rather than using the quick_scan shortcut.
    """
    torch.manual_seed(0)
    dim, n_layers, n_steps = 16, 4, 60

    def train_and_scan(use_residual: bool):
        model = TinyStack(dim=dim, n_layers=n_layers, use_residual=use_residual)
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        loss_fn = _loss_fn_factory(model, dim=dim)
        with ResidualScope(model) as scope:
            for step in range(n_steps):
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model)
                loss.backward()
                scope.step(step)
                opt.step()
            return scope.report()

    report_with = train_and_scan(True)
    report_without = train_and_scan(False)

    comparison = compare({"with_residual": report_with, "without_residual": report_without})

    last_layer = report_with.layer_names[-1]
    table = comparison.hidden_norm_growth_table(last_layer)

    assert "with_residual" in table
    assert "without_residual" in table
    assert table["with_residual"] is not None
    assert table["without_residual"] is not None
    # With real weight updates across 60 steps, the two configurations
    # must diverge -- this is the differentiated reading the comparison
    # API exists to surface.
    assert table["with_residual"] != table["without_residual"]


def test_comparison_summary_table_has_expected_keys():
    torch.manual_seed(0)
    model_a = TinyStack(n_layers=2, use_residual=True)
    model_b = TinyStack(n_layers=2, use_residual=False)

    report_a = quick_scan(model_a, _loss_fn_factory(model_a), n_steps=3)
    report_b = quick_scan(model_b, _loss_fn_factory(model_b), n_steps=3)

    comparison = compare({"A": report_a, "B": report_b})
    rows = comparison.summary_table(report_a.layer_names[0])

    assert len(rows) == 2
    for row in rows:
        assert set(row.keys()) == {
            "config", "final_grad_norm",
            "hidden_norm_growth_ratio", "dead_neuron_frac_final",
        }


def test_plot_comparison_bar_runs_without_error():
    """Smoke test: plotting code should not crash on a real comparison."""
    import matplotlib
    matplotlib.use("Agg")

    torch.manual_seed(0)
    model_a = TinyStack(n_layers=2, use_residual=True)
    model_b = TinyStack(n_layers=2, use_residual=False)
    report_a = quick_scan(model_a, _loss_fn_factory(model_a), n_steps=3)
    report_b = quick_scan(model_b, _loss_fn_factory(model_b), n_steps=3)
    comparison = compare({"A": report_a, "B": report_b})

    fig, ax = plot_comparison_bar(comparison, report_a.layer_names[0])
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)
