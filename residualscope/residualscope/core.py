"""
residualscope.core
==================

Model-agnostic diagnostic engine for transformer residual pathway analysis.

Design principle: this module knows NOTHING about your specific model
architecture. It works via forward/backward hooks registered on whatever
nn.Linear / nn.Module instances you point it at. This is what makes it
reusable beyond the ResidualGPT experiments it was extracted from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class LayerSnapshot:
    """Single-layer, single-step diagnostic readout."""
    layer_name: str
    step: int
    grad_norm: Optional[float] = None
    hidden_norm: Optional[float] = None
    sublayer_output_norm: Optional[float] = None
    dead_neuron_frac: Optional[float] = None
    activation_mean_abs: Optional[float] = None


@dataclass
class ScopeReport:
    """Full diagnostic history across all tracked layers and steps."""
    layer_names: List[str] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    snapshots: List[LayerSnapshot] = field(default_factory=list)

    def by_layer(self, layer_name: str) -> List[LayerSnapshot]:
        return [s for s in self.snapshots if s.layer_name == layer_name]

    def by_step(self, step: int) -> List[LayerSnapshot]:
        return [s for s in self.snapshots if s.step == step]

    def grad_norm_trajectory(self, layer_name: str) -> List[float]:
        """Gradient norm at this layer across all recorded steps."""
        return [s.grad_norm for s in self.by_layer(layer_name) if s.grad_norm is not None]

    def hidden_norm_growth_ratio(self, layer_name: str) -> Optional[float]:
        """final hidden norm / initial hidden norm for one layer. None if insufficient data."""
        vals = [s.hidden_norm for s in self.by_layer(layer_name) if s.hidden_norm is not None]
        if len(vals) < 2 or vals[0] == 0:
            return None
        return vals[-1] / vals[0]

    def dead_neuron_fraction_final(self, layer_name: str) -> Optional[float]:
        vals = [s.dead_neuron_frac for s in self.by_layer(layer_name) if s.dead_neuron_frac is not None]
        return vals[-1] if vals else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_names": self.layer_names,
            "steps": self.steps,
            "snapshots": [vars(s) for s in self.snapshots],
        }


# ──────────────────────────────────────────────────────────────────────────
# The tracker
# ──────────────────────────────────────────────────────────────────────────

class ResidualScope:
    """
    Attaches forward/backward hooks to named modules and records the four
    core diagnostics at every call to `.step()`.

    Usage
    -----
    >>> scope = ResidualScope(model, layer_filter=lambda name, mod: isinstance(mod, nn.Linear))
    >>> for step in range(n_steps):
    ...     loss = train_step(...)
    ...     scope.step(step)          # call AFTER loss.backward(), BEFORE optimizer.step()
    >>> report = scope.report()
    >>> report.hidden_norm_growth_ratio("blocks.0.mlp.down")

    Parameters
    ----------
    model : nn.Module
        Any PyTorch module. ResidualScope does not assume any particular
        architecture — it just walks `model.named_modules()`.
    layer_filter : Callable[[str, nn.Module], bool], optional
        Predicate deciding which named modules to track. Defaults to
        tracking every `nn.Linear`. Pass your own filter to track only
        specific sublayers (e.g. only FFN output projections).
    track_activations : bool
        If True, also tracks the input activations to each tracked module
        for dead-neuron detection. Requires forward hooks (default True).
    """

    def __init__(
        self,
        model: nn.Module,
        layer_filter: Optional[Callable[[str, nn.Module], bool]] = None,
        track_activations: bool = True,
    ):
        self.model = model
        self.track_activations = track_activations
        self._filter = layer_filter or (lambda name, mod: isinstance(mod, nn.Linear))

        self._tracked: Dict[str, nn.Module] = {}
        self._fwd_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._bwd_handles: List[torch.utils.hooks.RemovableHandle] = []

        # Per-step scratch state, cleared after each .step() call
        self._last_input: Dict[str, torch.Tensor] = {}
        self._last_output: Dict[str, torch.Tensor] = {}
        self._last_grad: Dict[str, torch.Tensor] = {}

        self._report = ScopeReport()

        self._register_hooks()

    # ── hook registration ──────────────────────────────────────────────

    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if name == "":
                continue
            if not self._filter(name, module):
                continue
            self._tracked[name] = module
            self._report.layer_names.append(name)

            fwd = module.register_forward_hook(self._make_fwd_hook(name))
            self._fwd_handles.append(fwd)

            # Gradient hook on the module's weight, not the activation —
            # this is what gives us the gradient-starvation signature
            # (e.g. Layer 0 fc1 gradient collapsing to 0.000).
            if hasattr(module, "weight") and module.weight is not None:
                bwd = module.weight.register_hook(self._make_weight_grad_hook(name))
                self._bwd_handles.append(bwd)

    def _make_fwd_hook(self, name: str):
        def hook(module, inputs, output):
            if self.track_activations and len(inputs) > 0 and torch.is_tensor(inputs[0]):
                self._last_input[name] = inputs[0].detach()
            if torch.is_tensor(output):
                self._last_output[name] = output.detach()
        return hook

    def _make_weight_grad_hook(self, name: str):
        def hook(grad: torch.Tensor):
            self._last_grad[name] = grad.detach().clone()
        return hook

    # ── per-step recording ──────────────────────────────────────────────

    def step(self, step: int) -> None:
        """
        Record one diagnostic snapshot per tracked layer at this step.

        Call this AFTER loss.backward() so weight gradients are populated,
        and BEFORE optimizer.step() so you're reading this step's gradient,
        not next step's already-updated weights.
        """
        self._report.steps.append(step)
        for name in self._tracked:
            snap = LayerSnapshot(layer_name=name, step=step)

            grad = self._last_grad.get(name)
            if grad is not None:
                snap.grad_norm = float(grad.norm(p=2).item())

            out = self._last_output.get(name)
            if out is not None:
                # mean L2 norm across the batch/sequence dims, scalar per layer
                snap.hidden_norm = float(out.detach().norm(dim=-1).mean().item())
                snap.sublayer_output_norm = snap.hidden_norm

            inp = self._last_input.get(name)
            if inp is not None:
                snap.activation_mean_abs = float(inp.abs().mean().item())
                # "Dead" = activation is exactly zero for this forward pass.
                # For ReLU/GELU-family activations measured pre-activation,
                # this approximates the standard dead-neuron definition.
                snap.dead_neuron_frac = float((inp <= 0).float().mean().item())

            self._report.snapshots.append(snap)

        # Clear scratch state so stale values can't leak into the next step
        self._last_input.clear()
        self._last_output.clear()
        self._last_grad.clear()

    def report(self) -> ScopeReport:
        return self._report

    def close(self) -> None:
        """Remove all hooks. Call when done to avoid memory leaks on long-lived models."""
        for h in self._fwd_handles:
            h.remove()
        for h in self._bwd_handles:
            h.remove()
        self._fwd_handles.clear()
        self._bwd_handles.clear()

    def __enter__(self) -> "ResidualScope":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ──────────────────────────────────────────────────────────────────────────
# Convenience: load a checkpoint and run N forward+backward passes for a
# quick one-shot diagnostic without a full training loop.
# ──────────────────────────────────────────────────────────────────────────

def quick_scan(
    model: nn.Module,
    loss_fn: Callable[[nn.Module], torch.Tensor],
    n_steps: int = 5,
    layer_filter: Optional[Callable[[str, nn.Module], bool]] = None,
) -> ScopeReport:
    """
    Run `n_steps` forward+backward passes through `loss_fn(model)` and
    return a ScopeReport. Does not call optimizer.step() — this is a
    read-only diagnostic, not a training loop. Useful for a quick health
    check on a freshly loaded checkpoint without wiring up a full trainer.

    Parameters
    ----------
    model : nn.Module
    loss_fn : Callable[[nn.Module], torch.Tensor]
        A zero-argument-aside-from-model closure that runs one forward
        pass and returns a scalar loss, e.g.:
            lambda m: m(batch_x, batch_y)[1]
    n_steps : int
        Number of forward/backward passes to run.
    layer_filter : optional filter, see ResidualScope.

    Returns
    -------
    ScopeReport
    """
    with ResidualScope(model, layer_filter=layer_filter) as scope:
        for step in range(n_steps):
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model)
            loss.backward()
            scope.step(step)
        return scope.report()
