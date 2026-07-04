from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class LayerSnapshot:
    layer_name: str
    step: int
    grad_norm: Optional[float] = None
    hidden_norm: Optional[float] = None
    sublayer_output_norm: Optional[float] = None
    dead_neuron_frac: Optional[float] = None
    activation_mean_abs: Optional[float] = None


@dataclass
class ScopeReport:
    layer_names: List[str] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    snapshots: List[LayerSnapshot] = field(default_factory=list)

    def by_layer(self, layer_name: str) -> List[LayerSnapshot]:
        return [s for s in self.snapshots if s.layer_name == layer_name]

    def by_step(self, step: int) -> List[LayerSnapshot]:
        return [s for s in self.snapshots if s.step == step]

    def grad_norm_trajectory(self, layer_name: str) -> List[float]:
        return [s.grad_norm for s in self.by_layer(layer_name) if s.grad_norm is not None]

    def hidden_norm_growth_ratio(self, layer_name: str) -> Optional[float]:
        """Final hidden norm divided by initial — >1 means representation drift."""
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


class ResidualScope:
    """
    Attaches hooks to named modules and records gradient norms, hidden-state
    norms, and dead-neuron fractions at each training step.

    Extracted from the diagnostic code used in "A Reproducibility Study of
    Partial Residual Ablations" (Patel, 2026). The gradient-starvation
    signature in that paper — Layer 0 grad norm collapsing to 0.000 from
    step 300 onward in all partial-residual configs — was surfaced using
    exactly this hook pattern.

    Usage:
        scope = ResidualScope(model)
        for step in range(n_steps):
            loss = model(x, y)[1]
            loss.backward()
            scope.step(step)   # after backward, before optimizer.step()
            optimizer.step()
            optimizer.zero_grad()
        report = scope.report()
        report.hidden_norm_growth_ratio("blocks.0.mlp.fc1")
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
        self._fwd_handles = []
        self._bwd_handles = []

        self._last_input: Dict[str, torch.Tensor] = {}
        self._last_output: Dict[str, torch.Tensor] = {}
        self._last_grad: Dict[str, torch.Tensor] = {}

        self._report = ScopeReport()
        self._register_hooks()

    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if name == "" or not self._filter(name, module):
                continue
            self._tracked[name] = module
            self._report.layer_names.append(name)

            self._fwd_handles.append(
                module.register_forward_hook(self._make_fwd_hook(name))
            )

            # Hook on weight.grad, not activation grad — this gives the
            # gradient-starvation signature (weight grad → 0 when identity
            # path is removed and gradients can't flow back to early layers).
            if hasattr(module, "weight") and module.weight is not None:
                self._bwd_handles.append(
                    module.weight.register_hook(self._make_weight_grad_hook(name))
                )

    def _make_fwd_hook(self, name: str):
        def hook(module, inputs, output):
            if self.track_activations and inputs and torch.is_tensor(inputs[0]):
                self._last_input[name] = inputs[0].detach()
            if torch.is_tensor(output):
                self._last_output[name] = output.detach()
        return hook

    def _make_weight_grad_hook(self, name: str):
        def hook(grad: torch.Tensor):
            self._last_grad[name] = grad.detach().clone()
        return hook

    def step(self, step: int) -> None:
        """
        Record one snapshot per tracked layer.
        Call after loss.backward(), before optimizer.step().
        """
        self._report.steps.append(step)
        for name in self._tracked:
            snap = LayerSnapshot(layer_name=name, step=step)

            grad = self._last_grad.get(name)
            if grad is not None:
                snap.grad_norm = float(grad.norm(p=2).item())

            out = self._last_output.get(name)
            if out is not None:
                snap.hidden_norm = float(out.norm(dim=-1).mean().item())
                snap.sublayer_output_norm = snap.hidden_norm

            inp = self._last_input.get(name)
            if inp is not None:
                snap.activation_mean_abs = float(inp.abs().mean().item())
                snap.dead_neuron_frac = float((inp <= 0).float().mean().item())

            self._report.snapshots.append(snap)

        self._last_input.clear()
        self._last_output.clear()
        self._last_grad.clear()

    def report(self) -> ScopeReport:
        return self._report

    def close(self) -> None:
        for h in self._fwd_handles + self._bwd_handles:
            h.remove()
        self._fwd_handles.clear()
        self._bwd_handles.clear()

    def __enter__(self) -> "ResidualScope":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def quick_scan(
    model: nn.Module,
    loss_fn: Callable[[nn.Module], torch.Tensor],
    n_steps: int = 5,
    layer_filter: Optional[Callable[[str, nn.Module], bool]] = None,
) -> ScopeReport:
    """
    Run n_steps forward+backward passes and return a ScopeReport.
    Does not call optimizer.step() — read-only diagnostic.

    Example:
        report = quick_scan(model, lambda m: m(x, y)[1], n_steps=10)
        report.grad_norm_trajectory("blocks.0.attn.out")
    """
    with ResidualScope(model, layer_filter=layer_filter) as scope:
        for step in range(n_steps):
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model)
            loss.backward()
            scope.step(step)
        return scope.report()
