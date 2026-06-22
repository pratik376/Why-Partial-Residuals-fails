# ResidualScope

Model-agnostic diagnostics for transformer internals — gradient norm tracking, hidden-state representation drift, dead-neuron detection, and cross-configuration comparison.

This package is the generalized, pip-installable extraction of the diagnostic code originally written to investigate and explain the **2.13×–2.36× asymmetry between attention-only and FFN-only residual configurations** described in [*Why Partial Residuals Fail: Asymmetric Pathway Necessity in Transformer Language Models*](https://github.com/pratik376/why-partial-residuals-fail). The four diagnostics here — gradient starvation, hidden-norm drift, dead-neuron fraction, and cross-config comparison — are exactly the ones used to discover that finding, rewritten to work on **any** PyTorch `nn.Module**, not just the original `ResidualGPT` implementation.

## Why this exists

While running the residual-pathway ablations, the same four diagnostic snippets kept getting hand-copied between notebooks: a `gradient_snapshot()` function, a hidden-norm tracker, a dead-neuron counter, and ad-hoc comparison bar charts. ResidualScope packages those as a reusable hook-based tool so the next architecture question — "does removing X change gradient flow / representation drift / dead units?" — doesn't require rebuilding the instrumentation from scratch.

## Installation

```bash
pip install residualscope
```

Or from source:

```bash
git clone https://github.com/pratik376/residualscope
cd residualscope
pip install -e .
```

**Requirements:** Python ≥3.9, PyTorch ≥1.13, matplotlib ≥3.5, numpy ≥1.21.

## Quick start

```python
from residualscope import ResidualScope

scope = ResidualScope(model)  # tracks every nn.Linear by default

for step in range(n_steps):
    loss = model(x, y)[1]
    loss.backward()
    scope.step(step)          # record AFTER backward(), BEFORE optimizer.step()
    optimizer.step()
    optimizer.zero_grad()

report = scope.report()
report.hidden_norm_growth_ratio("blocks.3.mlp.down")   # e.g. 14.03
report.grad_norm_trajectory("blocks.0.mlp.fc1")        # [0.114, 0.098, 0.0, 0.0, ...]
scope.close()
```

Or use the context-manager form, which calls `close()` for you:

```python
with ResidualScope(model) as scope:
    ...
report = scope.report()
```

### One-shot health check on a loaded checkpoint

```python
from residualscope import quick_scan

report = quick_scan(model, loss_fn=lambda m: m(x, y)[1], n_steps=5)
```

`quick_scan` runs forward+backward passes and records diagnostics but **never calls `optimizer.step()`** — it's a read-only probe, not a training loop.

### Comparing configurations (the core use case)

This is the comparison shape that originally surfaced the AttnOnly/FFNOnly asymmetry: same diagnostic, same layer, different model variants, side by side.

```python
from residualscope import compare, plot_comparison_bar

reports = {
    "FullResidual": scan_one_config(full_residual_model),
    "AttnOnly":     scan_one_config(attn_only_model),
    "FFNOnly":      scan_one_config(ffn_only_model),
    "NoResidual":   scan_one_config(no_residual_model),
}

comparison = compare(reports)
comparison.summary_table("blocks.0.mlp.fc1")
# [{'config': 'FullResidual', 'final_grad_norm': 0.114, 'hidden_norm_growth_ratio': 1.38, ...},
#  {'config': 'AttnOnly',     'final_grad_norm': 0.0,   'hidden_norm_growth_ratio': 14.03, ...},
#  ...]

fig, ax = plot_comparison_bar(comparison, "blocks.0.mlp.fc1", metric="hidden_norm_growth_ratio")
fig.savefig("comparison.png")
```

### Plotting

```python
from residualscope import plot_summary_dashboard

fig = plot_summary_dashboard(report, suptitle="AttnOnly diagnostics")
fig.savefig("dashboard.png")
```

Individual panels (`plot_gradient_norms`, `plot_hidden_norm_growth`, `plot_dead_neurons`) each accept an optional `ax=` so you can compose your own multi-panel figures.

## Worked example

`examples/attnonly_ffnonly_demo.py` builds a small toggleable-residual transformer (the same `attn_res` / `ffn_res` switch structure as the original `ResidualGPT`) and runs all four configurations:

```bash
python examples/attnonly_ffnonly_demo.py
```

```
Running ResidualScope on 4 toggleable-residual configurations...

  FullResidual   final_loss=0.0218
  AttnOnly       final_loss=0.0272
  FFNOnly        final_loss=0.5818
  NoResidual     final_loss=0.5955

Final gradient norm at earliest layer (blocks.0.mlp.fc1):
  FullResidual   0.052371
  AttnOnly       0.115669
  FFNOnly        0.484583
  NoResidual     0.395563
```

Even at this toy scale (4 layers, 150 steps), AttnOnly's final loss tracks FullResidual while FFNOnly tracks NoResidual — the same qualitative asymmetry reported in the paper at 10M/124M scale. To point ResidualScope at one of the **original** checkpoints from that research instead of a toy model, see the `load_real_checkpoint_example()` template in the same file.

## API reference

| Function / class | Purpose |
|---|---|
| `ResidualScope(model, layer_filter=None, track_activations=True)` | Registers hooks; call `.step(step)` every training step, `.report()` when done. |
| `quick_scan(model, loss_fn, n_steps=5, layer_filter=None)` | One-shot diagnostic scan without a full training loop. Does not call `optimizer.step()`. |
| `ScopeReport` | Holds all recorded `LayerSnapshot`s. Key methods: `.grad_norm_trajectory(layer)`, `.hidden_norm_growth_ratio(layer)`, `.dead_neuron_fraction_final(layer)`. |
| `compare({name: report, ...})` | Builds a `ComparisonResult` for side-by-side analysis across configurations. |
| `ComparisonResult.summary_table(layer)` | One row per configuration with all four core metrics. |
| `plot_gradient_norms`, `plot_hidden_norm_growth`, `plot_dead_neurons`, `plot_summary_dashboard`, `plot_comparison_bar` | Matplotlib helpers; each returns `(fig, ax)` or accepts `ax=` to compose into your own figure. |

## What this package deliberately does not do

This is a **read-only diagnostic tool**, not a training framework. It does not:
- Run training loops or manage optimizers
- Handle checkpointing or experiment tracking
- Sync to cloud storage
- Assume any specific model architecture (it works via `named_modules()` + hooks, not a hardcoded class)

If you need those things, wire ResidualScope into your own training loop — that's the intended usage pattern, demonstrated in the worked example above.

## Caveat on dead-neuron interpretation

A high dead-neuron fraction is not inherently bad and a low one is not inherently good. In the source research, the healthy, well-converged Full Residual configuration showed *higher* dead-neuron rates (≈82%, indicating learned sparse, selective features) than the collapsed partial-residual configurations (≈51–53%, indicating the model never developed structured features at all and activations stayed close to random-initialization statistics). Always read this metric alongside the loss curve and hidden-norm growth ratio, never in isolation.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

12 tests covering hook lifecycle, diagnostic correctness against hand-computable synthetic cases, and the cross-configuration comparison API.

## Citation

If you use this tool in research, please cite the originating paper:

```bibtex
@misc{patel2026residuals,
  title  = {Why Partial Residuals Fail: Asymmetric Pathway Necessity in Transformer Language Models},
  author = {Patel, Pratik},
  year   = {2026},
  note   = {arXiv preprint}
}
```

## License

MIT
