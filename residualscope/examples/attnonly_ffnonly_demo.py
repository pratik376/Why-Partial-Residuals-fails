"""
Worked example: reproducing the AttnOnly / FFNOnly asymmetry diagnostic.

This script demonstrates ResidualScope on a toggleable-residual transformer
block, the same structural pattern used in the "Why Partial Residuals Fail"
research. It does NOT require the original 10M/124M checkpoints (those are
multi-GB and not bundled with this package) -- instead it builds small
local models with the residual switch wired up exactly as in that project,
and reproduces the SHAPE of the diagnostic finding at toy scale:

    - Gradient norm at the earliest layer collapses to ~0 when a residual
      connection is removed (gradient starvation).
    - Hidden-state norm growth diverges between AttnOnly-style and
      FFNOnly-style configurations even when final loss is similar.

To reproduce the diagnostics on YOUR OWN checkpoints from the original
research (or any other transformer), see `load_real_checkpoint_example()`
below, which shows the exact pattern for pointing ResidualScope at a real
nn.Module loaded from a .pt file.

Run with:
    python examples/attnonly_ffnonly_demo.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from residualscope import ResidualScope, compare, plot_comparison_bar, plot_summary_dashboard


# ──────────────────────────────────────────────────────────────────────────
# Minimal toggleable-residual transformer block
# (structurally identical to the ResidualRoutingBlock switch used in the
#  source research -- attn_res / ffn_res flags control each skip path)
# ──────────────────────────────────────────────────────────────────────────

class ToyAttention(nn.Module):
    def __init__(self, dim, heads=2):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(attn)


class ToyMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class ToggleableBlock(nn.Module):
    """
    attn_res / ffn_res mirror the use_attention_skip / use_mlp_skip flags
    from the ResidualGPT experiments. Setting attn_res=False and
    ffn_res=True reproduces "FFNOnly"; attn_res=True, ffn_res=False
    reproduces "AttnOnly".
    """

    def __init__(self, dim, attn_res: bool, ffn_res: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = ToyAttention(dim)
        self.mlp = ToyMLP(dim)
        self.attn_res = attn_res
        self.ffn_res = ffn_res

    def forward(self, x):
        a = self.attn(self.ln1(x))
        x = x + a if self.attn_res else a
        m = self.mlp(self.ln2(x))
        x = x + m if self.ffn_res else m
        return x


class ToyTransformer(nn.Module):
    def __init__(self, dim=32, n_layers=4, attn_res=True, ffn_res=True):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ToggleableBlock(dim, attn_res, ffn_res) for _ in range(n_layers)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# ──────────────────────────────────────────────────────────────────────────
# Demo: run all four configurations and compare
# ──────────────────────────────────────────────────────────────────────────

CONFIGS = {
    "FullResidual": dict(attn_res=True, ffn_res=True),
    "AttnOnly": dict(attn_res=True, ffn_res=False),
    "FFNOnly": dict(attn_res=False, ffn_res=True),
    "NoResidual": dict(attn_res=False, ffn_res=False),
}


def run_config(name: str, n_steps: int = 150, dim: int = 32, seq_len: int = 16,
                batch_size: int = 8, lr: float = 1e-3, seed: int = 1337):
    torch.manual_seed(seed)
    model = ToyTransformer(dim=dim, n_layers=4, **CONFIGS[name])
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    x = torch.randn(batch_size, seq_len, dim)
    target = torch.randn(batch_size, seq_len, dim)

    # Track only the FFN's first linear layer per block -- this is the
    # exact tracking granularity ("Layer 0 fc1") used to surface the
    # gradient-starvation signature in the original research.
    layer_filter = lambda name, mod: isinstance(mod, nn.Linear) and "fc1" in name

    with ResidualScope(model, layer_filter=layer_filter) as scope:
        for step in range(n_steps):
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = F.mse_loss(out, target)
            loss.backward()
            scope.step(step)
            opt.step()
        final_loss = float(loss.item())

    return scope.report(), final_loss


def main():
    print("Running ResidualScope on 4 toggleable-residual configurations...")
    print("(toy scale: dim=32, 4 layers, 150 steps -- for full-scale")
    print(" reproduction see the original paper's 10M/124M experiments)\n")

    reports = {}
    final_losses = {}
    for name in CONFIGS:
        report, final_loss = run_config(name)
        reports[name] = report
        final_losses[name] = final_loss
        print(f"  {name:<14} final_loss={final_loss:.4f}")

    comparison = compare(reports)

    # The earliest layer is where gradient starvation is most visible
    earliest_layer = reports["FullResidual"].layer_names[0]

    print(f"\nFinal gradient norm at earliest layer ({earliest_layer}):")
    grad_table = comparison.final_grad_norm_table(earliest_layer)
    for name, val in grad_table.items():
        flag = " <- starved" if (val is not None and val < 1e-4) else ""
        print(f"  {name:<14} {val:.6f}{flag}" if val is not None else f"  {name:<14} N/A")

    print(f"\nHidden-norm growth ratio at earliest layer ({earliest_layer}):")
    growth_table = comparison.hidden_norm_growth_table(earliest_layer)
    for name, val in growth_table.items():
        print(f"  {name:<14} {val:.2f}x" if val is not None else f"  {name:<14} N/A")

    # Save the comparison figure
    fig, ax = plot_comparison_bar(
        comparison, earliest_layer, metric="hidden_norm_growth_ratio",
        title="Hidden-norm growth ratio by residual configuration (toy demo)"
    )
    fig.savefig("attnonly_ffnonly_demo.png", dpi=150, bbox_inches="tight")
    print("\nSaved comparison figure to attnonly_ffnonly_demo.png")


def load_real_checkpoint_example():
    """
    Template showing how to point ResidualScope at a REAL checkpoint
    from the original research (or any other saved model). Not run by
    default since it requires the original .pt files, which are not
    bundled with this package -- see the paper's repository for the
    full checkpoint files.
    """
    # checkpoint = torch.load("runs/AttnOnly_std/checkpoints/best.pt",
    #                          map_location="cpu", weights_only=False)
    # model = ResidualTransformerLM(...)  # your model class
    # model.load_state_dict(checkpoint["model_state"])
    # model.eval()
    #
    # scope = ResidualScope(model)
    # with torch.enable_grad():
    #     logits, loss = model(batch_x, batch_y)
    #     loss.backward()
    #     scope.step(step=0)
    # report = scope.report()
    raise NotImplementedError(
        "This is a template -- adapt the commented code above to your "
        "own checkpoint and model class."
    )


if __name__ == "__main__":
    main()
