"""
How to use ResidualScope with the paper's actual checkpoints.

The gradient-starvation signature from the paper requires a real language
modeling task at sufficient scale — it does not emerge on toy data. This
script shows the exact pattern for pointing ResidualScope at a real checkpoint
from the 10M or 124M experiments.

To run:
    1. Download checkpoints from the paper's results/ folder
    2. Update the paths below
    3. python examples/attnonly_ffnonly_demo.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from residualscope import ResidualScope, compare, plot_comparison_bar


# ── Minimal ResidualGPT block ──────────

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(out.transpose(1, 2).contiguous().view(B, T, D))


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, n_heads, attn_res: bool, ffn_res: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.mlp = MLP(dim)
        self.attn_res = attn_res
        self.ffn_res = ffn_res

    def forward(self, x):
        a = self.attn(self.ln1(x))
        x = x + a if self.attn_res else a
        m = self.mlp(self.ln2(x))
        x = x + m if self.ffn_res else m
        return x


class ResidualGPT(nn.Module):
    """Matches the 10M config: vocab=65, dim=384, heads=6, layers=6, ctx=256."""
    def __init__(self, vocab=65, dim=384, n_heads=6, n_layers=6,
                 ctx=256, attn_res=True, ffn_res=True):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(ctx, dim)
        self.blocks = nn.ModuleList([
            Block(dim, n_heads, attn_res, ffn_res) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.tok_emb.weight = self.head.weight  # weight tying

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


# ── Real checkpoint usage ──────────────────────────────────────────────────

def diagnose_checkpoint(checkpoint_path: str, config: dict, batch_x, batch_y):
    """
    Load a checkpoint and run one forward+backward pass to get diagnostics.

    Args:
        checkpoint_path: path to a .pt checkpoint saved during training
        config: dict with attn_res and ffn_res booleans
        batch_x, batch_y: token tensors of shape (B, T)

    Returns:
        ScopeReport with gradient norms and hidden-state norms
    """
    model = ResidualGPT(**config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.train()  # need train mode for gradients

    scope = ResidualScope(model)
    logits, loss = model(batch_x, batch_y)
    loss.backward()
    scope.step(step=0)
    scope.close()

    return scope.report()


# ── Example: compare AttnOnly vs FFNOnly from paper checkpoints ────────────

if __name__ == "__main__":
    # Update these paths to point at your actual checkpoints
    CHECKPOINTS = {
        "AttnOnly": "results/10M_ResidualGPT/checkpoints/AttnOnly_std_seed42.json",
        "FFNOnly":  "results/10M_ResidualGPT/checkpoints/FFNOnly_std_seed42.json",
    }

    CONFIGS = {
        "AttnOnly": dict(attn_res=True,  ffn_res=False),
        "FFNOnly":  dict(attn_res=False, ffn_res=True),
    }

    # Dummy batch — replace with a real batch from your validation set
    batch_x = torch.randint(0, 65, (4, 256))
    batch_y = torch.randint(0, 65, (4, 256))

    reports = {}
    for name, path in CHECKPOINTS.items():
        try:
            reports[name] = diagnose_checkpoint(path, CONFIGS[name], batch_x, batch_y)
            layer = reports[name].layer_names[0]
            grad = reports[name].grad_norm_trajectory(layer)
            print(f"{name}: grad_norm={grad[0]:.6f}, hidden_growth={reports[name].hidden_norm_growth_ratio(layer):.2f}x")
        except FileNotFoundError:
            print(f"{name}: checkpoint not found at {path}")
            print("  Download checkpoints from the paper's results/ folder first.")

    # Expected output from the paper's 10M experiments (seed 1337, step 300+):
    #   FullResidual:  grad_norm ≈ 0.114  (sustained)
    #   AttnOnly:      grad_norm ≈ 0.000  (starved from step 300)
    #   FFNOnly:       grad_norm ≈ 0.000  (starved from step 300)
    #   NoResidual:    grad_norm ≈ 0.000  (starved from step 300)
    #
    # The asymmetry shows up in hidden-state norm growth, not gradient norms:
    #   AttnOnly:  14.03x  (high drift, still converging)
    #   FFNOnly:    5.37x  (high drift, collapsed to floor)
    #   FullRes:    1.38x  (stable)
