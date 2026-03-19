from __future__ import annotations

import torch
import torch.nn as nn

import synapsefast as sf


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, *, device=None, dtype=None):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D]
        return sf.rms_norm(x, self.weight, eps=self.eps)


class SynapseAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, *, causal: bool = True, backend: str = "auto"):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.backend = backend

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, E]
        B, T, E = x.shape
        qkv = self.qkv(x)  # [B, T, 3E]
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, H, T, D]

        y = sf.attention(q, k, v, causal=self.causal, backend=self.backend)  # [B, H, T, D]
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, E)
        return self.out_proj(y)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_mult: float = 4.0):
        super().__init__()
        hidden = int(embed_dim * hidden_mult)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swi/GEGLU variants are faster with fused kernels; MVP uses GELU.
        x = self.fc1(x)
        x = sf.gelu(x)
        return self.fc2(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        causal: bool = True,
        eps: float = 1e-5,
        hidden_mult: float = 4.0,
        backend: str = "auto",
    ):
        super().__init__()
        self.attn_norm = RMSNorm(embed_dim, eps=eps)
        self.attn = SynapseAttention(embed_dim, num_heads, causal=causal, backend=backend)
        self.ff_norm = RMSNorm(embed_dim, eps=eps)
        self.ff = FeedForward(embed_dim, hidden_mult=hidden_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual blocks.
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x


class ToyGPT(nn.Module):
    """
    Minimal causal LM for demos/benchmarks.

    This is intentionally small and uses `synapsefast` attention/norm/gelu
    so performance improvements surface in training.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        seq_len: int,
        *,
        hidden_mult: float = 4.0,
        backend: str = "auto",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    causal=True,
                    hidden_mult=hidden_mult,
                    backend=backend,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T]
        if tokens.dim() != 2:
            raise ValueError("tokens must be rank-2 [B, T]")
        B, T = tokens.shape
        if T > self.seq_len:
            raise ValueError("sequence length exceeds model seq_len")

        x = self.tok_emb(tokens) + self.pos_emb[:, :T, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.lm_head(x)  # [B, T, vocab]
