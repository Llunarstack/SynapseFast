from __future__ import annotations

import torch
import torch.nn.functional as F


def attention_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool) -> torch.Tensor:
    # PyTorch expects [B, H, T, D] for multi-head.
    # Use dropout_p=0 for determinism in MVP.
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal)


def rms_norm_torch(x: torch.Tensor, weight: torch.Tensor, *, eps: float) -> torch.Tensor:
    # x: [..., D], weight: [D]
    mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(mean_sq + eps)
    return x_norm * weight


def layer_norm_torch(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, *, eps: float
) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
    x_norm = (x - mean) * torch.rsqrt(var + eps)
    return x_norm * weight + bias


def gelu_torch(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def matmul_bias_torch(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    y = a @ b
    return y + bias


def kv_cache_prefill_torch(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    *,
    start_pos: int,
) -> None:
    # k_cache/v_cache: [B, H, L, D], k_new/v_new: [B, H, T, D]
    T = k_new.shape[2]
    k_cache[:, :, start_pos : start_pos + T, :].copy_(k_new)
    v_cache[:, :, start_pos : start_pos + T, :].copy_(v_new)


def kv_cache_decode_torch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    pos: int,
    causal: bool = True,
) -> torch.Tensor:
    # q: [B, H, 1, D], attend to keys up to pos (inclusive).
    # Output: [B, H, 1, D]
    # Important: we already slice keys to enforce the causal constraint.
    # Calling SDPA with `is_causal=True` would apply an additional causal mask
    # relative to q_len (which is 1 for decode), leading to incorrect behavior.
    if causal:
        k = k_cache[:, :, : pos + 1, :]
        v = v_cache[:, :, : pos + 1, :]
        return attention_torch(q, k, v, causal=False)
    # Non-causal decode: attend over the full cache.
    return attention_torch(q, k_cache, v_cache, causal=False)
