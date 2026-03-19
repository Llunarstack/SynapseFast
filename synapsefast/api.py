from __future__ import annotations

import torch

from ._cuda_ops import cuda_available
from .env import env_flag
from ._planner import plan_attention, plan_norm, plan_activation, plan_matmul
from .autotune import autotune_attention_backend
from ._torch_reference import (
    attention_torch,
    rms_norm_torch,
    layer_norm_torch,
    gelu_torch,
    matmul_bias_torch,
    kv_cache_prefill_torch,
    kv_cache_decode_torch,
)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    backend: str = "auto",
) -> torch.Tensor:
    """
    Memory-efficient attention API (correctness-first).

    Expected shapes:
    - `q`, `k`, `v`: [batch, heads, seqlen, head_dim]
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must be rank-4 tensors: [B, H, T, D].")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have identical shapes [B, H, T, D] (MVP supports self-attention).")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q/k/v must be on the same device.")

    backend = backend.lower()
    if backend == "torch":
        return attention_torch(q, k, v, causal=causal)

    plan = plan_attention(q, causal=causal)
    if env_flag("SYNAPSEFAST_LOG_BACKEND", default=False):
        try:
            print(f"[synapsefast] attention backend={plan.get('backend')} requested={backend}")
        except Exception:
            pass
    if backend == "cuda":
        if cuda_available() and plan["backend"].startswith("cuda_"):
            from ._cuda_ops import attention_forward

            from ._cuda_ops import cuda_ext_loaded

            if cuda_ext_loaded():
                try:
                    return attention_forward(q, k, v, causal=causal, config=plan.get("config", {}))
                except RuntimeError:
                    pass
            # CUDA requested, but extension isn't available -> fallback.
            return attention_torch(q, k, v, causal=causal)
        return attention_torch(q, k, v, causal=causal)

    # backend == "auto"
    if env_flag("SYNAPSEFAST_AUTOTUNE", default=False) and q.is_cuda:
        # Autotune chooses the fastest among torch/cuda/xformers and caches it.
        best = autotune_attention_backend(
            B=int(q.shape[0]),
            H=int(q.shape[1]),
            T=int(q.shape[2]),
            D=int(q.shape[3]),
            dtype=q.dtype,
            causal=causal,
        )
        if best == "torch":
            return attention_torch(q, k, v, causal=causal)
        if best == "xformers":
            try:
                from xformers.ops import memory_efficient_attention

                if causal:
                    from xformers.ops.fmha.attn_bias import LowerTriangularMask

                    bias = LowerTriangularMask()
                else:
                    bias = None
                qx = q.transpose(1, 2)
                kx = k.transpose(1, 2)
                vx = v.transpose(1, 2)
                out = memory_efficient_attention(qx, kx, vx, attn_bias=bias)
                return out.transpose(1, 2)
            except Exception:
                return attention_torch(q, k, v, causal=causal)
        # best == "cuda"
        return attention(q, k, v, causal=causal, backend="cuda")

    # If custom CUDA is enabled globally, we still default to torch SDPA unless
    # explicitly forced. This ensures "auto" is never slower than torch by default.
    # Use `backend="cuda"` (or set SYNAPSEFAST_FORCE_CUSTOM_CUDA=1) to force custom.
    if env_flag("SYNAPSEFAST_USE_CUSTOM_CUDA", default=False) and not env_flag(
        "SYNAPSEFAST_FORCE_CUSTOM_CUDA", default=False
    ):
        return attention_torch(q, k, v, causal=causal)

    if cuda_available() and plan["backend"] == "cuda_flash_attention":
        from ._cuda_ops import attention_forward
        from ._cuda_ops import cuda_ext_loaded

        if cuda_ext_loaded():
            try:
                return attention_forward(q, k, v, causal=causal, config=plan["config"])
            except RuntimeError:
                pass

    return attention_torch(q, k, v, causal=causal)


def kv_cache_prefill(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    *,
    start_pos: int,
) -> None:
    """Write a prefill segment into `k_cache` / `v_cache` in-place."""
    if cuda_available() and k_cache.is_cuda and k_cache.dtype in (torch.float16, torch.bfloat16):
        try:
            from ._cuda_ops import kv_cache_prefill_forward

            _ = kv_cache_prefill_forward(k_cache, v_cache, k_new, v_new, start_pos=start_pos)
            return None
        except RuntimeError:
            pass
    return kv_cache_prefill_torch(k_cache, v_cache, k_new, v_new, start_pos=start_pos)


def kv_cache_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    pos: int,
    causal: bool = True,
) -> torch.Tensor:
    """Decode one token at position `pos` (correctness-first)."""
    if cuda_available() and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        try:
            from ._cuda_ops import kv_cache_decode_forward

            return kv_cache_decode_forward(q, k_cache, v_cache, pos=pos, causal=causal)
        except RuntimeError:
            pass
    return kv_cache_decode_torch(q, k_cache, v_cache, pos=pos, causal=causal)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, *, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm over the last dimension."""
    plan = plan_norm(x, weight, eps=eps, norm_type="rms")
    if cuda_available() and plan["backend"] == "cuda_rms_norm":
        from ._cuda_ops import rms_norm_forward

        try:
            return rms_norm_forward(x, weight, eps=eps)
        except RuntimeError:
            pass
    return rms_norm_torch(x, weight, eps=eps)


def layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, *, eps: float = 1e-5
) -> torch.Tensor:
    plan = plan_norm(x, weight, eps=eps, norm_type="layer")
    if cuda_available() and plan["backend"] == "cuda_layer_norm":
        from ._cuda_ops import layer_norm_forward

        try:
            return layer_norm_forward(x, weight, bias, eps=eps)
        except RuntimeError:
            pass
    return layer_norm_torch(x, weight, bias, eps=eps)


def gelu(x: torch.Tensor) -> torch.Tensor:
    plan = plan_activation(x)
    if cuda_available() and plan["backend"] == "cuda_gelu":
        from ._cuda_ops import gelu_forward

        try:
            return gelu_forward(x)
        except RuntimeError:
            pass
    return gelu_torch(x)


def matmul_bias(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    plan = plan_matmul(a, b, bias)
    # MVP: compute matmul in PyTorch, then run bias/activation in CUDA when possible.
    if cuda_available() and plan["backend"] == "cuda_matmul_bias":
        from ._cuda_ops import matmul_bias_forward

        try:
            return matmul_bias_forward(a, b, bias)
        except RuntimeError:
            pass
    return matmul_bias_torch(a, b, bias)
