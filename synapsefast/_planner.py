from __future__ import annotations

import os
import torch


def _dtype_str(t: torch.dtype) -> str:
    if t == torch.float16:
        return "fp16"
    if t == torch.bfloat16:
        return "bf16"
    if t == torch.float32:
        return "fp32"
    return str(t)


def _fallback_plan_attention(q: torch.Tensor, *, causal: bool) -> dict:
    # Heuristic: attempt CUDA FlashAttention-like kernel only for fp16/bf16.
    head_dim = q.shape[-1]
    use_custom = os.environ.get("SYNAPSEFAST_USE_CUSTOM_CUDA", "").strip().lower()
    if use_custom not in ("1", "true", "yes"):
        return {"backend": "torch_sdp", "config": {}}
    if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16) and head_dim <= 128 and head_dim % 8 == 0:
        return {
            "backend": "cuda_flash_attention",
            "config": {
                "head_dim": int(head_dim),
                "causal": bool(causal),
                # Block sizes are placeholders; the CUDA kernel may ignore some.
                "q_tile": 64,
                "k_tile": 64,
                "num_warps": 4,
            },
        }
    return {"backend": "torch_sdp", "config": {}}


def plan_attention(q: torch.Tensor, *, causal: bool) -> dict:
    try:
        from ._planner_ext import plan_attention as rust_plan_attention  # type: ignore

        plan = rust_plan_attention(
            q.shape[0],
            q.shape[1],
            q.shape[2],
            q.shape[3],
            _dtype_str(q.dtype),
            causal,
        )
        # Default behavior: use PyTorch's high-performance SDPA backend unless the
        # user explicitly enables custom CUDA kernels.
        use_custom = os.environ.get("SYNAPSEFAST_USE_CUSTOM_CUDA", "").strip().lower()
        if use_custom not in ("1", "true", "yes"):
            if str(plan.get("backend", "")).startswith("cuda_"):
                return {"backend": "torch_sdp", "config": {}}
        return plan
    except Exception:
        return _fallback_plan_attention(q, causal=causal)


def plan_norm(x: torch.Tensor, weight: torch.Tensor, *, eps: float, norm_type: str) -> dict:
    # MVP: CUDA kernels are implemented for RMSNorm; LayerNorm falls back to torch unless kernel exists.
    if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
        if norm_type == "rms":
            return {"backend": "cuda_rms_norm"}
        if norm_type == "layer":
            return {"backend": "cuda_layer_norm"}
    return {"backend": "torch"}


def plan_activation(x: torch.Tensor) -> dict:
    if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
        return {"backend": "cuda_gelu"}
    return {"backend": "torch"}


def plan_matmul(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> dict:
    # Placeholder: we only optimize bias+activation kernels when available.
    if (
        a.is_cuda
        and b.is_cuda
        and bias is not None
        and a.dtype in (torch.float16, torch.bfloat16)
        and b.dtype == a.dtype
        and bias.dtype == a.dtype
    ):
        return {"backend": "cuda_matmul_bias"}
    return {"backend": "torch"}
