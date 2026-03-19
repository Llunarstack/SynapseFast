from __future__ import annotations

import os
from functools import lru_cache
import shutil
from typing import Any, Optional

import torch


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    # If torch has CUDA support but nvcc/kernels are missing, compilation may fail at runtime;
    # we treat that as "not available" by catching import/compile errors in `_get_ext()`.
    return torch.cuda.is_available()


def _sources() -> list[str]:
    here = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join(here, os.pardir))
    src_root = os.path.join(root, "csrc", "synapsefast_cuda")
    return [
        os.path.join(src_root, "bindings.cpp"),
        os.path.join(src_root, "attention.cu"),
        os.path.join(src_root, "kv_cache.cu"),
        os.path.join(src_root, "rms_norm.cu"),
        os.path.join(src_root, "layer_norm.cu"),
        os.path.join(src_root, "activations.cu"),
        os.path.join(src_root, "bias_ops.cu"),
    ]


def _extra_cuda_cflags() -> list[str]:
    return [
        "-lineinfo",
        "--use_fast_math",
        # CUDA 13.x / CCCL headers require the standard conforming preprocessor.
        "-Xcompiler",
        "/Zc:preprocessor",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ]


@lru_cache(maxsize=1)
def _get_ext() -> Optional[Any]:
    if not cuda_available():
        return None

    # torch JIT builds on Windows require MSVC host tooling (cl.exe).
    # If it's not present, fail fast so the Python API can fall back cleanly.
    if shutil.which("cl") is None:
        return None

    try:
        from torch.utils.cpp_extension import load

        # Runtime compilation (JIT). This keeps install from hard-failing when nvcc isn't present.
        return load(
            name="synapsefast_cuda_ext",
            sources=_sources(),
            extra_cuda_cflags=_extra_cuda_cflags(),
            extra_cflags=["/O2"],
            with_cuda=True,
            verbose=False,
        )
    except Exception as e:
        # Compilation can fail when nvcc/host compiler aren't available.
        debug = os.environ.get("SYNAPSEFAST_DEBUG_CUDA_EXT", "").strip().lower()
        if debug in ("1", "true", "yes"):
            import traceback

            print("synapsefast: CUDA extension build failed:", repr(e))
            traceback.print_exc()
        return None


def _require_ext() -> Any:
    ext = _get_ext()
    if ext is None:
        raise RuntimeError(
            "synapsefast CUDA extension is unavailable (nvcc not present or compilation failed)."
        )
    return ext


def cuda_ext_loaded() -> bool:
    # Fail-fast + cached (when `cl` isn't on PATH, _get_ext() returns None quickly).
    return _get_ext() is not None


def attention_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool, config: dict
) -> torch.Tensor:
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("CUDA attention MVP supports only fp16/bf16 tensors.")
    ext = _require_ext()
    # Config is a placeholder for future autotuning; currently ignored.
    return ext.attention_forward(q, k, v, causal)


def rms_norm_forward(x: torch.Tensor, weight: torch.Tensor, *, eps: float) -> torch.Tensor:
    ext = _require_ext()
    return ext.rms_norm_forward(x, weight, eps)


def layer_norm_forward(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, *, eps: float
) -> torch.Tensor:
    ext = _require_ext()
    return ext.layer_norm_forward(x, weight, bias, eps)


def gelu_forward(x: torch.Tensor) -> torch.Tensor:
    ext = _require_ext()
    return ext.gelu_forward(x)


def matmul_bias_forward(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # MVP: rely on PyTorch for GEMM and only run bias add on GPU.
    # This keeps the project scope reasonable while the fused epilogue work is developed.
    y = a @ b
    ext = _require_ext()
    return ext.bias_add_forward(y, bias)


def kv_cache_prefill_forward(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    *,
    start_pos: int,
) -> torch.Tensor:
    ext = _require_ext()
    return ext.kv_cache_prefill_forward(k_cache, v_cache, k_new, v_new, start_pos)


def kv_cache_decode_forward(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    pos: int,
    causal: bool = True,
) -> torch.Tensor:
    ext = _require_ext()
    return ext.kv_cache_decode_forward(q, k_cache, v_cache, pos, causal)
