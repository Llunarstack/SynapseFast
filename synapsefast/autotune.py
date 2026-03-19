from __future__ import annotations

import contextlib
import io
import json
import os
import platform
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F


Backend = str  # "torch" | "cuda" | "xformers"


def _cuda_time_ms(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _try_xformers_fn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool
) -> Optional[Callable[[], torch.Tensor]]:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from xformers.ops import memory_efficient_attention
        qx = q.transpose(1, 2)  # [B, T, H, D]
        kx = k.transpose(1, 2)
        vx = v.transpose(1, 2)
        bias = None
        if causal:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                from xformers.ops.fmha.attn_bias import LowerTriangularMask

            bias = LowerTriangularMask()

        def call():
            return memory_efficient_attention(qx, kx, vx, attn_bias=bias)

        return call
    except Exception:
        return None


def _cache_dir() -> Path:
    p = os.environ.get("SYNAPSEFAST_CACHE_DIR")
    if p:
        return Path(p)
    return Path.home() / ".synapsefast"


def _cache_path() -> Path:
    return _cache_dir() / "autotune_attention.json"


def _device_key() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    return f"{name} cc={cc[0]}.{cc[1]} torch={torch.__version__} cuda={torch.version.cuda} os={platform.platform()}"


def _shape_bucket(T: int) -> int:
    # Bucket sequence lengths so nearby sizes reuse decisions.
    # You can make this finer if you care about specific sizes.
    for b in (64, 128, 256, 512, 1024, 2048, 4096, 8192):
        if T <= b:
            return b
    return int(((T + 8191) // 8192) * 8192)


def _dtype_str(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "fp32"
    return str(dtype)


def _load_cache() -> dict:
    p = _cache_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    d = _cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    _cache_path().write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def autotune_attention_backend(
    *,
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype,
    causal: bool,
    warmup: int = 20,
    iters: int = 100,
) -> Backend:
    """
    Benchmark available attention backends and cache the fastest choice.
    Returns one of: "torch", "cuda", "xformers".
    """
    if not torch.cuda.is_available():
        return "torch"

    cache = _load_cache()
    devk = _device_key()
    key = f"{devk} | B={B} H={H} T<={_shape_bucket(T)} D={D} dtype={_dtype_str(dtype)} causal={bool(causal)}"
    if key in cache:
        return cache[key]["best"]

    q = torch.randn(B, H, T, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, T, D, device="cuda", dtype=dtype)
    v = torch.randn(B, H, T, D, device="cuda", dtype=dtype)

    # Torch SDPA
    def torch_fn():
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)

    # SynapseFast CUDA path (may fall back internally if extension unavailable)
    import synapsefast as sf

    def sf_cuda_fn():
        return sf.attention(q, k, v, causal=causal, backend="cuda")

    xformers_fn = _try_xformers_fn(q, k, v, causal=causal)

    timings: Dict[str, float] = {}
    timings["torch"] = _cuda_time_ms(torch_fn, warmup=warmup, iters=iters)
    timings["cuda"] = _cuda_time_ms(sf_cuda_fn, warmup=warmup, iters=iters)
    if xformers_fn is not None:
        timings["xformers"] = _cuda_time_ms(xformers_fn, warmup=min(10, warmup), iters=min(50, iters))

    best = min(timings.items(), key=lambda kv: kv[1])[0]
    cache[key] = {
        "best": best,
        "timings_ms": timings,
        "ts": time.time(),
    }
    _save_cache(cache)
    return best
