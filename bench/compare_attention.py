from __future__ import annotations

import argparse
import contextlib
import io
import os
import shutil
import json
from typing import Callable, Optional

import torch
import torch.nn.functional as F

import synapsefast as sf


def cuda_time_ms(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> float:
    # Use CUDA events for accurate GPU timing.
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


def get_xformers_fn(
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


def get_synapse_ext_loaded() -> bool:
    try:
        import synapsefast._cuda_ops as co

        return co.cuda_ext_loaded()
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--T_list", default="128,256,512,1024")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    Ts = [int(x) for x in args.T_list.split(",") if x.strip()]

    cl_present = shutil.which("cl") is not None
    ext_loaded = get_synapse_ext_loaded()
    custom_env = os.environ.get("SYNAPSEFAST_USE_CUSTOM_CUDA", None)
    print(
        f"CUDA compare: dtype={args.dtype} causal={args.causal} B={args.B} H={args.H} D={args.D} "
        f"warmup={args.warmup} iters={args.iters} cl_present={cl_present} ext_loaded={ext_loaded} "
        f"SYNAPSEFAST_USE_CUSTOM_CUDA={custom_env!r}"
    )

    results = []
    for T in Ts:
        q = torch.randn(args.B, args.H, T, args.D, device="cuda", dtype=dtype)
        k = torch.randn(args.B, args.H, T, args.D, device="cuda", dtype=dtype)
        v = torch.randn(args.B, args.H, T, args.D, device="cuda", dtype=dtype)

        def torch_fn():
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=args.causal)

        def sf_auto_fn():
            return sf.attention(q, k, v, causal=args.causal, backend="auto")

        def sf_torch_fn():
            return sf.attention(q, k, v, causal=args.causal, backend="torch")

        def sf_cuda_fn():
            return sf.attention(q, k, v, causal=args.causal, backend="cuda")

        xformers_fn = get_xformers_fn(q, k, v, causal=args.causal)

        torch_ms = cuda_time_ms(torch_fn, warmup=args.warmup, iters=args.iters)
        auto_ms = cuda_time_ms(sf_auto_fn, warmup=args.warmup, iters=args.iters)
        sf_torch_ms = cuda_time_ms(sf_torch_fn, warmup=args.warmup, iters=args.iters)
        sf_cuda_ms = cuda_time_ms(sf_cuda_fn, warmup=args.warmup, iters=args.iters)

        line = f"T={T:5d}  torch={torch_ms:8.4f}ms  sf_auto={auto_ms:8.4f}ms  sf_torch={sf_torch_ms:8.4f}ms  sf_cuda={sf_cuda_ms:8.4f}ms"
        if xformers_fn is not None:
            x_ms = cuda_time_ms(xformers_fn, warmup=min(10, args.warmup), iters=min(50, args.iters))
            line += f"  xformers={x_ms:8.4f}ms"
        print(line)
        results.append(
            {
                "T": T,
                "torch_ms": torch_ms,
                "sf_auto_ms": auto_ms,
                "sf_torch_ms": sf_torch_ms,
                "sf_cuda_ms": sf_cuda_ms,
                "xformers_ms": x_ms if xformers_fn is not None else None,
            }
        )

    if args.json_out:
        payload = {
            "meta": {
                "dtype": args.dtype,
                "causal": bool(args.causal),
                "B": args.B,
                "H": args.H,
                "D": args.D,
                "warmup": args.warmup,
                "iters": args.iters,
                "ext_loaded": ext_loaded,
                "cl_present": cl_present,
                "SYNAPSEFAST_USE_CUSTOM_CUDA": custom_env,
            },
            "rows": results,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
