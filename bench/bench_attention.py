from __future__ import annotations

import argparse
import contextlib
import io
import os
import shutil
import time
from typing import Any

import torch

import synapsefast as sf


def _time_ms(fn, *, warmup: int, iters: int) -> float:
    # CUDA timing needs explicit sync.
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def _get_plan_backend(q: torch.Tensor, *, causal: bool) -> str | None:
    try:
        # Private import: used only for benchmarking visibility.
        import synapsefast._planner as planner

        plan = planner.plan_attention(q, causal=causal)
        backend = plan.get("backend", None)
        return str(backend) if backend is not None else None
    except Exception:
        return None


def bench_one(
    B: int,
    H: int,
    T: int,
    D: int,
    *,
    causal: bool,
    dtype: torch.dtype,
    device: str,
    backend: str,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)

    # Note: the planner may override the backend in "auto" mode depending on env vars.
    plan_backend = _get_plan_backend(q, causal=causal)

    # Determine whether the custom CUDA extension is actually loadable in this process.
    cuda_ext_loaded: bool | None = None
    try:
        import synapsefast._cuda_ops as co

        # If cl isn't present, _get_ext() will fail-fast without JIT compilation.
        # Still safe to call and gives us visibility into whether the CUDA path is real.
        cuda_ext_loaded = co._get_ext() is not None
    except Exception:
        cuda_ext_loaded = False

    # SynapseFast
    sf_ms = _time_ms(
        lambda: sf.attention(q, k, v, causal=causal, backend=backend), warmup=warmup, iters=iters
    )

    # PyTorch SDPA
    def torch_sdpa():
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal
        )

    torch_ms = _time_ms(torch_sdpa, warmup=warmup, iters=iters)

    out: dict[str, Any] = {
        "sf_ms": sf_ms,
        "torch_ms": torch_ms,
        "plan_backend": plan_backend,
        "sf_backend_arg": backend,
        "cuda_ext_loaded": cuda_ext_loaded,
    }

    # Optional xFormers benchmark
    try:
        # xFormers import can emit noisy warnings/errors if Triton isn't available.
        # We silence stdout/stderr for import, but still time the kernel if it works.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import xformers  # noqa: F401
            from xformers.ops import memory_efficient_attention

        # xFormers expects [B, T, H, D]
        qx = q.transpose(1, 2)  # [B, T, H, D]
        kx = k.transpose(1, 2)
        vx = v.transpose(1, 2)

        # Build attn bias for causal/non-causal
        if causal:
            from xformers.ops.fmha.attn_bias import LowerTriangularMask

            bias = LowerTriangularMask()
        else:
            bias = None

        def xformers_call():
            return memory_efficient_attention(qx, kx, vx, attn_bias=bias)

        out["xformers_ms"] = _time_ms(xformers_call, warmup=min(5, warmup), iters=min(20, iters))
    except Exception:
        pass

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--T_list", default="128,256,512")
    parser.add_argument("--backend", choices=["auto", "torch", "cuda"], default="auto")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    Ts = [int(x) for x in args.T_list.split(",") if x.strip()]

    use_custom_env = os.environ.get("SYNAPSEFAST_USE_CUSTOM_CUDA", None)
    cl_present = shutil.which("cl") is not None

    print(
        f"Device={args.device}, dtype={args.dtype}, causal={args.causal}, B={args.B}, H={args.H}, D={args.D} "
        f"backend_arg={args.backend}, warmup={args.warmup}, iters={args.iters} "
        f"SYNAPSEFAST_USE_CUSTOM_CUDA={use_custom_env!r} cl_present={cl_present}"
    )
    for T in Ts:
        res = bench_one(
            args.B,
            args.H,
            T,
            args.D,
            causal=args.causal,
            dtype=dtype,
            device=args.device,
            backend=args.backend,
            warmup=args.warmup,
            iters=args.iters,
        )
        sf_ms = res["sf_ms"]
        torch_ms = res["torch_ms"]
        speedup = torch_ms / sf_ms if sf_ms > 0 else float("inf")
        line = f"T={T:4d}  sf={sf_ms:8.3f}ms  torch={torch_ms:8.3f}ms  speedup={speedup:5.2f}x"
        if "xformers_ms" in res:
            line += f"  xformers={res['xformers_ms']:8.3f}ms"
        if "plan_backend" in res and res["plan_backend"] is not None:
            line += f"  plan={res['plan_backend']}"
        print(line)


if __name__ == "__main__":
    main()
