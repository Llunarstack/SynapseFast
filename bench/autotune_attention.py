from __future__ import annotations

import argparse

import torch

from synapsefast.autotune import autotune_attention_backend


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--causal", action="store_true")
    p.add_argument("--T_list", default="128,256,512,1024")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    Ts = [int(x) for x in args.T_list.split(",") if x.strip()]

    for T in Ts:
        best = autotune_attention_backend(
            B=args.B,
            H=args.H,
            T=T,
            D=args.D,
            dtype=dtype,
            causal=args.causal,
            warmup=args.warmup,
            iters=args.iters,
        )
        print(f"T={T:5d} best={best}")


if __name__ == "__main__":
    main()
