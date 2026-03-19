from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F

from synapsefast.nn import ToyGPT
from synapsefast.train import TrainConfig, Trainer


class RandomTokenDataset:
    def __init__(self, vocab_size: int, seq_len: int, *, device: str):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.device = device

    def get_batch(self, batch_size: int):
        # Tokens are random; loss is still well-defined and gradients exist.
        x = torch.randint(0, self.vocab_size, (batch_size, self.seq_len), device=self.device)
        y = x.roll(shifts=-1, dims=1)
        return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--hidden_mult", type=float, default=4.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--sf_backend", choices=["auto", "torch", "cuda"], default="auto")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="runs")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")

    # Head dim is what your attention kernels key off of (fast path is D==64).
    head_dim = args.embed_dim // args.num_heads
    if args.embed_dim % args.num_heads != 0:
        raise SystemExit("embed_dim must be divisible by num_heads")
    if device == "cuda" and (args.sf_backend in ("auto", "cuda")):
        custom_env = os.environ.get("SYNAPSEFAST_USE_CUSTOM_CUDA", None)
        print(f"[synapsefast] head_dim={head_dim} custom_env={custom_env!r}")
        if head_dim != 64:
            print(
                "[synapsefast] Note: current custom CUDA attention fast path is specialized for head_dim=64."
            )

    model = ToyGPT(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        seq_len=args.seq_len,
        hidden_mult=args.hidden_mult,
        backend=args.sf_backend,
    )

    dataset = RandomTokenDataset(args.vocab_size, args.seq_len, device=device)

    def batch_fn(bs: int):
        return dataset.get_batch(bs)

    def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B,T,V], targets: [B,T]
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    cfg = TrainConfig(
        device=device,
        dtype=args.dtype,
        seed=args.seed,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
        grad_clip=args.grad_clip,
        grad_accum_steps=args.grad_accum,
        compile=args.compile,
        out_dir=args.out_dir,
        run_name=args.run_name,
        resume=args.resume,
        log_every=10,
        ckpt_every=50,
    )

    trainer = Trainer(cfg=cfg, model=model, batch_fn=batch_fn, loss_fn=loss_fn, batch_size=args.batch_size)
    trainer.train()


if __name__ == "__main__":
    main()
