from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


@dataclass
class TrainConfig:
    # Core
    device: str = "cuda"
    dtype: str = "fp16"  # fp16|bf16|fp32
    seed: int = 0

    # Optimization
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    grad_accum_steps: int = 1

    # Schedule
    warmup_steps: int = 100
    total_steps: int = 1000

    # Runtime
    log_every: int = 10
    eval_every: int = 0
    ckpt_every: int = 200
    out_dir: str = "runs"
    run_name: str = ""
    resume: str = ""  # optional checkpoint path to resume from

    # Performance
    compile: bool = False  # torch.compile
    allow_tf32: bool = True


def resolve_amp_dtype(dtype: str) -> torch.dtype:
    dtype = dtype.lower()
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp32":
        return torch.float32
    raise ValueError("dtype must be one of: fp16, bf16, fp32")


def cosine_lr(step: int, *, base_lr: float, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if total_steps <= max(1, warmup_steps):
        return base_lr
    import math

    t = (step - warmup_steps) / (total_steps - warmup_steps)
    t = min(max(t, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


class Checkpointer:
    def __init__(self, out_dir: str, run_name: str):
        self.root = Path(out_dir) / (run_name or _now_ts())
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, step: int) -> Path:
        return self.root / f"ckpt_step_{step:08d}.pt"

    def save(
        self,
        *,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        extra: Optional[dict] = None,
    ) -> Path:
        p = self.path_for(step)
        payload = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, p)
        return p

    def load(
        self,
        path: str | os.PathLike,
        *,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> dict:
        payload = torch.load(path, map_location="cpu")
        model.load_state_dict(payload["model"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer"])
        return payload


class Trainer:
    """
    A minimal, dependency-free trainer inspired by:
    - nanoGPT (tight loop, speed knobs)
    - PyTorch Lightning (structured config, checkpointing)
    - HF Trainer (logging + schedules)
    """

    def __init__(
        self,
        *,
        cfg: TrainConfig,
        model: torch.nn.Module,
        batch_fn: Callable[[int], Tuple[torch.Tensor, torch.Tensor]],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        batch_size: int,
    ):
        self.cfg = cfg
        self.model = model
        self.batch_fn = batch_fn
        self.loss_fn = loss_fn
        self.batch_size = int(batch_size)

        if cfg.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")

        self.device = torch.device(cfg.device)
        self.amp_dtype = resolve_amp_dtype(cfg.dtype)
        self.use_amp = self.device.type == "cuda" and self.amp_dtype in (torch.float16, torch.bfloat16)

        if self.device.type == "cuda" and cfg.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        set_seed(cfg.seed)

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )

        self.scaler = torch.amp.GradScaler(enabled=(self.use_amp and self.amp_dtype == torch.float16))
        self.ckpt = Checkpointer(cfg.out_dir, cfg.run_name)

        self.start_step = 0
        if cfg.resume:
            payload = self.ckpt.load(cfg.resume, model=self.model, optimizer=self.optimizer)
            self.start_step = int(payload.get("step", 0)) + 1
            print(f"resumed from {cfg.resume} at step={self.start_step}")

        if cfg.compile:
            self.model = torch.compile(self.model, mode="max-autotune", fullgraph=False)

    def train(self, *, eval_fn: Optional[Callable[[torch.nn.Module, int], Dict[str, float]]] = None) -> None:
        cfg = self.cfg
        model = self.model
        opt = self.optimizer

        t0 = time.time()
        for step in range(self.start_step, cfg.total_steps):
            lr = cosine_lr(step, base_lr=cfg.lr, warmup_steps=cfg.warmup_steps, total_steps=cfg.total_steps)
            for pg in opt.param_groups:
                pg["lr"] = lr

            opt.zero_grad(set_to_none=True)
            loss_acc = 0.0

            for micro in range(cfg.grad_accum_steps):
                x, y = self.batch_fn(self.batch_size)
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                        logits = model(x)
                        loss = self.loss_fn(logits, y) / cfg.grad_accum_steps
                    self.scaler.scale(loss).backward()
                else:
                    logits = model(x)
                    loss = self.loss_fn(logits, y) / cfg.grad_accum_steps
                    loss.backward()

                loss_acc += float(loss.detach().item())

            if cfg.grad_clip and cfg.grad_clip > 0:
                if self.use_amp:
                    self.scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            if self.use_amp:
                self.scaler.step(opt)
                self.scaler.update()
            else:
                opt.step()

            if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.total_steps - 1):
                elapsed = time.time() - t0
                print(f"step={step:6d} loss={loss_acc:.4f} lr={lr:.2e} elapsed={elapsed:.1f}s")

            if (
                eval_fn is not None
                and cfg.eval_every
                and cfg.eval_every > 0
                and step > 0
                and step % cfg.eval_every == 0
            ):
                model.eval()
                with torch.inference_mode():
                    metrics = eval_fn(model, step)
                model.train()
                if metrics:
                    mstr = " ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
                    print(f"eval step={step:6d} {mstr}")

            if cfg.ckpt_every and cfg.ckpt_every > 0 and step > 0 and step % cfg.ckpt_every == 0:
                p = self.ckpt.save(step=step, model=model, optimizer=opt)
                print(f"saved checkpoint: {p}")
