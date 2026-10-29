"""Shared helpers for the training scripts: seeding, step timing, GPU memory,
learning-rate schedule, checkpoints, sleep prevention and wandb."""
import ctypes
import math
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

GIB = 2**30


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class StepTimer:
    """Times GPU compute steps and stops training if they become far slower than expected.

    On Windows, exceeding VRAM can make the driver spill into system memory
    instead of raising out-of-memory; steps then run tens of times slower
    without any error. After `warmup` steps, `patience` consecutive steps over
    `max_seconds` raise RuntimeError.
    """

    def __init__(self, max_seconds, warmup=3, patience=2):
        self.max_seconds = max_seconds
        self.warmup = warmup
        self.patience = patience
        self.steps = 0
        self.slow = 0
        self._start = None

    def start(self):
        torch.cuda.synchronize()
        self._start = time.perf_counter()

    def stop(self):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start
        self.steps += 1
        if self.steps > self.warmup and elapsed > self.max_seconds:
            self.slow += 1
            if self.slow >= self.patience:
                raise RuntimeError(
                    f"{self.slow} consecutive steps took over {self.max_seconds:.1f}s (last {elapsed:.1f}s). "
                    "GPU memory has probably spilled into system RAM; reduce batch or patch size."
                )
        else:
            self.slow = 0
        return elapsed


def gpu_memory():
    free, total = torch.cuda.mem_get_info()
    return {
        "gpu/peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "gpu/peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
        "gpu/device_free_gib": free / GIB,
    }


def warmup_cosine(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
    """LambdaLR: linear warmup from 0 to each group's base rate, then cosine decay to min_lr_ratio x base.

    Stepped once per optimizer step. Warmup avoids large early updates while
    AdamW's moment estimates are still noisy; cosine decay lowers the rate
    smoothly so the final epochs refine rather than oscillate.
    """
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def save_checkpoint(path, **state):
    """Write atomically (temporary file, then rename) so an interruption never leaves a corrupt checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)


@contextmanager
def keep_awake():
    """Stop Windows from sleeping on idle while training runs (no effect elsewhere).

    Uses SetThreadExecutionState, which needs no admin rights. It does not
    override closing the laptop lid if the power plan sleeps on lid close.
    """
    if sys.platform != "win32":
        yield
        return
    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    try:
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous)


def init_wandb(cfg, job_type, enabled, name=None, extra_config=None, group=None, run_id=None):
    """Start a wandb run; with run_id, resume that run (used when training resumes from a checkpoint)."""
    if not enabled:
        return None
    import wandb

    return wandb.init(
        project=cfg.wandb_project,
        job_type=job_type,
        name=name,
        group=group,
        id=run_id,
        resume="allow" if run_id else None,
        config={**cfg.to_dict(), **(extra_config or {})},
    )


def log(run, metrics, step):
    """Log to wandb; a tracking failure (e.g. lost connection) is reported but never stops training."""
    if run is None:
        return
    try:
        run.log(metrics, step=step)
    except Exception as exc:  # checkpoints and stdout logs are the source of truth, tracking is best effort
        print(f"warning: wandb logging failed at step {step}: {type(exc).__name__}: {exc}", flush=True)
