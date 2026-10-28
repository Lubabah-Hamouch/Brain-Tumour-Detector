"""Shared helpers for the training scripts: seeding, step timing, GPU memory, wandb."""
import random
import time

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


def init_wandb(cfg, job_type, enabled, name=None, extra_config=None):
    if not enabled:
        return None
    import wandb

    return wandb.init(
        project=cfg.wandb_project,
        job_type=job_type,
        name=name,
        config={**cfg.to_dict(), **(extra_config or {})},
    )


def log(run, metrics, step):
    if run is not None:
        run.log(metrics, step=step)
