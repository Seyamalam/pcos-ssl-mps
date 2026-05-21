from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RuntimeConfig:
    device: torch.device
    device_name: str
    cpu_threads: int
    mps_available: bool
    torch_version: str


def configure_runtime(seed: int = 42, cpu_threads: int = 18) -> RuntimeConfig:
    """Configure reproducibility and Apple Silicon friendly defaults."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    available_threads = os.cpu_count() or 1
    threads = max(1, min(cpu_threads, available_threads))
    torch.set_num_threads(threads)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "mps"
    else:
        device = torch.device("cpu")
        device_name = "cpu"

    return RuntimeConfig(
        device=device,
        device_name=device_name,
        cpu_threads=torch.get_num_threads(),
        mps_available=torch.backends.mps.is_available(),
        torch_version=torch.__version__,
    )

