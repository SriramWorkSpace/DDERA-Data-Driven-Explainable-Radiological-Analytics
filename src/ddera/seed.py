"""Deterministic seeding.

Every notebook and script calls ``set_seed(cfg.seed)`` before doing anything else.
Reproducibility is not optional here: the project's deliverable is a methodological claim,
and a claim that cannot be reproduced is not a claim.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch.

    Args:
        seed: the seed.
        deterministic: request deterministic cuDNN/MIOpen algorithms. Costs some throughput;
            worth it for a reproducibility-critical project. Disable only for a run whose
            exact numbers are not being reported.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - signature fixed by DataLoader
    """DataLoader ``worker_init_fn``: keeps augmentation reproducible across workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 42) -> torch.Generator:
    """Generator for DataLoader shuffling, so batch order is reproducible."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
