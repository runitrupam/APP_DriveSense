"""Deterministic seeding across the libraries used by the pipeline."""

from __future__ import annotations

import os
import random

import numpy as np

_seed: int | None = None


def set_global_seed(seed: int) -> int:
    """Seed random, numpy, torch (if importable) and PYTHONHASHSEED."""
    global _seed
    _seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(_seed))
    random.seed(_seed)
    np.random.seed(_seed)
    try:
        import torch

        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)
    except Exception:
        pass
    try:
        import cv2

        cv2.setRNGSeed(_seed)
    except Exception:
        pass
    return _seed


def current_seed() -> int | None:
    """Seed applied by :func:`set_global_seed`, if any."""
    return _seed
