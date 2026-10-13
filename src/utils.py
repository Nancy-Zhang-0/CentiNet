"""Seeding, reproducibility, and small helpers."""
from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path

import numpy as np


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except ImportError:
        pass


def get_git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent.parent, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown (not a git repository)"


def foreground_mask(vol: np.ndarray, frac: float = 0.10, ref_percentile: float = 99.0) -> np.ndarray:
    """Foreground = voxels above `frac` * (ref_percentile of the volume).

    Purely intensity-based; no anatomical or atlas information is used anywhere.
    """
    ref = np.percentile(vol, ref_percentile)
    return vol > frac * ref
