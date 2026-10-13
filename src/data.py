"""PETDataset and training augmentations.

Augmentations use torch RNG only, so DataLoader worker seeding keeps them
deterministic given the global seed. The multiplicative intensity scaling in
[0.5, 2.0] is intentional: it forces the network to be invariant to global
intensity scale, since native-space PET has no reference-region normalization.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _clean_mask(df: pd.DataFrame) -> pd.Series:
    """Rows passing FOV QC and (if present) the outlier exclusion list."""
    mask = df["qc_fov_flag"] == 0
    if "outlier_flag" in df.columns:
        mask &= df["outlier_flag"] == 0
    return mask


def compute_target_stats(manifest: pd.DataFrame, targets: list[str]) -> dict:
    """Per-channel mean/std from the *train* split only (leakage guard)."""
    train = manifest[(manifest["split"] == "train") & _clean_mask(manifest)]
    stats = {}
    for t in targets:
        stats[t] = {"mean": float(train[t].mean()), "std": float(train[t].std(ddof=0))}
    return stats


def save_target_stats(stats: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_target_stats(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def standardize(y: np.ndarray, stats: dict, targets: list[str]) -> np.ndarray:
    means = np.array([stats[t]["mean"] for t in targets], dtype=np.float32)
    stds = np.array([stats[t]["std"] for t in targets], dtype=np.float32)
    return (y - means) / stds


def unstandardize(y: np.ndarray, stats: dict, targets: list[str]) -> np.ndarray:
    means = np.array([stats[t]["mean"] for t in targets], dtype=np.float32)
    stds = np.array([stats[t]["std"] for t in targets], dtype=np.float32)
    return y * stds + means


def _rotation_matrix(ax: float, ay: float, az: float) -> torch.Tensor:
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float32)
    ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32)
    rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float32)
    return rz @ ry @ rx


class TrainAugment:
    """Random affine + LR flip + multiplicative intensity scale + Gaussian noise."""

    def __init__(self, rotate_deg: float = 15.0, translate_vox: float = 8.0,
                 scale_range=(0.9, 1.1), flip_prob: float = 0.5,
                 intensity_scale=(0.5, 2.0), noise_std: float = 0.02):
        self.rotate_rad = math.radians(rotate_deg)
        self.translate_vox = translate_vox
        self.scale_range = tuple(scale_range)
        self.flip_prob = flip_prob
        self.intensity_scale = tuple(intensity_scale)
        self.noise_std = noise_std

    def __call__(self, vol: torch.Tensor) -> torch.Tensor:
        # vol: [1, D, H, W] float32
        size = vol.shape[-1]
        angles = (torch.rand(3) * 2 - 1) * self.rotate_rad
        rot = _rotation_matrix(*angles.tolist())
        scale = self.scale_range[0] + torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0])
        # affine_grid maps output coords -> input sampling coords (normalized [-1, 1])
        mat = rot * scale
        trans = (torch.rand(3) * 2 - 1) * self.translate_vox * (2.0 / size)
        theta = torch.cat([mat, trans.view(3, 1)], dim=1).unsqueeze(0)  # [1, 3, 4]
        grid = F.affine_grid(theta, [1, 1, size, size, size], align_corners=False)
        vol = F.grid_sample(vol.unsqueeze(0), grid, mode="bilinear",
                            padding_mode="zeros", align_corners=False).squeeze(0)
        if torch.rand(1).item() < self.flip_prob:
            vol = torch.flip(vol, dims=[1])  # axis 1 of [C, X, Y, Z] = left-right in RAS
        s = self.intensity_scale[0] + torch.rand(1).item() * (self.intensity_scale[1] - self.intensity_scale[0])
        vol = vol * s
        if self.noise_std > 0:
            vol = vol + torch.randn_like(vol) * self.noise_std
        return vol


class PETDataset(Dataset):
    """Native-space PET volumes -> 3 standardized Centiloid targets.

    Returns (volume [1, S, S, S] float32, targets [3] float32, scan_id str).
    QC-flagged scans (qc_fov_flag == 1) are excluded unless include_flagged=True.
    """

    def __init__(self, manifest: pd.DataFrame | str | Path, split: str, augment: bool,
                 cache_dir: str | Path, target_stats: dict | str | Path,
                 targets: list[str] = ("CL_pons", "CL_WC", "CL_WCB"),
                 augment_cfg: dict | None = None, include_flagged: bool = False):
        if not isinstance(manifest, pd.DataFrame):
            manifest = pd.read_csv(manifest)
        df = manifest[manifest["split"] == split].copy()
        if not include_flagged:
            df = df[_clean_mask(df)]
        self.df = df.reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.targets = list(targets)
        if not isinstance(target_stats, dict):
            target_stats = load_target_stats(target_stats)
        self.stats = target_stats
        self.transform = TrainAugment(**(augment_cfg or {})) if augment else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        vol = np.load(self.cache_dir / f"{row['scan_id']}.npy").astype(np.float32)
        vol = torch.from_numpy(vol).unsqueeze(0)  # [1, S, S, S]
        if self.transform is not None:
            vol = self.transform(vol)
        y = row[self.targets].to_numpy(dtype=np.float32)
        y = standardize(y, self.stats, self.targets)
        return vol, torch.from_numpy(y), row["scan_id"]
