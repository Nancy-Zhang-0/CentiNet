"""Volume preprocessing: reorient, resample, crop, scale-free intensity norm.

Deliberately anatomical-information-free: no cerebellar, pontine, or atlas-based
region is used anywhere (that would leak the Centiloid reference region).
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output

from src.utils import foreground_mask


def load_ras_resampled(nii_path: str | Path, voxel_size: float = 2.0) -> tuple[np.ndarray, tuple]:
    """Load NIfTI, reorient to RAS, resample to isotropic `voxel_size` (trilinear)."""
    img = nib.load(str(nii_path))
    img = nib.as_closest_canonical(img)
    img = resample_to_output(img, voxel_sizes=(voxel_size,) * 3, order=1, cval=0.0)
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    zooms = img.header.get_zooms()[:3]
    return data, zooms


def crop_or_pad_to(vol: np.ndarray, center: np.ndarray, size: int) -> np.ndarray:
    """Extract a `size`^3 window centered on `center` (voxel coords), zero-padded."""
    out = np.zeros((size, size, size), dtype=vol.dtype)
    half = size // 2
    start = np.round(center).astype(int) - half
    src_lo = np.maximum(start, 0)
    src_hi = np.minimum(start + size, np.array(vol.shape))
    dst_lo = src_lo - start
    dst_hi = dst_lo + (src_hi - src_lo)
    if np.any(src_hi <= src_lo):
        return out
    out[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = \
        vol[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
    return out


def preprocess_volume(
    nii_path: str | Path,
    voxel_size: float = 2.0,
    crop_size: int = 96,
    fg_threshold_frac: float = 0.10,
    norm_percentile: float = 95.0,
) -> np.ndarray:
    """Full preprocessing for one scan -> float16 array [crop_size]^3.

    1. Reorient to RAS, resample to isotropic voxels.
    2. Foreground mask by intensity threshold; center-crop/pad to bounding-box center.
    3. Divide by the `norm_percentile` of foreground voxels of that same volume.
    """
    vol, _ = load_ras_resampled(nii_path, voxel_size)
    mask = foreground_mask(vol, frac=fg_threshold_frac)
    if mask.sum() < 100:
        raise ValueError(f"Foreground mask nearly empty for {nii_path}")
    idx = np.argwhere(mask)
    center = (idx.min(0) + idx.max(0)) / 2.0
    cropped = crop_or_pad_to(vol, center, crop_size)
    p = np.percentile(vol[mask], norm_percentile)
    if p <= 0:
        raise ValueError(f"Non-positive normalization percentile for {nii_path}")
    cropped = cropped / p
    return cropped.astype(np.float16)


def inferior_coverage_mm(
    nii_path: str | Path,
    voxel_size: float = 2.0,
    fg_threshold_frac: float = 0.10,
    min_slice_voxels: int = 10,
) -> float:
    """QC metric for inferior field-of-view truncation.

    Number of axial foreground slices *below* (inferior to) the slice with
    maximal foreground area, in millimetres. Low values suggest the inferior
    part of the head (cerebellum/pons region) is cut off.
    """
    vol, zooms = load_ras_resampled(nii_path, voxel_size)
    mask = foreground_mask(vol, frac=fg_threshold_frac)
    area = mask.sum(axis=(0, 1))  # per-axial-slice foreground area; z = axis 2 (RAS: +z superior)
    if area.max() == 0:
        return 0.0
    z_max = int(np.argmax(area))
    n_below = int(np.sum(area[:z_max] >= min_slice_voxels))
    return float(n_below * zooms[2])
