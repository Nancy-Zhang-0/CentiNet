"""Step 2: preprocess and cache all manifest scans.

RAS reorient -> 2 mm isotropic -> foreground bbox center-crop/pad to 96^3 ->
divide by p95 of foreground voxels (same volume; nothing anatomical) ->
float16 .npy under data/cache/. Also writes orthogonal mid-slice PNG montages
for 20 random scans to data/qc_png/.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402
from src.preprocessing import preprocess_volume  # noqa: E402

_P = None


def _process_one(args: tuple[str, str]) -> tuple[str, bool, str]:
    scan_id, pet_path = args
    out = Path(_P["cache_dir"]) / f"{scan_id}.npy"
    if out.exists() and not _P["overwrite"]:
        return scan_id, True, "cached"
    try:
        vol = preprocess_volume(
            pet_path, voxel_size=_P["voxel_size"], crop_size=_P["crop_size"],
            fg_threshold_frac=_P["fg_frac"], norm_percentile=_P["norm_pct"])
        np.save(out, vol)
        return scan_id, True, "ok"
    except Exception as e:
        return scan_id, False, str(e)


def save_montage(cache_dir: Path, scan_id: str, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vol = np.load(cache_dir / f"{scan_id}.npy").astype(np.float32)
    mids = [s // 2 for s in vol.shape]
    slices = [vol[mids[0], :, :], vol[:, mids[1], :], vol[:, :, mids[2]]]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, sl, name in zip(axes, slices, ["sagittal", "coronal", "axial"]):
        ax.imshow(np.rot90(sl), cmap="hot", vmin=0, vmax=np.percentile(vol, 99.5))
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    fig.suptitle(scan_id, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"{scan_id}.png", dpi=110)
    plt.close(fig)


def main() -> None:
    global _P
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--n_workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    manifest = pd.read_csv(cfg.resolve_path(cfg.paths.manifest_csv))
    cache_dir = cfg.resolve_path(cfg.paths.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _P = {"cache_dir": str(cache_dir), "overwrite": args.overwrite,
          "voxel_size": cfg.preprocess.voxel_size, "crop_size": cfg.preprocess.crop_size,
          "fg_frac": cfg.preprocess.fg_threshold_frac,
          "norm_pct": cfg.preprocess.norm_percentile}

    jobs = list(zip(manifest["scan_id"], manifest["pet_path"]))
    n_workers = args.n_workers or cfg.preprocess.n_workers
    failures = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for scan_id, ok, msg in ex.map(_process_one, jobs, chunksize=8):
            if not ok:
                failures.append((scan_id, msg))
    print(f"cached {len(jobs) - len(failures)}/{len(jobs)} scans -> {cache_dir}")
    if failures:
        fail_path = cache_dir.parent / "preprocess_failures.csv"
        pd.DataFrame(failures, columns=["scan_id", "error"]).to_csv(fail_path, index=False)
        print(f"FAILED {len(failures)} scans -> {fail_path}")

    # QC montages for a fixed random sample of successfully cached scans
    qc_dir = cfg.resolve_path(cfg.paths.qc_png_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)
    ok_ids = [s for s in manifest["scan_id"] if (cache_dir / f"{s}.npy").exists()]
    rng = np.random.default_rng(cfg.split.seed)
    sample = rng.choice(ok_ids, size=min(cfg.preprocess.n_qc_png, len(ok_ids)), replace=False)
    for s in sample:
        save_montage(cache_dir, s, qc_dir)
    print(f"QC montages for {len(sample)} scans -> {qc_dir}")


if __name__ == "__main__":
    main()
