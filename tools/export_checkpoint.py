"""Package a training run into a self-contained release checkpoint.

A run's best.pt contains the full training config (with local absolute paths
and the per-epoch validation MAE). This tool writes a minimal, portable file
that `predict.py` can load without the training manifest or config:

    model_name, model_kwargs, targets, target_stats, preprocess, state_dict

Usage:
    python tools/export_checkpoint.py --run_dir runs/<run> \
        --out weights/resnet_transformer_hybrid_seed0.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402
from src.data import load_target_stats  # noqa: E402

RELEASE_FORMAT = 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target_stats", default=None,
                    help="path to target_stats.json; defaults to the config's "
                         "path, falling back to <release>/weights/target_stats.json")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    ckpt = torch.load(run_dir / args.checkpoint, map_location="cpu", weights_only=False)
    stats_path = Path(args.target_stats) if args.target_stats else \
        cfg.resolve_path(cfg.paths.target_stats)
    if not stats_path.exists():
        fallback = Path(__file__).resolve().parent.parent / "weights" / "target_stats.json"
        if not fallback.exists():
            raise SystemExit(f"target stats not found at {stats_path} or {fallback}; "
                             "pass --target_stats explicitly")
        stats_path = fallback
    stats = load_target_stats(stats_path)

    release = {
        "format": RELEASE_FORMAT,
        "model_name": cfg.train.model,
        "model_kwargs": dict(cfg.train.model_kwargs),
        "dropout": cfg.train.dropout,
        "targets": list(cfg.columns.targets),
        "primary_target": cfg.columns.primary_target,
        "target_stats": stats,
        "preprocess": {
            "voxel_size": cfg.preprocess.voxel_size,
            "crop_size": cfg.preprocess.crop_size,
            "fg_threshold_frac": cfg.preprocess.fg_threshold_frac,
            "norm_percentile": cfg.preprocess.norm_percentile,
        },
        "source_run": run_dir.name,
        "epoch": int(ckpt["epoch"]),
        "val_mae": ckpt.get("val_mae"),
        "state_dict": ckpt["model"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(release, out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    (out.with_suffix(".json")).write_text(json.dumps(
        {k: v for k, v in release.items() if k != "state_dict"} | {"sha256": sha},
        indent=2))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)\nsha256 {sha}")


if __name__ == "__main__":
    main()
