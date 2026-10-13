"""Step 5: training.

Loss: sum of SmoothL1 (beta = huber_delta, standardized units) over 3 targets.
AdamW + cosine schedule with warmup, AMP, early stopping on val MAE of the
primary target (CL_WC) in original units. Deterministic given the seed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402
from src.data import (PETDataset, compute_target_stats, load_target_stats,  # noqa: E402
                      save_target_stats, unstandardize)
from src.models import build_model  # noqa: E402
from src.split import assert_split_integrity  # noqa: E402
from src.utils import get_git_hash, seed_everything  # noqa: E402

AUTO_LR = {"default": 1e-4, "vit3d_small": 5e-5, "swinunetr_encoder": 5e-5}


def resolve_lr(cfg) -> float:
    if cfg.train.lr is not None:
        return float(cfg.train.lr)
    return AUTO_LR.get(cfg.train.model, AUTO_LR["default"])


def cosine_warmup(optimizer, warmup_epochs: int, total_epochs: int):
    def fn(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@torch.no_grad()
def evaluate_mae(model, loader, device, stats, targets, amp: bool) -> dict:
    model.eval()
    preds, trues = [], []
    for vol, y, _ in loader:
        vol = vol.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp):
            out = model(vol)
        preds.append(out.float().cpu().numpy())
        trues.append(y.numpy())
    preds = unstandardize(np.concatenate(preds), stats, targets)
    trues = unstandardize(np.concatenate(trues), stats, targets)
    return {t: float(np.abs(preds[:, i] - trues[:, i]).mean()) for i, t in enumerate(targets)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--pretrained_path", default=None)
    args = ap.parse_args()
    overrides = list(args.overrides)
    if args.model:
        overrides.append(f"train.model={args.model}")
    if args.seed is not None:
        overrides.append(f"train.seed={args.seed}")
    if args.pretrained_path:
        overrides.append(f"train.pretrained_path={args.pretrained_path}")
    cfg = load_config(args.config, overrides)
    targets = cfg.columns.targets
    primary = cfg.columns.primary_target
    seed_everything(cfg.train.seed)

    manifest = pd.read_csv(cfg.resolve_path(cfg.paths.manifest_csv))
    assert_split_integrity(manifest)

    # target stats: computed once from train split, reused everywhere
    stats_path = cfg.resolve_path(cfg.paths.target_stats)
    fresh = compute_target_stats(manifest, targets)
    if stats_path.exists():
        stats = load_target_stats(stats_path)
        drift = max(abs(stats[t][k] - fresh[t][k]) for t in targets for k in ("mean", "std"))
        if drift > 1e-3:
            raise SystemExit(
                f"{stats_path} does not match the current manifest's train split "
                f"(max |diff| = {drift:.3f}). Delete it to recompute, or restore the "
                "manifest it was computed from.")
    else:
        stats = fresh
        save_target_stats(stats, stats_path)
    print("target stats:", json.dumps(stats))

    cache_dir = cfg.resolve_path(cfg.paths.cache_dir)
    aug_cfg = asdict(cfg.augment)
    ds_train = PETDataset(manifest, "train", augment=True, cache_dir=cache_dir,
                          target_stats=stats, targets=targets, augment_cfg=aug_cfg)
    ds_val = PETDataset(manifest, "val", augment=False, cache_dir=cache_dir,
                        target_stats=stats, targets=targets)
    print(f"train scans: {len(ds_train)}, val scans: {len(ds_val)}")

    gen = torch.Generator().manual_seed(cfg.train.seed)
    dl_train = DataLoader(ds_train, batch_size=cfg.train.batch_size, shuffle=True,
                          num_workers=cfg.train.num_workers, pin_memory=True,
                          drop_last=True, generator=gen, persistent_workers=cfg.train.num_workers > 0)
    dl_val = DataLoader(ds_val, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=cfg.train.num_workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = cfg.train.amp and device == "cuda"
    model = build_model(cfg.train.model, dropout=cfg.train.dropout,
                        n_targets=len(targets),
                        pretrained_path=cfg.train.pretrained_path,
                        **cfg.train.model_kwargs).to(device)
    lr = resolve_lr(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = cosine_warmup(optimizer, cfg.train.warmup_epochs, cfg.train.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    kw_tag = "".join(f"_{k}-{v}" for k, v in sorted(cfg.train.model_kwargs.items()))
    if cfg.train.pretrained_path:
        kw_tag += "_pretrained"
    run_name = f"{cfg.train.model}{kw_tag}_{cfg.train.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = cfg.resolve_path(cfg.paths.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump_yaml(run_dir / "config.yaml")
    (run_dir / "meta.json").write_text(json.dumps({
        "git_hash": get_git_hash(), "lr_resolved": lr, "device": device,
        "amp": amp,
        "torch": torch.__version__, "run_name": run_name}, indent=2))
    print(f"run dir: {run_dir} (lr={lr})")

    use_wandb = cfg.train.wandb
    if use_wandb:
        import wandb
        wandb.init(project=cfg.train.wandb_project, name=run_name, config=cfg.to_dict())

    log_path = run_dir / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss"] + [f"val_mae_{t}" for t in targets]
                               + ["lr", "seconds"])

    best_mae, best_epoch = float("inf"), -1
    for epoch in range(cfg.train.epochs):
        t0 = time.time()
        model.train()
        losses = []
        for vol, y, _ in dl_train:
            vol = vol.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                pred = model(vol)
                # mean over batch, sum over the 3 targets, equal weights
                loss = F.smooth_l1_loss(pred, y, beta=cfg.train.huber_delta,
                                        reduction="none").mean(dim=0).sum()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()

        val_mae = evaluate_mae(model, dl_val, device, stats, targets, amp)
        cur_lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        row = [epoch, float(np.mean(losses))] + [val_mae[t] for t in targets] + [cur_lr, dt]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])
        if use_wandb:
            wandb.log({"epoch": epoch, "train_loss": np.mean(losses), "lr": cur_lr,
                       **{f"val_mae_{t}": val_mae[t] for t in targets}})
        print(f"epoch {epoch:3d}  loss {np.mean(losses):.4f}  "
              + "  ".join(f"{t} {val_mae[t]:.2f}" for t in targets)
              + f"  lr {cur_lr:.2e}  {dt:.0f}s")

        state = {"epoch": epoch, "model": model.state_dict(), "val_mae": val_mae,
                 "config": cfg.to_dict()}
        torch.save(state, run_dir / "last.pt")
        if val_mae[primary] < best_mae:
            best_mae, best_epoch = val_mae[primary], epoch
            torch.save(state, run_dir / "best.pt")
        if epoch - best_epoch >= cfg.train.patience:
            print(f"early stop at epoch {epoch} (best {primary} MAE {best_mae:.2f} "
                  f"@ epoch {best_epoch})")
            break

    print(f"done: best val {primary} MAE = {best_mae:.2f} CL (epoch {best_epoch})")
    (run_dir / "best_summary.json").write_text(json.dumps(
        {"best_epoch": best_epoch, f"best_val_mae_{primary}": best_mae}, indent=2))
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
