"""Step 6: evaluation of a finished run.

Loads best.pt, predicts on val and test, writes predictions.csv,
metrics.json / metrics.md (per target), scatter + Bland-Altman plots, and a
stratified MAE table by tracer and site (shortcut-learning check).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Config, load_config, _SECTIONS  # noqa: E402
from src.data import PETDataset, load_target_stats, unstandardize  # noqa: E402
from src.metrics import regression_metrics, stratified_mae  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import seed_everything  # noqa: E402


def load_run_config(run_dir: Path) -> Config:
    return load_config(run_dir / "config.yaml")


@torch.no_grad()
def predict_split(model, manifest, split, cfg, stats, device) -> pd.DataFrame:
    targets = cfg.columns.targets
    ds = PETDataset(manifest, split, augment=False,
                    cache_dir=cfg.resolve_path(cfg.paths.cache_dir),
                    target_stats=stats, targets=targets)
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                    num_workers=cfg.train.num_workers)
    preds, ids = [], []
    for vol, _, scan_id in dl:
        with torch.autocast("cuda", enabled=cfg.train.amp and device == "cuda"):
            out = model(vol.to(device))
        preds.append(out.float().cpu().numpy())
        ids.extend(scan_id)
    preds = unstandardize(np.concatenate(preds), stats, targets)
    df = ds.df.set_index("scan_id").loc[ids].reset_index()
    out = df[["scan_id", "subject_id", "split", "tracer", "site"]].copy()
    for i, t in enumerate(targets):
        short = t.replace("CL_", "")
        out[f"y_true_{short}"] = df[t].to_numpy()
        out[f"y_pred_{short}"] = preds[:, i]
    return out


def plot_target(df: pd.DataFrame, target_short: str, split: str, out_dir: Path) -> None:
    yt = df[f"y_true_{target_short}"].to_numpy()
    yp = df[f"y_pred_{target_short}"].to_numpy()
    m = regression_metrics(yt, yp)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yt, yp, s=8, alpha=0.5, edgecolors="none")
    lims = [min(yt.min(), yp.min()) - 5, max(yt.max(), yp.max()) + 5]
    ax.plot(lims, lims, "k--", lw=1, label="identity")
    xs = np.array(lims)
    ax.plot(xs, m["slope"] * xs + m["intercept"], "r-", lw=1,
            label=f"fit: y={m['slope']:.2f}x+{m['intercept']:.1f}")
    ax.set_xlabel(f"True CL_{target_short}")
    ax.set_ylabel(f"Predicted CL_{target_short}")
    ax.set_title(f"{split}  MAE={m['mae']:.2f}  R2={m['r2']:.3f}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{split}_CL_{target_short}_scatter.png", dpi=130)
    plt.close(fig)

    diff = yp - yt
    mean = (yp + yt) / 2
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(mean, diff, s=8, alpha=0.5, edgecolors="none")
    for v, style, lbl in [(m["ba_bias"], "r-", f"bias {m['ba_bias']:.2f}"),
                          (m["ba_loa_low"], "r--", f"LoA {m['ba_loa_low']:.1f}"),
                          (m["ba_loa_high"], "r--", f"LoA {m['ba_loa_high']:.1f}")]:
        ax.axhline(v, color=style[0], ls=style[1:], lw=1, label=lbl)
    ax.set_xlabel(f"Mean of true and predicted CL_{target_short}")
    ax.set_ylabel("Predicted - true (CL)")
    ax.set_title(f"{split} Bland-Altman CL_{target_short}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{split}_CL_{target_short}_bland_altman.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_run_config(run_dir)
    targets = cfg.columns.targets
    seed_everything(cfg.train.seed)

    manifest = pd.read_csv(cfg.resolve_path(cfg.paths.manifest_csv))
    stats = load_target_stats(cfg.resolve_path(cfg.paths.target_stats))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(run_dir / args.checkpoint, map_location=device, weights_only=False)
    model = build_model(cfg.train.model, dropout=cfg.train.dropout,
                        n_targets=len(targets), **cfg.train.model_kwargs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.checkpoint} (epoch {ckpt['epoch']})")

    eval_dir = run_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    frames = [predict_split(model, manifest, s, cfg, stats, device) for s in ["val", "test"]]
    pred = pd.concat(frames, ignore_index=True)
    pred.to_csv(eval_dir / "predictions.csv", index=False)

    metrics: dict = {"checkpoint_epoch": int(ckpt["epoch"])}
    md = [f"# Evaluation: {run_dir.name}", ""]
    for split in ["val", "test"]:
        sub = pred[pred["split"] == split]
        metrics[split] = {}
        md.append(f"## {split} (n = {len(sub)} scans, "
                  f"{sub['subject_id'].nunique()} subjects)\n")
        md.append("| target | MAE | RMSE | slope | intercept | R2 | r | bias | LoA | "
                  "within ±5CL | pos@20 acc | sens | spec | gray-zone acc (n) |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for t in targets:
            short = t.replace("CL_", "")
            m = regression_metrics(sub[f"y_true_{short}"], sub[f"y_pred_{short}"])
            metrics[split][t] = m
            md.append(
                f"| {t} | {m['mae']:.2f} | {m['rmse']:.2f} | {m['slope']:.3f} | "
                f"{m['intercept']:.2f} | {m['r2']:.3f} | {m['pearson_r']:.3f} | "
                f"{m['ba_bias']:.2f} | [{m['ba_loa_low']:.1f}, {m['ba_loa_high']:.1f}] | "
                f"{m['frac_within_5cl']:.1%} | {m['pos20_accuracy']:.1%} | "
                f"{m['pos20_sensitivity']:.1%} | {m['pos20_specificity']:.1%} | "
                f"{m['grayzone_pos20_accuracy']:.1%} ({m['grayzone_n']}) |")
            plot_target(sub, short, split, eval_dir)
        md.append("")

        # clinical decision: positivity concordance across thresholds + kappa
        md.append(f"### {split} positivity concordance by threshold\n")
        md.append("| target | threshold (CL) | accuracy | sensitivity | specificity | kappa |")
        md.append("|---|---|---|---|---|---|")
        for t in targets:
            for key, pm in metrics[split][t]["positivity_by_threshold"].items():
                thr = key.replace("pos", "")
                md.append(f"| {t} | {thr} | {pm['accuracy']:.1%} | {pm['sensitivity']:.1%} | "
                          f"{pm['specificity']:.1%} | {pm['kappa']:.3f} |")
        md.append("")

        # stratified MAE on the primary target (shortcut-learning check)
        primary_short = cfg.columns.primary_target.replace("CL_", "")
        for by in ["tracer", "site"]:
            strat = stratified_mae(sub[f"y_true_{primary_short}"],
                                   sub[f"y_pred_{primary_short}"], sub[by])
            metrics[split][f"mae_{cfg.columns.primary_target}_by_{by}"] = strat
            md.append(f"### {split} MAE of {cfg.columns.primary_target} by {by}\n")
            md.append("| group | n | MAE |")
            md.append("|---|---|---|")
            order = sorted(strat.items(), key=lambda kv: -kv[1]["n"])
            for g, v in order:
                md.append(f"| {g} | {v['n']} | {v['mae']:.2f} |")
            md.append("")

    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (eval_dir / "metrics.md").write_text("\n".join(md))
    tm = metrics["test"][cfg.columns.primary_target]
    print(f"test {cfg.columns.primary_target}: MAE {tm['mae']:.2f}, R2 {tm['r2']:.3f}, "
          f"pos@20 acc {tm['pos20_accuracy']:.1%}")
    print(f"outputs -> {eval_dir}")


if __name__ == "__main__":
    main()
