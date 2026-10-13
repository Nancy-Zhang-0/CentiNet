"""Aggregate the multi-seed study: mean ± std per architecture over seeds.

Includes only default-configuration runs (3 targets, empty model_kwargs,
default lr/weight-decay) so ablation variants, single-task and HPO runs are
excluded automatically. In particular, for (resnet_transformer_hybrid, seed 0)
the canonical run is the multi-task default run — the single-task CL_WC-only
run with the same model/seed fails the 3-target check and never enters the
table. If several default runs exist for one (model, seed), the newest wins. Reports per model and target: n seeds, MAE mean ± std,
plus RMSE / R2 / pos20-kappa means. Writes results/seed_study.md (+ .csv).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PROJECT_ROOT  # noqa: E402

TARGETS = ["CL_WC", "CL_WCB", "CL_pons"]


def is_default_run(cfg: dict) -> bool:
    """Default-protocol runs only; pretrained runs are kept as their own arm."""
    t = cfg["train"]
    return (not t.get("model_kwargs")
            and len(cfg["columns"]["targets"]) == 3
            and t.get("lr") in (None, "null")
            and abs(float(t.get("weight_decay", 0.05)) - 0.05) < 1e-9)


def model_label(cfg: dict) -> str:
    name = cfg["train"]["model"]
    if cfg["train"].get("pretrained_path") not in (None, "null", ""):
        name += " (pretrained)"
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_glob", default=str(PROJECT_ROOT / "runs" / "*"))
    args = ap.parse_args()

    rows = []
    for run_dir in sorted(glob.glob(args.runs_glob)):
        run_dir = Path(run_dir)
        mfile = run_dir / "eval" / "metrics.json"
        cfile = run_dir / "config.yaml"
        if not (mfile.exists() and cfile.exists()):
            continue
        with open(cfile) as f:
            cfg = yaml.safe_load(f)
        if not is_default_run(cfg):
            continue
        metrics = json.loads(mfile.read_text())
        for t in TARGETS:
            m = metrics["test"].get(t)
            if not m:
                continue
            rows.append({"model": model_label(cfg), "seed": cfg["train"]["seed"],
                         "target": t, "run": run_dir.name, "mae": m["mae"],
                         "rmse": m["rmse"], "r2": m["r2"],
                         "kappa20": m.get("pos20_kappa", np.nan)})
    df = pd.DataFrame(rows)
    # one run per (model, seed, target): keep the newest if duplicated
    df = df.sort_values("run").drop_duplicates(["model", "seed", "target"], keep="last")
    df.to_csv(PROJECT_ROOT / "results" / "seed_study_runs.csv", index=False)

    md = ["# Multi-seed study (test split, default configuration runs)", ""]
    for t in TARGETS:
        sub = df[df.target == t]
        g = sub.groupby("model").agg(
            n_seeds=("seed", "nunique"), mae_mean=("mae", "mean"), mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"), r2_mean=("r2", "mean"),
            kappa20_mean=("kappa20", "mean")).sort_values("mae_mean")
        md.append(f"## {t}\n")
        md.append("| model | seeds | MAE mean ± std | RMSE | R2 | kappa@20 |")
        md.append("|---|---|---|---|---|---|")
        for model, r in g.iterrows():
            md.append(f"| {model} | {int(r.n_seeds)} | "
                      f"{r.mae_mean:.2f} ± {r.mae_std:.2f} | {r.rmse_mean:.2f} | "
                      f"{r.r2_mean:.3f} | {r.kappa20_mean:.3f} |")
        md.append("")
    out = PROJECT_ROOT / "results" / "seed_study.md"
    out.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
