"""Stratified robustness: is performance uniform across patient groups?

Splits the test set by (a) baseline diagnosis (CN / MCI / AD, from the ADNI
group table joined via the manifest) and (b) true-CL interval
(<0, 0-40, 40-100, >100), and reports n, MAE, bias (mean of pred - true), and
RMSE per stratum for every evaluated run. Uniform errors across strata support
generalizability; a clearly worse stratum is a limitation to report.

Reads predictions.csv (no re-inference). Writes results/stratified_robustness.md (+ .json).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PROJECT_ROOT, load_config  # noqa: E402

TARGETS = ["CL_WC", "CL_WCB", "CL_pons"]
CL_BINS = [(-np.inf, 0.0, "< 0"), (0.0, 40.0, "0-40"),
           (40.0, 100.0, "40-100"), (100.0, np.inf, "> 100")]
DIAG_ORDER = ["CN", "MCI", "AD", "Other", "Unknown"]


def stratum_stats(err: np.ndarray) -> dict:
    return {"n": int(len(err)),
            "mae": float(np.abs(err).mean()) if len(err) else float("nan"),
            "bias": float(err.mean()) if len(err) else float("nan"),
            "rmse": float(np.sqrt((err ** 2).mean())) if len(err) else float("nan")}


def fmt(s: dict) -> str:
    return f"| {s['n']} | {s['mae']:.2f} | {s['bias']:+.2f} | {s['rmse']:.2f} |"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_glob", default=str(PROJECT_ROOT / "runs" / "*"))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    cfg = load_config()
    manifest = pd.read_csv(cfg.resolve_path(cfg.paths.manifest_csv))
    diag = manifest.set_index("scan_id")["diagnosis"]

    results = {}
    md = ["# Stratified robustness (test split)", "",
          "Strata: baseline diagnosis (CN/MCI/AD) and true-CL interval. "
          "bias = mean(pred - true).", ""]
    for run_dir in sorted(glob.glob(args.runs_glob)):
        pred_csv = Path(run_dir) / "eval" / "predictions.csv"
        if not pred_csv.exists():
            continue
        run = Path(run_dir).name
        df = pd.read_csv(pred_csv)
        df = df[df["split"] == args.split].copy()
        df["diagnosis"] = df["scan_id"].map(diag).fillna("Unknown")
        results[run] = {}
        md.append(f"## {run}\n")
        for t in TARGETS:
            short = t.replace("CL_", "")
            if f"y_true_{short}" not in df.columns:
                continue  # single-task run without this target
            err = (df[f"y_pred_{short}"] - df[f"y_true_{short}"]).to_numpy()
            yt = df[f"y_true_{short}"].to_numpy()
            res_t = {"by_diagnosis": {}, "by_cl_bin": {}}

            md.append(f"### {t}\n")
            md.append("| diagnosis | n | MAE | bias | RMSE |")
            md.append("|---|---|---|---|---|")
            for g in DIAG_ORDER:
                m = df["diagnosis"] == g
                if not m.any():
                    continue
                s = stratum_stats(err[m.to_numpy()])
                res_t["by_diagnosis"][g] = s
                md.append(f"| {g} " + fmt(s))
            md.append("")
            md.append(f"| true {t} interval | n | MAE | bias | RMSE |")
            md.append("|---|---|---|---|---|")
            for lo, hi, label in CL_BINS:
                m = (yt > lo) & (yt <= hi)
                s = stratum_stats(err[m])
                res_t["by_cl_bin"][label] = s
                md.append(f"| {label} " + fmt(s))
            md.append("")
            results[run][t] = res_t

    out = PROJECT_ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "stratified_robustness.json").write_text(json.dumps(results, indent=2))
    (out / "stratified_robustness.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
