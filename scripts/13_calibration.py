"""Raw vs post-hoc-calibrated test metrics.

IMPORTANT: the pipeline's standard metrics are computed on RAW model outputs
(the only transform is un-standardization with fixed train-set target mean/std,
applied identically to every model before any evaluation). This script adds the
comparison row: an affine calibration y_true = a * y_pred + b fitted on the
VALIDATION predictions of each run/target, applied to the TEST predictions.

The raw-vs-calibrated delta shows (a) how much regression-to-the-mean the raw
model has (raw slope), and (b) how much a cheap recalibration contributes.

Writes results/calibration.md (+ .json).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PROJECT_ROOT  # noqa: E402
from src.metrics import regression_metrics  # noqa: E402

TARGETS = ["CL_WC", "CL_WCB", "CL_pons"]
FIELDS = ["mae", "rmse", "r2", "slope", "intercept", "ba_bias",
          "ba_loa_low", "ba_loa_high", "frac_within_5cl", "pos20_accuracy"]


def row(tag: str, m: dict) -> str:
    return (f"| {tag} | {m['mae']:.2f} | {m['rmse']:.2f} | {m['r2']:.3f} | "
            f"{m['slope']:.3f} | {m['intercept']:.2f} | {m['ba_bias']:+.2f} | "
            f"[{m['ba_loa_low']:.1f}, {m['ba_loa_high']:.1f}] | "
            f"{m['frac_within_5cl']:.1%} | {m['pos20_accuracy']:.1%} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_glob", default=str(PROJECT_ROOT / "runs" / "*"))
    args = ap.parse_args()

    results = {}
    md = ["# Raw vs post-hoc calibrated test metrics", "",
          "Raw = model output (fixed train-stats un-standardization only; no fit "
          "to val or test). Calibrated = affine y = a*pred + b fitted on VAL "
          "predictions, applied to TEST.", ""]
    for run_dir in sorted(glob.glob(args.runs_glob)):
        pred_csv = Path(run_dir) / "eval" / "predictions.csv"
        if not pred_csv.exists():
            continue
        run = Path(run_dir).name
        df = pd.read_csv(pred_csv)
        val = df[df["split"] == "val"]
        test = df[df["split"] == "test"]
        results[run] = {}
        md.append(f"## {run}\n")
        md.append("| variant | MAE | RMSE | R2 | slope | intercept | bias | "
                  "95% LoA | ±5CL | pos@20 |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for t in TARGETS:
            short = t.replace("CL_", "")
            if f"y_true_{short}" not in df.columns:
                continue
            yt_v, yp_v = val[f"y_true_{short}"], val[f"y_pred_{short}"]
            yt_t = test[f"y_true_{short}"].to_numpy()
            yp_t = test[f"y_pred_{short}"].to_numpy()
            fit = sps.linregress(yp_v, yt_v)  # y_true = a * pred + b, on val
            yp_cal = fit.slope * yp_t + fit.intercept
            m_raw = regression_metrics(yt_t, yp_t)
            m_cal = regression_metrics(yt_t, yp_cal)
            results[run][t] = {
                "cal_fit": {"a": round(float(fit.slope), 4),
                            "b": round(float(fit.intercept), 3)},
                "raw": {k: m_raw[k] for k in FIELDS},
                "calibrated": {k: m_cal[k] for k in FIELDS},
            }
            md.append(f"| **{t} raw** " + row("", m_raw)[2:])
            md.append(f"| {t} calibrated (a={fit.slope:.3f}, b={fit.intercept:+.2f}) "
                      + row("", m_cal)[2:])
        md.append("")

    out = PROJECT_ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(results, indent=2))
    (out / "calibration.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
