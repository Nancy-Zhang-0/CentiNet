"""Longitudinal validity: predicted vs true annual Centiloid rate of change.

For each test-split subject with >= 2 clean scans and >= `min_span` years of
follow-up, the annual rate (CL/year) is the OLS slope of CL against time —
computed identically for the reference and the predicted values. Agreement
between the two rates is summarized with Pearson r and ICC(2,1) (two-way
random effects, absolute agreement, single measures), plus MAE/bias of rates.

This is a stronger test than cross-sectional MAE: rates difference out any
constant per-subject offset and probe whether the model tracks within-subject
amyloid accumulation.

Writes results/longitudinal_rates.md (+ .json) and a scatter PNG per run/target
under <run>/eval/longitudinal/.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.metrics import positivity_metrics  # noqa: E402

TARGETS = ["CL_WC", "CL_WCB", "CL_pons"]
ACCUMULATOR_THRESHOLDS = (1.5, 2.0, 3.0)  # CL/year; 2.0 primary


def icc_2_1(x: np.ndarray, y: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measures."""
    data = np.stack([x, y], axis=1)  # n subjects x k=2 raters
    n, k = data.shape
    grand = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)
    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_cols = n * np.sum((col_means - grand) ** 2)
    ss_total = np.sum((data - grand) ** 2)
    ss_err = ss_total - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denom) if denom != 0 else float("nan")


def annual_rate(dates_yr: np.ndarray, values: np.ndarray) -> float:
    """OLS slope in CL/year (equals delta/delta-t for two scans)."""
    t = dates_yr - dates_yr.mean()
    return float(np.sum(t * (values - values.mean())) / np.sum(t ** 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_glob", default=str(PROJECT_ROOT / "runs" / "*"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--min_span", type=float, default=0.9,
                    help="minimum follow-up span in years")
    args = ap.parse_args()

    cfg = load_config()
    manifest = pd.read_csv(cfg.resolve_path(cfg.paths.manifest_csv))
    dates = pd.to_datetime(manifest.set_index("scan_id")["scan_date"])

    results = {}
    md = ["# Longitudinal validity: predicted vs true annual CL rate", "",
          f"Split: {args.split}; subjects with >= 2 scans and >= {args.min_span} y span. "
          "Rate = OLS slope of CL vs time (CL/year). ICC = ICC(2,1), absolute agreement.", ""]
    for run_dir in sorted(glob.glob(args.runs_glob)):
        pred_csv = Path(run_dir) / "eval" / "predictions.csv"
        if not pred_csv.exists():
            continue
        run = Path(run_dir).name
        df = pd.read_csv(pred_csv)
        df = df[df["split"] == args.split].copy()
        df["date"] = df["scan_id"].map(dates)
        df["t_yr"] = (df["date"] - df["date"].min()).dt.days / 365.25

        out_fig = Path(run_dir) / "eval" / "longitudinal"
        out_fig.mkdir(parents=True, exist_ok=True)
        results[run] = {}
        md.append(f"## {run}\n")
        md.append("| target | n subj | median span (y) | r | ICC(2,1) | slope | intercept | "
                  "MAE rate | bias rate | mean true rate |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        accum_rows = []
        for t in TARGETS:
            short = t.replace("CL_", "")
            if f"y_true_{short}" not in df.columns:
                continue  # single-task run without this target
            rates_true, rates_pred, spans = [], [], []
            for _, g in df.groupby("subject_id"):
                if len(g) < 2:
                    continue
                span = g["t_yr"].max() - g["t_yr"].min()
                if span < args.min_span:
                    continue
                tt = g["t_yr"].to_numpy()
                rates_true.append(annual_rate(tt, g[f"y_true_{short}"].to_numpy()))
                rates_pred.append(annual_rate(tt, g[f"y_pred_{short}"].to_numpy()))
                spans.append(span)
            rt, rp = np.array(rates_true), np.array(rates_pred)
            if len(rt) < 3:
                md.append(f"| {t} | {len(rt)} | - | - | - | - | - | - |")
                continue
            r = float(sps.pearsonr(rt, rp).statistic)
            icc = icc_2_1(rt, rp)
            # regression of predicted rate on true rate: slope < 1 means fast
            # accumulators are underestimated (attenuation)
            lr = sps.linregress(rt, rp)
            res = {"n_subjects": len(rt), "median_span_years": float(np.median(spans)),
                   "pearson_r": round(r, 3), "icc_2_1": round(icc, 3),
                   "slope": round(float(lr.slope), 3),
                   "intercept": round(float(lr.intercept), 3),
                   "mae_rate": round(float(np.abs(rp - rt).mean()), 3),
                   "bias_rate": round(float((rp - rt).mean()), 3),
                   "mean_true_rate": round(float(rt.mean()), 3),
                   "sd_true_rate": round(float(rt.std(ddof=1)), 3)}
            # accumulator classification: true rate >= thr vs predicted rate >= thr
            res["accumulator"] = {}
            for thr in ACCUMULATOR_THRESHOLDS:
                m = positivity_metrics(rt, rp, thr)
                m["n_true_accumulators"] = int((rt >= thr).sum())
                res["accumulator"][f"thr_{thr:g}"] = m
                accum_rows.append((t, thr, m))
            results[run][t] = res
            md.append(f"| {t} | {len(rt)} | {np.median(spans):.1f} | {r:.3f} | "
                      f"{icc:.3f} | {lr.slope:.3f} | {lr.intercept:+.2f} | "
                      f"{res['mae_rate']:.2f} | {res['bias_rate']:+.2f} | "
                      f"{res['mean_true_rate']:+.2f} ± {res['sd_true_rate']:.2f} |")

            fig, ax = plt.subplots(figsize=(4.6, 4.6))
            ax.scatter(rt, rp, s=14, alpha=0.6, edgecolors="none")
            lims = [min(rt.min(), rp.min()) - 1, max(rt.max(), rp.max()) + 1]
            ax.plot(lims, lims, "k--", lw=1)
            ax.set_xlabel(f"True {t} rate (CL/year)")
            ax.set_ylabel(f"Predicted {t} rate (CL/year)")
            ax.set_title(f"n={len(rt)}  r={r:.2f}  ICC={icc:.2f}", fontsize=10)
            fig.tight_layout()
            fig.savefig(out_fig / f"rate_{t}.png", dpi=130)
            plt.close(fig)

        if accum_rows:
            md.append("\n### Accumulator classification (true rate >= thr vs predicted)\n")
            md.append("| target | thr (CL/y) | n true accum. | accuracy | "
                      "sensitivity | specificity | kappa |")
            md.append("|---|---|---|---|---|---|---|")
            for t, thr, m in accum_rows:
                md.append(f"| {t} | {thr:g} | {m['n_true_accumulators']} | "
                          f"{m['accuracy']:.1%} | {m['sensitivity']:.1%} | "
                          f"{m['specificity']:.1%} | {m['kappa']:.3f} |")
        md.append("")

    out = PROJECT_ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "longitudinal_rates.json").write_text(json.dumps(results, indent=2))
    (out / "longitudinal_rates.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
