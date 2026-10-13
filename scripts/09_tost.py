"""TOST equivalence testing: are model predictions equivalent to the reference
Centiloid pipeline within a pre-specified margin?

For each run/target, paired differences d = pred - true (original CL units) on
the test split. Two one-sided t-tests against margins +/-delta:
    H0: |mean(d)| >= delta   vs   H1: |mean(d)| < delta
Equivalence at alpha = 0.05  <=>  the 90% CI of mean(d) lies within (-delta, +delta).

Primary analysis is SUBJECT-level (differences averaged within subject first)
because scans cluster within subjects; scan-level results are reported as a
sensitivity analysis. Margins: 2.5, 5 (primary), 10 CL. delta_min is the
smallest margin at which equivalence would be declared (= widest |90% CI bound|).

Note: TOST addresses the *mean bias* only; individual-level agreement is
covered by the Bland-Altman limits of agreement in metrics.md.
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

TARGETS = ["CL_WC", "CL_WCB", "CL_pons"]
MARGINS = [2.5, 5.0, 10.0]
ALPHA = 0.05


def tost(d: np.ndarray, delta: float) -> dict:
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    df = n - 1
    t_lower = (mean + delta) / se          # H0: mean <= -delta
    t_upper = (mean - delta) / se          # H0: mean >= +delta
    p_lower = 1 - sps.t.cdf(t_lower, df)
    p_upper = sps.t.cdf(t_upper, df)
    p = float(max(p_lower, p_upper))
    tcrit = sps.t.ppf(1 - ALPHA, df)       # 90% CI = mean +/- t_{0.95} * se
    ci = (mean - tcrit * se, mean + tcrit * se)
    return {"n": n, "mean_bias": round(mean, 3), "se": round(se, 3),
            "ci90": [round(ci[0], 3), round(ci[1], 3)],
            "p_tost": float(f"{p:.2e}" if p < 1e-3 else round(p, 4)),
            "equivalent": bool(p < ALPHA)}


def delta_min(d: np.ndarray) -> float:
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = sps.t.ppf(1 - ALPHA, n - 1)
    lo, hi = d.mean() - tcrit * se, d.mean() + tcrit * se
    return float(max(abs(lo), abs(hi)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_glob", default=str(PROJECT_ROOT / "runs" / "*"))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    results = {}
    md = ["# TOST equivalence tests (predicted vs reference Centiloid)", "",
          f"Split: {args.split}. alpha = {ALPHA} (equivalence <=> 90% CI within +/-delta).",
          "Primary: subject-level (within-subject averaged differences); "
          "scan-level shown as sensitivity.", ""]
    for run_dir in sorted(glob.glob(args.runs_glob)):
        pred_csv = Path(run_dir) / "eval" / "predictions.csv"
        if not pred_csv.exists():
            continue
        run = Path(run_dir).name
        df = pd.read_csv(pred_csv)
        df = df[df["split"] == args.split]
        results[run] = {}
        md.append(f"## {run}\n")
        md.append("| target | level | n | bias | 90% CI | delta_min | "
                  + " | ".join(f"TOST p (±{m:g})" for m in MARGINS) + " |")
        md.append("|---|---|---|---|---|---|" + "---|" * len(MARGINS))
        for t in TARGETS:
            short = t.replace("CL_", "")
            if f"y_true_{short}" not in df.columns:
                continue  # run trained without this target (single-task ablation)
            d_scan = (df[f"y_pred_{short}"] - df[f"y_true_{short}"]).to_numpy()
            d_subj = (df.assign(d=df[f"y_pred_{short}"] - df[f"y_true_{short}"])
                      .groupby("subject_id")["d"].mean().to_numpy())
            for level, d in [("subject", d_subj), ("scan", d_scan)]:
                res = {f"margin_{m:g}": tost(d, m) for m in MARGINS}
                dmin = round(delta_min(d), 2)
                results[run].setdefault(t, {})[level] = {
                    "delta_min": dmin, **res}
                r0 = res[f"margin_{MARGINS[0]:g}"]
                cells = " | ".join(
                    f"{res[f'margin_{m:g}']['p_tost']:.2g}"
                    f"{' ✓' if res[f'margin_{m:g}']['equivalent'] else ' ✗'}"
                    for m in MARGINS)
                md.append(f"| {t} | {level} | {r0['n']} | {r0['mean_bias']:.2f} | "
                          f"[{r0['ci90'][0]:.2f}, {r0['ci90'][1]:.2f}] | {dmin} | {cells} |")
        md.append("")

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "tost.json").write_text(json.dumps(results, indent=2))
    (out_dir / "tost.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\n-> {out_dir}/tost.md, tost.json")


if __name__ == "__main__":
    main()
