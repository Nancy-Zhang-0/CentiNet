"""Clinical decision metrics from existing predictions.csv (no re-inference).

For every evaluated run: amyloid-positivity concordance with the reference
pipeline at thresholds 20 / 24 / 30 CL — accuracy, sensitivity, specificity,
and Cohen's kappa — to check that conclusions do not depend on the threshold
choice. Writes results/clinical_thresholds.md (+ .json).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PROJECT_ROOT  # noqa: E402
from src.metrics import POSITIVITY_THRESHOLDS, positivity_metrics  # noqa: E402

TARGETS = ["CL_WC", "CL_WCB", "CL_pons"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_glob", default=str(PROJECT_ROOT / "runs" / "*"))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    results = {}
    md = ["# Clinical decision metrics: positivity concordance by threshold", "",
          f"Split: {args.split}. Reference = pipeline Centiloid; "
          "positive if CL >= threshold for both reference and prediction.", ""]
    for run_dir in sorted(glob.glob(args.runs_glob)):
        pred_csv = Path(run_dir) / "eval" / "predictions.csv"
        if not pred_csv.exists():
            continue
        run = Path(run_dir).name
        df = pd.read_csv(pred_csv)
        df = df[df["split"] == args.split]
        results[run] = {}
        md.append(f"## {run} (n = {len(df)})\n")
        md.append("| target | threshold (CL) | accuracy | sensitivity | specificity | kappa |")
        md.append("|---|---|---|---|---|---|")
        for t in TARGETS:
            short = t.replace("CL_", "")
            if f"y_true_{short}" not in df.columns:
                continue  # single-task run without this target
            yt, yp = df[f"y_true_{short}"], df[f"y_pred_{short}"]
            for thr in POSITIVITY_THRESHOLDS:
                m = positivity_metrics(yt, yp, thr)
                results[run].setdefault(t, {})[f"thr_{thr:g}"] = m
                md.append(f"| {t} | {thr:g} | {m['accuracy']:.1%} | {m['sensitivity']:.1%} | "
                          f"{m['specificity']:.1%} | {m['kappa']:.3f} |")
        md.append("")

    out = PROJECT_ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "clinical_thresholds.json").write_text(json.dumps(results, indent=2))
    (out / "clinical_thresholds.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
