"""Step 7 helper: aggregate all evaluated runs into results/summary.md."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PROJECT_ROOT  # noqa: E402

COLS = ["mae", "rmse", "slope", "intercept", "r2", "pearson_r", "ba_bias",
        "ba_loa_low", "ba_loa_high", "frac_within_5cl", "pos20_accuracy",
        "pos20_sensitivity", "pos20_specificity", "pos20_kappa",
        "grayzone_pos20_accuracy"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default=str(PROJECT_ROOT / "runs"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "summary.md"))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    rows = []
    for run_dir in sorted(Path(args.runs_dir).iterdir()):
        mfile = run_dir / "eval" / "metrics.json"
        if not mfile.exists():
            continue
        metrics = json.loads(mfile.read_text())
        with open(run_dir / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        pretrained = bool(cfg["train"].get("pretrained_path"))
        rows.append({"run": run_dir.name, "model": cfg["train"]["model"],
                     "seed": cfg["train"]["seed"], "pretrained": pretrained,
                     "metrics": metrics.get(args.split, {})})

    targets = ["CL_WC", "CL_pons", "CL_WCB"]  # primary target first
    lines = [f"# Summary ({args.split} split)", ""]
    for t in targets:
        lines.append(f"## {t}\n")
        lines.append("| run | model | seed | pretrained | " + " | ".join(COLS) + " |")
        lines.append("|" + "---|" * (len(COLS) + 4))
        for r in rows:
            m = r["metrics"].get(t)
            if not m:
                continue
            vals = " | ".join(
                f"{m[c]:.3f}" if isinstance(m.get(c), float) else str(m.get(c, "-"))
                for c in COLS)
            lines.append(f"| {r['run']} | {r['model']} | {r['seed']} | "
                         f"{'yes' if r['pretrained'] else 'no'} | {vals} |")
        lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"{len(rows)} evaluated runs -> {out}")


if __name__ == "__main__":
    main()
