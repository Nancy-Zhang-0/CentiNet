"""Table 1: cohort characteristics per split (clean analysis set).

Descriptive only. Subject-level rows use the baseline (earliest clean) scan;
scan-level rows use all clean scans. Age is the ADNI baseline age; sex from the
ADNI demographic table. Writes results/table1.md and results/table1.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PROJECT_ROOT, load_config  # noqa: E402

SPLITS = ["train", "val", "test"]
CL_BINS = [-np.inf, 0, 20, 40, 100, np.inf]
CL_LABELS = ["<0", "0-20", "20-40", "40-100", ">100"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="CL_WC")
    args = ap.parse_args()
    t = args.target

    cfg = load_config()
    df = pd.read_csv(cfg.resolve_path(cfg.paths.manifest_csv))
    d = df[(df.qc_fov_flag == 0) & (df.outlier_flag == 0)].copy()
    base = d.sort_values(["subject_id", "scan_date"]).groupby("subject_id").first()

    rows: list[tuple[str, dict]] = []

    def add(name: str, fn):
        rows.append((name, {s: fn(s) for s in SPLITS + ["all"]}))

    def sel_scan(s):
        return d if s == "all" else d[d.split == s]

    def sel_subj(s):
        return base if s == "all" else base[base.split == s]

    add("Subjects, n", lambda s: f"{len(sel_subj(s))}")
    add("Scans, n", lambda s: f"{len(sel_scan(s))}")
    add("Scans per subject, median (range)",
        lambda s: (lambda g: f"{g.median():.0f} ({g.min()}-{g.max()})")(
            sel_scan(s).groupby("subject_id").size()))
    add("Tracer, FBP / FBB scans",
        lambda s: f"{(sel_scan(s).tracer == 'FBP').sum()} / "
                  f"{(sel_scan(s).tracer == 'FBB').sum()}")
    add("Sex, female n (%)",
        lambda s: (lambda b: f"{(b.sex == 'F').sum()} ({(b.sex == 'F').mean():.0%})")(
            sel_subj(s)))
    add("Baseline age, mean ± SD",
        lambda s: (lambda a: f"{a.mean():.1f} ± {a.std():.1f}")(
            pd.to_numeric(sel_subj(s).age_baseline, errors="coerce").dropna()))
    for dx in ["CN", "MCI", "AD"]:
        add(f"Diagnosis {dx}, n (%)",
            lambda s, dx=dx: (lambda b: f"{(b.diagnosis == dx).sum()} "
                              f"({(b.diagnosis == dx).mean():.0%})")(sel_subj(s)))
    add(f"{t} (scans), mean ± SD",
        lambda s: (lambda c: f"{c.mean():.1f} ± {c.std():.1f}")(sel_scan(s)[t]))
    add(f"{t} (scans), median [range]",
        lambda s: (lambda c: f"{c.median():.1f} [{c.min():.1f}, {c.max():.1f}]")(
            sel_scan(s)[t]))
    add(f"Amyloid-positive scans ({t} >= 20), n (%)",
        lambda s: (lambda c: f"{(c >= 20).sum()} ({(c >= 20).mean():.0%})")(
            sel_scan(s)[t]))
    for lo, hi, lab in zip(CL_BINS[:-1], CL_BINS[1:], CL_LABELS):
        add(f"Baseline {t} {lab}, subjects n",
            lambda s, lo=lo, hi=hi: f"{((sel_subj(s)[t] > lo) & (sel_subj(s)[t] <= hi)).sum()}")

    md = ["# Table 1. Cohort characteristics (clean analysis set)", "",
          f"QC-flagged and outlier-flagged scans excluded. Baseline = earliest clean scan; "
          f"age = ADNI baseline age.", "",
          "| Characteristic | Train | Val | Test | All |", "|---|---|---|---|---|"]
    csv_rows = []
    for name, vals in rows:
        md.append(f"| {name} | " + " | ".join(vals[s] for s in SPLITS + ["all"]) + " |")
        csv_rows.append({"characteristic": name, **{s: vals[s] for s in SPLITS + ['all']}})

    out = PROJECT_ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "table1.md").write_text("\n".join(md))
    pd.DataFrame(csv_rows).to_csv(out / "table1.csv", index=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
