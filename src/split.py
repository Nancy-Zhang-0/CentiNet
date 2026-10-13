"""Subject-level stratified train/val/test split.

The split is computed once (scripts/01_make_manifest.py) and read from
manifest.csv everywhere else. All scans of a subject go to the same partition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_subject_split(
    scans: pd.DataFrame,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.10, 0.20),
    cl_wc_bins: list[float] | None = None,
    subject_col: str = "subject_id",
    date_col: str = "scan_date",
    tracer_col: str = "tracer",
    cl_wc_col: str = "CL_WC",
) -> dict[str, str]:
    """Assign each subject in `scans` to train/val/test.

    `scans` must contain only QC-eligible scans (one row per scan).
    Stratification: baseline (earliest scan) CL_WC binned, crossed with the
    subject's tracer set. Deterministic given `seed`.

    Returns {subject_id: split}.
    """
    if cl_wc_bins is None:
        cl_wc_bins = [-np.inf, 0, 20, 40, 100, np.inf]
    assert abs(sum(ratios) - 1.0) < 1e-9, "ratios must sum to 1"

    df = scans.copy()
    df = df.sort_values([subject_col, date_col])
    baseline = df.groupby(subject_col, sort=True).first()
    cl_bin = pd.cut(baseline[cl_wc_col], bins=cl_wc_bins, labels=False, include_lowest=True)
    tracer_set = df.groupby(subject_col, sort=True)[tracer_col].agg(
        lambda s: "+".join(sorted(set(s)))
    )
    strata = cl_bin.astype(str) + "|" + tracer_set.astype(str)

    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    for stratum in sorted(strata.unique()):
        subjects = sorted(strata.index[strata == stratum].tolist())
        subjects = list(np.array(subjects)[rng.permutation(len(subjects))])
        n = len(subjects)
        n_val = int(round(n * ratios[1]))
        n_test = int(round(n * ratios[2]))
        n_train = n - n_val - n_test
        if n_train < 0:  # tiny stratum edge case
            n_train, n_val, n_test = n, 0, 0
        for s in subjects[:n_train]:
            assignment[s] = "train"
        for s in subjects[n_train:n_train + n_val]:
            assignment[s] = "val"
        for s in subjects[n_train + n_val:]:
            assignment[s] = "test"
    return assignment


def assert_split_integrity(manifest: pd.DataFrame, subject_col: str = "subject_id",
                           split_col: str = "split") -> None:
    """Raise if any subject appears in more than one split."""
    valid = manifest[manifest[split_col].isin(["train", "val", "test"])]
    n_splits = valid.groupby(subject_col)[split_col].nunique()
    offenders = n_splits[n_splits > 1]
    if len(offenders) > 0:
        raise AssertionError(
            f"Subject-level split violated: {len(offenders)} subjects in multiple splits: "
            f"{offenders.index.tolist()[:10]}"
        )
