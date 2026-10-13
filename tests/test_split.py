"""Subject-level split integrity tests."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.split import assert_split_integrity, make_subject_split


def _synthetic_scans(n_subjects=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_subjects):
        subj = f"S{i:04d}"
        cl = rng.uniform(-20, 150)
        for k in range(rng.integers(1, 4)):
            rows.append({"subject_id": subj, "scan_date": f"2015-0{k + 1}-01",
                         "tracer": "FBP" if i % 10 else "FBB",
                         "CL_WC": cl + rng.normal(0, 5)})
    return pd.DataFrame(rows)


def test_no_subject_in_two_splits():
    df = _synthetic_scans()
    assign = make_subject_split(df, seed=42)
    df["split"] = df["subject_id"].map(assign)
    df["qc_fov_flag"] = 0
    assert_split_integrity(df)  # must not raise
    per_subj = df.groupby("subject_id")["split"].nunique()
    assert (per_subj == 1).all()


def test_all_subjects_assigned_and_ratios():
    df = _synthetic_scans()
    assign = make_subject_split(df, seed=42, ratios=(0.7, 0.1, 0.2))
    assert set(assign) == set(df["subject_id"])
    counts = pd.Series(assign).value_counts()
    n = len(assign)
    assert abs(counts.get("train", 0) / n - 0.7) < 0.05
    assert abs(counts.get("val", 0) / n - 0.1) < 0.05
    assert abs(counts.get("test", 0) / n - 0.2) < 0.05


def test_deterministic_given_seed():
    df = _synthetic_scans()
    a = make_subject_split(df, seed=42)
    b = make_subject_split(df, seed=42)
    c = make_subject_split(df, seed=43)
    assert a == b
    assert a != c


def test_integrity_check_catches_violation():
    df = pd.DataFrame({
        "subject_id": ["A", "A", "B"], "split": ["train", "test", "val"],
        "qc_fov_flag": [0, 0, 0]})
    with pytest.raises(AssertionError):
        assert_split_integrity(df)
