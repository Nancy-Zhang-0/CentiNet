"""Metric functions against hand-computed values."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics import cohen_kappa, positivity_metrics, regression_metrics, stratified_mae


def test_cohen_kappa_hand_computed():
    # po = 0.75; pe = 0.75*0.5 + 0.25*0.5 = 0.5 -> kappa = 0.5
    a = np.array([True, True, True, False])
    b = np.array([True, True, False, False])
    assert cohen_kappa(a, b) == pytest.approx(0.5)
    assert cohen_kappa(a, a) == pytest.approx(1.0)
    # chance-level agreement -> kappa = 0
    assert cohen_kappa(np.array([True, True, False, False]),
                       np.array([True, False, True, False])) == pytest.approx(0.0)


def test_positivity_thresholds():
    y_true = np.array([10.0, 22.0, 26.0, 40.0])
    y_pred = np.array([10.0, 22.0, 26.0, 40.0])
    for thr in (20.0, 24.0, 30.0):
        m = positivity_metrics(y_true, y_pred, thr)
        assert m["accuracy"] == 1.0 and m["kappa"] == pytest.approx(1.0)
    # at thr 24: true pos = {26, 40}; pred = y_true + 3 -> 25 becomes pos
    m = positivity_metrics(y_true, y_true + 3.0, 24.0)
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == pytest.approx(0.5)  # 22+3=25 crosses 24


def test_perfect_prediction():
    y = np.array([0.0, 10.0, 30.0, 50.0])
    m = regression_metrics(y, y)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["slope"] == pytest.approx(1.0)
    assert m["intercept"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["frac_within_5cl"] == 1.0
    assert m["pos20_accuracy"] == 1.0


def test_hand_computed_errors():
    # errors: +2, -2, +4, 0 -> MAE = 2, RMSE = sqrt(24/4) = sqrt(6)
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 34.0, 40.0])
    m = regression_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(2.0)
    assert m["rmse"] == pytest.approx(np.sqrt(6.0))
    # Bland-Altman: diffs = [2, -2, 4, 0]; bias = 1; sd (ddof=1) = sqrt(20/3)
    assert m["ba_bias"] == pytest.approx(1.0)
    sd = np.sqrt(20.0 / 3.0)
    assert m["ba_loa_low"] == pytest.approx(1.0 - 1.96 * sd)
    assert m["ba_loa_high"] == pytest.approx(1.0 + 1.96 * sd)
    assert m["frac_within_5cl"] == pytest.approx(1.0)  # all |err| <= 5


def test_positivity_concordance():
    # true: [neg, neg, pos, pos]; pred: [neg, pos, neg, pos] -> acc 0.5, sens 0.5, spec 0.5
    y_true = np.array([5.0, 10.0, 30.0, 40.0])
    y_pred = np.array([5.0, 25.0, 10.0, 40.0])
    m = regression_metrics(y_true, y_pred)
    assert m["pos20_accuracy"] == pytest.approx(0.5)
    assert m["pos20_sensitivity"] == pytest.approx(0.5)
    assert m["pos20_specificity"] == pytest.approx(0.5)
    # gray zone = true CL in [0, 40] -> all 4 scans
    assert m["grayzone_n"] == 4
    assert m["grayzone_pos20_accuracy"] == pytest.approx(0.5)


def test_grayzone_restriction():
    y_true = np.array([-10.0, 10.0, 30.0, 90.0])  # gray zone: indices 1, 2
    y_pred = np.array([-10.0, 25.0, 35.0, 90.0])  # idx1 wrong call, idx2 right
    m = regression_metrics(y_true, y_pred)
    assert m["grayzone_n"] == 2
    assert m["grayzone_pos20_accuracy"] == pytest.approx(0.5)


def test_stratified_mae():
    y_true = np.array([0.0, 0.0, 10.0, 10.0])
    y_pred = np.array([1.0, 3.0, 10.0, 14.0])
    out = stratified_mae(y_true, y_pred, ["a", "a", "b", "b"])
    assert out["a"] == {"n": 2, "mae": pytest.approx(2.0)}
    assert out["b"] == {"n": 2, "mae": pytest.approx(2.0)}
