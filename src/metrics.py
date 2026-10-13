"""Evaluation metrics for continuous Centiloid regression."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


POSITIVITY_THRESHOLDS = (20.0, 24.0, 30.0)


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two binary raters."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n = len(a)
    if n == 0:
        return float("nan")
    po = float(np.mean(a == b))
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    if pe == 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def positivity_metrics(y_true: np.ndarray, y_pred: np.ndarray, thr: float) -> dict:
    """Binary amyloid-positivity concordance at a CL threshold, incl. Cohen's kappa."""
    tpos = np.asarray(y_true) >= thr
    ppos = np.asarray(y_pred) >= thr
    tp = int(np.sum(tpos & ppos))
    tn = int(np.sum(~tpos & ~ppos))
    fp = int(np.sum(~tpos & ppos))
    fn = int(np.sum(tpos & ~ppos))
    n = len(tpos)
    return {
        "accuracy": (tp + tn) / n if n else float("nan"),
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "kappa": cohen_kappa(tpos, ppos),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE, linear regression of predicted on true, Bland-Altman,
    |error| <= 5 CL fraction, positivity concordance at 20 CL, gray-zone accuracy."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true

    lr = sps.linregress(y_true, y_pred)
    bias = float(err.mean())
    sd = float(err.std(ddof=1)) if len(err) > 1 else 0.0

    pos20 = positivity_metrics(y_true, y_pred, 20.0)
    gray = (y_true >= 0) & (y_true <= 40)
    by_thr = {f"pos{thr:g}": positivity_metrics(y_true, y_pred, thr)
              for thr in POSITIVITY_THRESHOLDS}

    return {
        "positivity_by_threshold": by_thr,
        "n": int(len(y_true)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "slope": float(lr.slope),
        "intercept": float(lr.intercept),
        "r2": float(lr.rvalue ** 2),
        "pearson_r": float(lr.rvalue),
        "ba_bias": bias,
        "ba_loa_low": bias - 1.96 * sd,
        "ba_loa_high": bias + 1.96 * sd,
        "frac_within_5cl": float(np.mean(np.abs(err) <= 5.0)),
        "pos20_accuracy": pos20["accuracy"],
        "pos20_sensitivity": pos20["sensitivity"],
        "pos20_specificity": pos20["specificity"],
        "pos20_kappa": pos20["kappa"],
        "grayzone_n": int(gray.sum()),
        "grayzone_pos20_accuracy": (
            float(np.mean((y_true[gray] >= 20) == (y_pred[gray] >= 20))) if gray.any() else float("nan")
        ),
    }


def stratified_mae(y_true: np.ndarray, y_pred: np.ndarray, groups) -> dict:
    """MAE per group value (e.g. per tracer or per site)."""
    groups = np.asarray(groups)
    err = np.abs(np.asarray(y_pred) - np.asarray(y_true))
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        out[str(g)] = {"n": int(m.sum()), "mae": float(err[m].mean())}
    return out
