"""Proper scoring rules for probabilistic match forecasts (spec Parts
XX-XXI, XXXVII). Adapted from the World Cup predictor project's
calibration_utils.py (docs/world_cup_transfer_audit.md classification B —
portable mechanism, but the original hardcodes exactly 3 classes and this
project needs the same math applied across an independent model
tournament harness, so it's reimplemented cleanly here rather than
imported from an unrelated external repo).

All functions take `probs`: an (n, 3) array of [P(home win), P(draw),
P(away win)], and `outcomes`: a length-n sequence of 'H'/'D'/'A'.
"""
from __future__ import annotations

import numpy as np

LABELS = ["H", "D", "A"]
IDX = {"H": 0, "D": 1, "A": 2}


def _onehot(outcomes) -> np.ndarray:
    m = np.zeros((len(outcomes), 3))
    for k, o in enumerate(outcomes):
        m[k, IDX[o]] = 1
    return m


def log_loss(probs: np.ndarray, outcomes, eps: float = 1e-15) -> float:
    p = np.clip(probs, eps, 1)
    oh = _onehot(outcomes)
    return float(-np.mean(np.sum(oh * np.log(p), axis=1)))


def brier(probs: np.ndarray, outcomes) -> float:
    return float(np.mean(np.sum((probs - _onehot(outcomes)) ** 2, axis=1)))


def rps(probs: np.ndarray, outcomes) -> float:
    """Ranked probability score for the ordered outcome space H < D < A. Lower is better."""
    oh = _onehot(outcomes)
    cp = np.cumsum(probs, axis=1)
    co = np.cumsum(oh, axis=1)
    return float(np.mean(np.sum((cp - co) ** 2, axis=1) / (probs.shape[1] - 1)))


def accuracy(probs: np.ndarray, outcomes) -> float:
    pred = np.array(LABELS)[probs.argmax(axis=1)]
    return float(np.mean(pred == np.array(outcomes)))


def ece(probs: np.ndarray, outcomes, n_bins: int = 10) -> float:
    """Expected calibration error on the predicted (max-probability) class."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = np.array([1 if pred[k] == IDX[o] else 0 for k, o in enumerate(outcomes)])
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for b in range(n_bins):
        m = (conf > bins[b]) & (conf <= bins[b + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / len(outcomes)) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def goals_mae(pred_home: np.ndarray, pred_away: np.ndarray, actual_home: np.ndarray, actual_away: np.ndarray) -> float:
    return float(np.mean(np.abs(pred_home - actual_home) + np.abs(pred_away - actual_away)) / 2)


def log_loss_binary(p: np.ndarray, y, eps: float = 1e-15) -> float:
    """Binary proper scoring rule — used for the minutes model's
    P(appearance)/P(60+) forecasts and reusable for any future binary
    probability (e.g. clean-sheet probability calibration in Phase 5).
    Deliberately separate from log_loss() above rather than treating a
    binary outcome as a degenerate 3-class case."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_binary(p: np.ndarray, y) -> float:
    y = np.asarray(y, dtype=float)
    return float(np.mean((np.asarray(p, dtype=float) - y) ** 2))


def ece_binary(p: np.ndarray, y, n_bins: int = 10) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for b in range(n_bins):
        m = (p > bins[b]) & (p <= bins[b + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(e)


def full_binary_metrics(p: np.ndarray, y) -> dict:
    return {
        "n": len(y),
        "log_loss": round(log_loss_binary(p, y), 4),
        "brier": round(brier_binary(p, y), 4),
        "ece": round(ece_binary(p, y), 4),
        "mean_predicted": round(float(np.mean(p)), 4),
        "mean_observed": round(float(np.mean(np.asarray(y, dtype=float))), 4),
    }


def poisson_nll(rate: np.ndarray, actual_count, eps: float = 1e-9) -> float:
    """Mean Poisson negative log-likelihood — the proper scoring rule for
    count forecasts (spec Part XXI) used to evaluate allocated
    expected-goals/expected-assists against actually observed counts."""
    from scipy.stats import poisson
    rate = np.clip(np.asarray(rate, dtype=float), eps, None)
    return float(-np.mean(poisson.logpmf(np.asarray(actual_count, dtype=int), rate)))


def full_metrics(probs: np.ndarray, outcomes, pred_home=None, pred_away=None, actual_home=None, actual_away=None) -> dict:
    m = {
        "n": len(outcomes),
        "log_loss": round(log_loss(probs, outcomes), 4),
        "brier": round(brier(probs, outcomes), 4),
        "rps": round(rps(probs, outcomes), 4),
        "accuracy": round(accuracy(probs, outcomes), 4),
        "ece": round(ece(probs, outcomes), 4),
    }
    if pred_home is not None:
        m["goals_mae"] = round(goals_mae(np.array(pred_home), np.array(pred_away),
                                          np.array(actual_home), np.array(actual_away)), 4)
    return m
