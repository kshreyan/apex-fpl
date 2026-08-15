"""Binary probability calibration (spec Part XX).

Generalizes the pattern from the World Cup predictor project's
calibration_utils.py (docs/world_cup_transfer_audit.md classification B:
portable mechanism, but the original hardcodes exactly 3 classes) into a
binary calibrator usable for any FPL event probability — start/appearance,
60-minute, clean-sheet, defensive-contribution, haul, etc. all reduce to
"P(binary event) calibrated against realized outcome."

Both isotonic regression and Platt (logistic/sigmoid) scaling are fit;
the one with lower log loss on a DEDICATED calibration-fitting set is
selected — mirroring exactly how the World Cup repo chose between them,
and consistent with spec Part XX's explicit requirement that "calibration
must not use the evaluation sample."
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from apex_fpl.evaluation import metrics as em


@dataclass
class BinaryCalibrator:
    method: str  # "isotonic" | "platt" | "none"
    model: object | None

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.method == "none" or self.model is None:
            return p
        if self.method == "isotonic":
            return np.clip(self.model.predict(p), 1e-6, 1 - 1e-6)
        # platt
        return np.clip(self.model.predict_proba(p.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)


def fit_calibrator(p_raw: np.ndarray, y: np.ndarray) -> BinaryCalibrator:
    """Fits isotonic and Platt calibrators on (p_raw, y), picks whichever
    achieves lower log loss ON THIS SAME FITTING SET (standard practice for
    choosing the calibration *method*; the resulting calibrator is then
    applied to a genuinely separate test set by the caller — this function
    only ever sees the calibration-fitting data, never the evaluation
    data)."""
    p_raw = np.asarray(p_raw, dtype=float)
    y = np.asarray(y, dtype=int)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_raw, y)
    iso_cal = BinaryCalibrator("isotonic", iso)

    platt = LogisticRegression(C=1e6, solver="lbfgs")
    platt.fit(p_raw.reshape(-1, 1), y)
    platt_cal = BinaryCalibrator("platt", platt)

    none_cal = BinaryCalibrator("none", None)

    candidates = [none_cal, iso_cal, platt_cal]
    losses = [em.log_loss_binary(c.transform(p_raw), y) for c in candidates]
    best = candidates[int(np.argmin(losses))]
    return best


def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Per-bin (mean_predicted, empirical_frequency, count) — the
    substantive content of a reliability diagram, returned as structured
    data rather than a rendered plot (no plotting dependency added for
    this; the data fully determines the diagram)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for b in range(n_bins):
        mask = (p > bins[b]) & (p <= bins[b + 1]) if b > 0 else (p >= bins[b]) & (p <= bins[b + 1])
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "bin_low": round(float(bins[b]), 2), "bin_high": round(float(bins[b + 1]), 2),
            "n": n, "mean_predicted": round(float(p[mask].mean()), 4),
            "empirical_frequency": round(float(y[mask].mean()), 4),
        })
    return rows


def calibration_slope_intercept(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fits logit(y) ~ a + b*logit(p) via logistic regression on the logit-
    transformed predictions (the standard calibration-slope/intercept
    diagnostic: slope=1, intercept=0 is perfect calibration; slope<1
    indicates overconfidence, slope>1 underconfidence)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    y = np.asarray(y, dtype=int)
    if len(set(y.tolist())) < 2:
        return float("nan"), float("nan")
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(logit_p, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])
