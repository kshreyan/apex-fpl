from __future__ import annotations

import numpy as np

from apex_fpl.calibration import calibrator as cal


def test_fit_calibrator_improves_a_systematically_overconfident_forecaster():
    """Construct a forecaster that's overconfident: true event rate is 0.5
    but it always predicts 0.9 or 0.1. A calibrator should learn to pull
    these toward 0.5."""
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.binomial(1, 0.5, n)
    p_raw = np.where(y == 1, 0.9, 0.1)  # "confidently correct" but overconfident in magnitude
    # add some noise so it's not perfectly separable
    flip = rng.random(n) < 0.3
    y_noisy = np.where(flip, 1 - y, y)

    c = cal.fit_calibrator(p_raw, y_noisy)
    calibrated = c.transform(p_raw)
    raw_ll = cal.em.log_loss_binary(p_raw, y_noisy)
    cal_ll = cal.em.log_loss_binary(calibrated, y_noisy)
    assert cal_ll <= raw_ll + 1e-9


def test_calibrator_transform_returns_valid_probabilities():
    rng = np.random.default_rng(1)
    p = rng.random(200)
    y = rng.binomial(1, p)
    c = cal.fit_calibrator(p, y)
    out = c.transform(rng.random(50))
    assert (out >= 0).all() and (out <= 1).all()


def test_reliability_table_bins_cover_predictions_and_sum_to_total_n():
    rng = np.random.default_rng(2)
    p = rng.random(500)
    y = rng.binomial(1, p)
    table = cal.reliability_table(p, y, n_bins=10)
    total_n = sum(row["n"] for row in table)
    assert total_n == 500
    for row in table:
        assert 0 <= row["empirical_frequency"] <= 1
        assert row["bin_low"] < row["bin_high"]


def test_calibration_slope_intercept_near_one_zero_for_well_calibrated_forecaster():
    rng = np.random.default_rng(3)
    n = 5000
    p = rng.uniform(0.05, 0.95, n)
    y = rng.binomial(1, p)  # genuinely well-calibrated by construction
    slope, intercept = cal.calibration_slope_intercept(p, y)
    assert abs(slope - 1.0) < 0.2
    assert abs(intercept) < 0.2


def test_calibration_slope_flags_overconfidence():
    """An overconfident forecaster (predictions pushed toward 0/1 relative
    to the true rate) should show a calibration slope < 1."""
    rng = np.random.default_rng(4)
    n = 5000
    true_p = rng.uniform(0.2, 0.8, n)
    y = rng.binomial(1, true_p)
    logit_true = np.log(true_p / (1 - true_p))
    overconfident_p = 1 / (1 + np.exp(-2.0 * logit_true))  # exaggerates the signal
    slope, _ = cal.calibration_slope_intercept(overconfident_p, y)
    assert slope < 1.0
