from __future__ import annotations

import numpy as np

from apex_fpl.evaluation import metrics as em


def test_binary_log_loss_perfect_vs_wrong():
    perfect = em.log_loss_binary(np.array([0.999, 0.001]), [1, 0])
    wrong = em.log_loss_binary(np.array([0.001, 0.999]), [1, 0])
    assert perfect < 0.01
    assert wrong > 5


def test_binary_brier_bounds():
    assert em.brier_binary(np.array([1.0, 0.0]), [1, 0]) == 0.0
    assert em.brier_binary(np.array([0.0, 1.0]), [1, 0]) == 1.0


def test_binary_ece_zero_when_perfectly_calibrated_by_bin():
    # 10 predictions at p=0.7, 7 of which are actually 1 -> perfectly calibrated in that bin
    p = np.array([0.7] * 10)
    y = np.array([1] * 7 + [0] * 3)
    assert em.ece_binary(p, y) < 1e-9


def test_full_binary_metrics_reports_predicted_vs_observed_mean():
    out = em.full_binary_metrics(np.array([0.8, 0.6, 0.4]), [1, 1, 0])
    assert out["n"] == 3
    assert abs(out["mean_predicted"] - 0.6) < 1e-9
    assert abs(out["mean_observed"] - 2 / 3) < 1e-4  # full_binary_metrics rounds to 4dp by design


def test_poisson_nll_lower_for_correct_rate():
    actual = [0, 1, 2, 0, 1, 3, 0, 1]
    good_rate = np.full(len(actual), np.mean(actual))
    bad_rate = np.full(len(actual), 5.0)
    assert em.poisson_nll(good_rate, actual) < em.poisson_nll(bad_rate, actual)


def test_poisson_nll_handles_zero_rate_without_crashing():
    val = em.poisson_nll(np.array([0.0, 0.0]), [0, 1])
    assert np.isfinite(val)
