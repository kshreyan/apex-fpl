from __future__ import annotations

import numpy as np

from apex_fpl.evaluation import metrics as em


def test_perfect_forecast_has_near_zero_loss():
    probs = np.array([[0.999, 0.0005, 0.0005], [0.0005, 0.999, 0.0005], [0.0005, 0.0005, 0.999]])
    outcomes = ["H", "D", "A"]
    assert em.log_loss(probs, outcomes) < 0.01
    assert em.brier(probs, outcomes) < 0.01
    assert em.rps(probs, outcomes) < 0.01
    assert em.accuracy(probs, outcomes) == 1.0


def test_maximally_wrong_forecast_has_high_loss():
    probs = np.array([[0.0005, 0.0005, 0.999]])  # confidently predicts away win
    outcomes = ["H"]  # actually home win
    assert em.log_loss(probs, outcomes) > 5
    assert em.brier(probs, outcomes) > 1.5
    assert em.accuracy(probs, outcomes) == 0.0


def test_rps_penalizes_far_misses_more_than_near_misses():
    # Ordered outcome space H < D < A: predicting D when H happens should be
    # a smaller RPS penalty than predicting A when H happens.
    outcomes = ["H"]
    near_miss = np.array([[0.1, 0.9, 0.0]])   # predicted D, actual H (adjacent)
    far_miss = np.array([[0.1, 0.0, 0.9]])    # predicted A, actual H (opposite ends)
    assert em.rps(near_miss, outcomes) < em.rps(far_miss, outcomes)


def test_uniform_forecast_baseline_log_loss():
    probs = np.array([[1 / 3, 1 / 3, 1 / 3]] * 3)
    outcomes = ["H", "D", "A"]
    assert abs(em.log_loss(probs, outcomes) - np.log(3)) < 1e-6


def test_goals_mae_zero_for_exact_predictions():
    assert em.goals_mae(np.array([2, 1]), np.array([0, 1]), np.array([2, 1]), np.array([0, 1])) == 0.0


def test_full_metrics_includes_goals_mae_only_when_provided():
    probs = np.array([[0.5, 0.3, 0.2]])
    out = em.full_metrics(probs, ["H"])
    assert "goals_mae" not in out
    out2 = em.full_metrics(probs, ["H"], pred_home=[1.5], pred_away=[1.0], actual_home=[2], actual_away=[1])
    assert "goals_mae" in out2
