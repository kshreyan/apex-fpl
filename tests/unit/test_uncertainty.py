from __future__ import annotations

import numpy as np

from apex_fpl.simulation.monte_carlo import PlayerSimResult
from apex_fpl.simulation import uncertainty as unc


def _result(points, minutes):
    points = np.array(points, dtype=float)
    minutes = np.array(minutes, dtype=float)
    return PlayerSimResult("p", float(points.mean()), float(points.std()), points, minutes)


def test_decomposition_identity_holds():
    """selection_minutes_variance + aleatoric_variance should reconstruct
    total_variance (law of total variance), for a genuinely mixed case."""
    rng = np.random.default_rng(0)
    n = 20000
    minutes = rng.choice([0, 30, 90], size=n, p=[0.3, 0.2, 0.5])
    points = np.where(
        minutes == 0, 0,
        np.where(minutes == 30, rng.poisson(2, n), rng.poisson(5, n)),
    ).astype(float)
    result = _result(points, minutes)
    d = unc.decompose_variance(result)
    assert abs(d.selection_minutes_variance + d.aleatoric_variance - d.total_variance) < 1e-6
    assert abs(d.selection_minutes_share + d.aleatoric_share - 1.0) < 1e-6


def test_deterministic_state_has_zero_selection_variance():
    """A player who ALWAYS plays 90 minutes (only one played_state ever
    occurs) should show zero selection/minutes variance — all uncertainty
    is aleatoric (scoring randomness), since there's no "which state"
    question at all."""
    rng = np.random.default_rng(1)
    n = 10000
    minutes = np.full(n, 90.0)
    points = rng.poisson(4, n).astype(float)
    result = _result(points, minutes)
    d = unc.decompose_variance(result)
    assert d.selection_minutes_variance < 1e-9
    assert d.aleatoric_share > 0.99


def test_deterministic_scoring_given_state_has_zero_aleatoric_variance():
    """A player whose points are FIXED given their played_state (no
    scoring randomness at all) but whose played_state itself is uncertain
    (50/50 played-or-not) should show mostly selection/minutes variance."""
    n = 10000
    rng = np.random.default_rng(2)
    minutes = rng.choice([0, 90], size=n, p=[0.5, 0.5])
    points = np.where(minutes == 0, 0.0, 6.0)  # deterministic given state
    result = _result(points, minutes)
    d = unc.decompose_variance(result)
    assert d.aleatoric_variance < 1e-9
    assert d.selection_minutes_share > 0.99


def test_played_state_buckets_correctly():
    minutes = np.array([0, 0, 1, 45, 59, 60, 75, 90])
    states = unc._played_state(minutes)
    assert list(states) == [0, 0, 1, 1, 1, 2, 2, 2]


def test_model_disagreement_basic_arithmetic():
    md = unc.model_disagreement(1.5, 1.8)
    assert abs(md.absolute_disagreement - 0.3) < 1e-9
    assert abs(md.relative_disagreement - 0.2) < 1e-9


def test_model_disagreement_zero_when_models_agree():
    md = unc.model_disagreement(1.2, 1.2)
    assert md.absolute_disagreement == 0.0
    assert md.relative_disagreement == 0.0
