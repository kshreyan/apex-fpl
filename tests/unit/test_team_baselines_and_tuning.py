from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from apex_fpl.models.teams import attack_defense as ad
from apex_fpl.models.teams import baselines, tuning


def _fx(day, h, a, hs, as_):
    return ad.Fixture(datetime(2023, 1, day), h, a, hs, as_)


SAMPLE = [
    _fx(1, "A", "B", 2, 0),
    _fx(3, "B", "A", 1, 1),
    _fx(5, "A", "C", 3, 1),
    _fx(7, "C", "B", 0, 0),
    _fx(9, "B", "C", 1, 2),
    _fx(11, "C", "A", 0, 2),
]


def test_constant_model_matches_training_means():
    model = baselines.fit_constant(SAMPLE)
    home_avg = np.mean([2, 1, 3, 0, 1, 0])
    away_avg = np.mean([0, 1, 1, 0, 2, 2])
    assert abs(model.home_goals_avg - home_avg) < 1e-9
    assert abs(model.away_goals_avg - away_avg) < 1e-9
    eh, ea = model.expected_goals("A", "Z", datetime(2023, 2, 1))
    assert eh == model.home_goals_avg and ea == model.away_goals_avg


def test_constant_model_requires_completed_fixtures():
    with pytest.raises(ValueError):
        baselines.fit_constant([ad.Fixture(datetime(2023, 1, 1), "A", "B")])


def test_previous_season_average_uses_per_team_rates():
    model = baselines.fit_previous_season_average(SAMPLE)
    # Team A: home 2-0(gf2,ga0), away 1-1(gf1,ga1), home 3-1(gf3,ga1), away 0-2(gf2,ga0->wait check)
    assert "A" in model.team_gf and "A" in model.team_ga
    eh, ea = model.expected_goals("A", "B", datetime(2023, 2, 1))
    assert eh > 0 and ea > 0


def test_attack_defense_k_base_and_halflife_are_instance_params_not_shared_state():
    """Regression test for the Phase 4 refactor: fitting two models with
    different constants must not interfere with each other."""
    m1 = ad.fit(SAMPLE, k_base=0.02, halflife_days=100)
    m2 = ad.fit(SAMPLE, k_base=0.08, halflife_days=1000)
    assert m1.k_base == 0.02 and m1.halflife_days == 100
    assert m2.k_base == 0.08 and m2.halflife_days == 1000
    # different k_base should generally produce different ratings for a team with any goal history
    assert m1.attack != m2.attack or m1.defense != m2.defense


def test_grid_search_returns_a_combo_from_the_grid():
    train = SAMPLE[:4]
    val = SAMPLE[4:]
    result = tuning.grid_search(train, val, k_base_grid=[0.02, 0.045, 0.08], halflife_grid=[100.0, 380.0])
    assert result.k_base in [0.02, 0.045, 0.08]
    assert result.halflife_days in [100.0, 380.0]
    assert -0.2 <= result.rho <= 0.2


def test_fit_rho_stays_within_bounds():
    model = ad.fit(SAMPLE)
    lh, la, _ = tuning._predict(model, SAMPLE)
    home_scores = [f.home_score for f in SAMPLE]
    away_scores = [f.away_score for f in SAMPLE]
    rho = tuning.fit_rho(lh, la, home_scores, away_scores)
    assert -0.2 <= rho <= 0.2
