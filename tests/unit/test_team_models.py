from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from apex_fpl.models.teams import attack_defense as ad
from apex_fpl.models.teams import scoreline as sl


def _fx(day, h, a, hs=None, as_=None):
    return ad.Fixture(datetime(2023, 1, day), h, a, hs, as_)


def test_attack_defense_fit_requires_completed_fixtures():
    with pytest.raises(ValueError):
        ad.fit([_fx(1, "A", "B")])  # no scores at all


def test_attack_defense_produces_finite_expected_goals():
    fixtures = [
        _fx(1, "A", "B", 2, 0),
        _fx(3, "B", "A", 1, 1),
        _fx(5, "A", "C", 3, 1),
        _fx(7, "C", "B", 0, 0),
    ]
    model = ad.fit(fixtures)
    eh, ea = model.expected_goals("A", "B", datetime(2023, 1, 10))
    assert 0.05 <= eh <= 6.0
    assert 0.05 <= ea <= 6.0


def test_stronger_attack_team_gets_higher_expected_goals():
    # Team A scores heavily and concedes little; team Z is the opposite.
    fixtures = [
        _fx(1, "A", "Z", 4, 0),
        _fx(3, "Z", "A", 0, 3),
        _fx(5, "A", "Z", 3, 0),
        _fx(7, "Z", "A", 0, 2),
    ]
    model = ad.fit(fixtures)
    eh_a_home, ea_z_away = model.expected_goals("A", "Z", datetime(2023, 1, 10))
    eh_z_home, ea_a_away = model.expected_goals("Z", "A", datetime(2023, 1, 10))
    assert eh_a_home > ea_a_away, "A at home should be expected to outscore A away, given A is much stronger"
    assert eh_a_home > eh_z_home, "A's home expected goals should exceed Z's home expected goals"


def test_score_matrix_sums_to_one_and_is_nonnegative():
    m = sl.score_matrix(1.4, 1.1)
    assert m.shape == (sl.MAX_GOALS + 1, sl.MAX_GOALS + 1)
    assert np.isclose(m.sum(), 1.0)
    assert (m >= 0).all()


def test_clean_sheet_prob_bounds_and_symmetry():
    m = sl.score_matrix(0.01, 0.01)  # both teams essentially never score
    home_cs = sl.clean_sheet_prob(m, "home")
    away_cs = sl.clean_sheet_prob(m, "away")
    assert home_cs > 0.9
    assert away_cs > 0.9

    m2 = sl.score_matrix(3.0, 3.0)  # both teams score heavily
    assert sl.clean_sheet_prob(m2, "home") < 0.1
    assert sl.clean_sheet_prob(m2, "away") < 0.1
