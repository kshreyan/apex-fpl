"""Unit tests for apex_fpl.models.defensive.defcon_model (Phase 13
Block 2.2)."""
from __future__ import annotations

from apex_fpl.models.defensive import defcon_model as dc


def test_defcon_action_count_def_excludes_recoveries():
    row = {"clearances_blocks_interceptions": 6, "tackles": 2, "recoveries": 99}
    assert dc.defcon_action_count(row, "DEF") == 8


def test_defcon_action_count_mid_includes_recoveries():
    row = {"clearances_blocks_interceptions": 4, "tackles": 2, "recoveries": 4}
    assert dc.defcon_action_count(row, "MID") == 10


def test_defcon_action_count_fwd_includes_recoveries():
    row = {"clearances_blocks_interceptions": 1, "tackles": 1, "recoveries": 1}
    assert dc.defcon_action_count(row, "FWD") == 3


def test_defcon_action_count_gk_is_zero():
    row = {"clearances_blocks_interceptions": 5, "tackles": 5, "recoveries": 5}
    assert dc.defcon_action_count(row, "GK") == 0


def test_forecast_defcon_hit_probability_gk_always_zero():
    assert dc.forecast_defcon_hit_probability([50, 50, 50], "GK") == 0.0


def test_forecast_defcon_hit_probability_empty_history_is_zero_not_a_guess():
    assert dc.forecast_defcon_hit_probability([], "DEF") == 0.0


def test_forecast_defcon_hit_probability_high_history_gives_high_probability():
    """A DEF who consistently records well above the threshold=10 should
    forecast a high P(hit) this gameweek."""
    p = dc.forecast_defcon_hit_probability([15, 16, 14, 15, 15], "DEF")
    assert p > 0.7


def test_forecast_defcon_hit_probability_low_history_gives_low_probability():
    p = dc.forecast_defcon_hit_probability([1, 2, 0, 1, 2], "DEF")
    assert p < 0.1


def test_forecast_defcon_hit_probability_recent_form_weighted_more_than_old():
    """A player whose count RECENTLY rose toward the threshold should
    forecast higher than one whose count recently fell, given the same
    average -- recency weighting, not a flat mean."""
    rising = dc.forecast_defcon_hit_probability([2, 4, 6, 8, 10], "DEF")
    falling = dc.forecast_defcon_hit_probability([10, 8, 6, 4, 2], "DEF")
    assert rising > falling


def test_forecast_defcon_expected_points_scales_by_defcon_points():
    p = dc.forecast_defcon_hit_probability([15, 16, 14, 15, 15], "DEF")
    ep = dc.forecast_defcon_expected_points([15, 16, 14, 15, 15], "DEF")
    assert ep == p * dc.DEFCON_POINTS


def test_thresholds_match_configs_seasons_2026_27():
    assert dc.DEFCON_THRESHOLDS == {"DEF": 10, "MID": 12, "FWD": 12, "GK": None}
    assert dc.DEFCON_POINTS == 2
