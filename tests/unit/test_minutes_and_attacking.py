from __future__ import annotations

from apex_fpl.models.attacking import proportional as prop
from apex_fpl.models.minutes import baseline as mb


def test_minutes_forecast_neutral_prior_when_no_history():
    fc = mb.forecast_minutes([])
    assert fc.n_history_gws == 0
    assert fc.p_appearance == mb.NEUTRAL_PRIOR_P_ANY
    assert fc.p_60_plus == mb.NEUTRAL_PRIOR_P60


def test_minutes_forecast_regular_starter():
    fc = mb.forecast_minutes([90, 90, 85, 90, 0, 90], lookback=6)
    assert fc.n_history_gws == 6
    assert fc.p_60_plus == 5 / 6
    assert fc.p_appearance == 5 / 6
    assert 85 <= fc.expected_minutes_if_played <= 90


def test_minutes_forecast_respects_lookback_window():
    history = [90] * 3 + [0] * 10  # started the first 3 GWs, then dropped out
    recent = mb.forecast_minutes(history[-3:], lookback=3)
    assert recent.p_appearance == 0.0


def test_attacking_shares_proportional_to_goals():
    history = {
        "star": [(2, 0), (1, 1), (0, 1)],   # 3 goals, 2 assists
        "squad_player": [(0, 0), (0, 1), (0, 0)],  # 0 goals, 1 assist
    }
    shares = prop.compute_shares(history)
    assert shares["star"].goal_share == 1.0  # squad_player scored 0
    assert abs(shares["star"].assist_share - 2 / 3) < 1e-9
    assert abs(shares["squad_player"].assist_share - 1 / 3) < 1e-9


def test_attacking_shares_fallback_uniform_when_team_scored_nothing():
    history = {"a": [(0, 0)], "b": [(0, 0)]}
    shares = prop.compute_shares(history)
    assert shares["a"].goal_share == 0.5
    assert shares["b"].goal_share == 0.5


def test_allocate_scales_by_team_expected_goals():
    shares = {"a": prop.AttackingShare(0.6, 0.4), "b": prop.AttackingShare(0.4, 0.6)}
    out = prop.allocate(2.0, shares)
    assert abs(out["a"][0] - 1.2) < 1e-9
    assert abs(out["b"][0] - 0.8) < 1e-9
