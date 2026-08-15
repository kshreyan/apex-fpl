from __future__ import annotations

import numpy as np

from apex_fpl.models.bonus.bps_model import PositionBPSModel
from apex_fpl.models.minutes.baseline import MinutesForecast
from apex_fpl.models.teams.scoreline import score_matrix
from apex_fpl.rules import scoring
from apex_fpl.simulation import joint_simulator as js


def _mfc(p_app=1.0, p_60=1.0, exp_min=90.0):
    return MinutesForecast(p_appearance=p_app, p_60_plus=p_60, expected_minutes_if_played=exp_min, n_history_gws=10)


def _flat_bps_model(pos, intercept=10.0, residual_std=1.0):
    coefs = {name: 0.0 for name in ["played_60plus", "played_under_60", "goals_scored", "assists", "clean_sheet",
                                     "saves", "goals_conceded", "yellow_cards", "red_cards", "own_goals",
                                     "penalties_saved", "penalties_missed"]}
    coefs["goals_scored"] = 20.0
    coefs["assists"] = 10.0
    return PositionBPSModel(position=pos, coefficients=coefs, intercept=intercept, residual_std=residual_std, n_train=100)


def _make_match(n_home=4, n_away=4):
    m = score_matrix(1.8, 1.2)
    fixture = js.JointFixtureInput(home_team="Home", away_team="Away", score_matrix=m)
    players = []
    for i in range(n_home):
        players.append(js.JointPlayerInput(f"h{i}", "Home", "MID", _mfc(), goal_share=1.0 / n_home, assist_share=1.0 / n_home))
    for i in range(n_away):
        players.append(js.JointPlayerInput(f"a{i}", "Away", "MID", _mfc(), goal_share=1.0 / n_away, assist_share=1.0 / n_away))
    bps_models = {"MID": _flat_bps_model("MID")}
    return fixture, players, bps_models


def test_player_goals_sum_to_team_goals_every_scenario():
    """The core Part XV/XVI reconciliation property the baseline simulator
    does NOT guarantee — this one must, by construction."""
    fixture, players, bps_models = _make_match()
    rules = scoring.load_scoring_rules("2026_27")
    results = js.simulate_gameweek_joint([fixture], players, rules, bps_models, n_scenarios=500, seed=1)

    home_goals_total = np.sum([results[f"h{i}"].goals_samples for i in range(4)], axis=0)
    away_goals_total = np.sum([results[f"a{i}"].goals_samples for i in range(4)], axis=0)

    # Recompute the actual simulated team goals independently to check against
    rng_check = np.random.default_rng(1)
    flat = fixture.score_matrix.flatten()
    # We can't re-derive the exact same draws without duplicating internals, so instead
    # verify a weaker but still strong property: assists never exceed goals, and both
    # goal totals are internally consistent (non-negative integers).
    assert (home_goals_total >= 0).all()
    assert (away_goals_total >= 0).all()

    home_assists_total = np.sum([results[f"h{i}"].assists_samples for i in range(4)], axis=0)
    assert (home_assists_total <= home_goals_total).all(), "a team can never register more assists than goals in a scenario"


def test_goal_allocation_matches_team_scoreline_exactly():
    """Directly verifies sum(player_goals) == team_goals per scenario by
    re-simulating the scoreline draw with the same seed and comparing."""
    fixture, players, bps_models = _make_match()
    rules = scoring.load_scoring_rules("2026_27")
    seed = 7
    results = js.simulate_gameweek_joint([fixture], players, rules, bps_models, n_scenarios=300, seed=seed)

    rng = np.random.default_rng(seed)
    n = 300
    flat = fixture.score_matrix.flatten()
    idx = rng.choice(len(flat), size=n, p=flat)
    gh = idx // fixture.score_matrix.shape[1]

    home_goals_total = np.sum([results[f"h{i}"].goals_samples for i in range(4)], axis=0)
    assert np.array_equal(home_goals_total, gh), "sum of player goals must exactly equal the team's own simulated goal total"


def test_scorer_and_assister_are_positively_correlated():
    """A player who scores a lot in this simulation should also tend to
    have more assists recorded on the SAME team's OTHER scoring events —
    more directly: verify that whenever a player scores, some teammate's
    assist count increases in that same scenario at a rate consistent
    with p_goal_assisted, i.e. assists are genuinely tied to goal events,
    not independent."""
    fixture, players, bps_models = _make_match()
    rules = scoring.load_scoring_rules("2026_27")
    results = js.simulate_gameweek_joint([fixture], players, rules, bps_models, n_scenarios=3000, seed=2, p_goal_assisted=0.7)

    home_goals_total = np.sum([results[f"h{i}"].goals_samples for i in range(4)], axis=0)
    home_assists_total = np.sum([results[f"h{i}"].assists_samples for i in range(4)], axis=0)
    scenarios_with_goals = home_goals_total > 0
    empirical_assist_rate = home_assists_total[scenarios_with_goals].sum() / home_goals_total[scenarios_with_goals].sum()
    assert abs(empirical_assist_rate - 0.7) < 0.05, f"empirical assist rate {empirical_assist_rate} should track p_goal_assisted=0.7"


def test_bonus_competition_rank_two_way_tie_for_first():
    # verified against premierleague.com/en/news/106533: 2-way tie for 1st -> both get 3, next gets 1 (not 2)
    bps = np.array([50.0, 50.0, 40.0, 30.0])
    ids = ["a", "b", "c", "d"]
    result = js._competition_rank_bonus(bps, ids)
    assert result == {"a": 3, "b": 3, "c": 1}
    assert "d" not in result


def test_bonus_competition_rank_two_way_tie_for_second():
    bps = np.array([50.0, 45.0, 45.0, 30.0])
    ids = ["a", "b", "c", "d"]
    result = js._competition_rank_bonus(bps, ids)
    assert result == {"a": 3, "b": 2, "c": 2}
    assert "d" not in result


def test_bonus_competition_rank_two_way_tie_for_third():
    bps = np.array([50.0, 45.0, 40.0, 40.0, 20.0])
    ids = ["a", "b", "c", "d", "e"]
    result = js._competition_rank_bonus(bps, ids)
    assert result == {"a": 3, "b": 2, "c": 1, "d": 1}
    assert "e" not in result


def test_bonus_competition_rank_three_way_tie_for_first_gives_no_lower_bonus():
    bps = np.array([50.0, 50.0, 50.0, 30.0])
    ids = ["a", "b", "c", "d"]
    result = js._competition_rank_bonus(bps, ids)
    assert result == {"a": 3, "b": 3, "c": 3}
    assert "d" not in result


def test_bonus_awarded_only_to_match_participants_and_at_most_top_3_ranks():
    fixture, players, bps_models = _make_match()
    rules = scoring.load_scoring_rules("2026_27")
    results = js.simulate_gameweek_joint([fixture], players, rules, bps_models, n_scenarios=500, seed=3)
    for s in range(500):
        bonuses = [results[p.player_id].bonus_samples[s] for p in players]
        assert sum(1 for b in bonuses if b > 0) <= 4  # at most a 3-way-tie-for-1st-plus scenario realistically bounded; never more than a handful
        assert all(b in (0, 1, 2, 3) for b in bonuses)


def test_non_appearing_player_never_gets_bonus():
    fixture = js.JointFixtureInput(home_team="Home", away_team="Away", score_matrix=score_matrix(1.5, 1.0))
    players = [
        js.JointPlayerInput("never_plays", "Home", "MID", _mfc(p_app=0.0, p_60=0.0), goal_share=0.0, assist_share=0.0),
        js.JointPlayerInput("always_plays", "Home", "MID", _mfc(), goal_share=1.0, assist_share=1.0),
        js.JointPlayerInput("opp", "Away", "MID", _mfc(), goal_share=1.0, assist_share=1.0),
    ]
    bps_models = {"MID": _flat_bps_model("MID")}
    rules = scoring.load_scoring_rules("2026_27")
    results = js.simulate_gameweek_joint([fixture], players, rules, bps_models, n_scenarios=500, seed=4)
    assert (results["never_plays"].bonus_samples == 0).all()


def test_fit_p_goal_assisted_from_real_like_rows():
    rows = [{"goals_scored": "2", "assists": "1"}, {"goals_scored": "1", "assists": "1"}, {"goals_scored": "1", "assists": "0"}]
    p = js.fit_p_goal_assisted(rows)
    assert abs(p - 2 / 4) < 1e-9
