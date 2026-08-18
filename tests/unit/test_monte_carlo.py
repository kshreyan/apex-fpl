from __future__ import annotations

import numpy as np

from apex_fpl.models.minutes.baseline import MinutesForecast
from apex_fpl.models.teams.scoreline import score_matrix
from apex_fpl.rules import scoring
from apex_fpl.simulation import monte_carlo as mc


def test_vectorized_scoring_matches_scoring_engine():
    """The simulation's fast-path numpy scoring must never silently drift
    from the authoritative deterministic scoring engine — this is the
    single most important correctness property of the simulator, since
    everything downstream (optimizer input) depends on it."""
    rules = scoring.load_scoring_rules("2026_27")
    rng = np.random.default_rng(0)

    n = 500
    for position in ["GK", "DEF", "MID", "FWD"]:
        minutes = rng.choice([0, 23, 60, 90], size=n)
        goals = rng.integers(0, 4, size=n)
        assists = rng.integers(0, 3, size=n)
        clean_sheet = rng.random(n) < 0.3
        goals_conceded = rng.integers(0, 5, size=n)

        vectorized = mc._vectorized_points(position, minutes, goals, assists, clean_sheet, goals_conceded, rules)

        for i in range(n):
            events = scoring.PlayerMatchEvents(
                position=position, minutes=int(minutes[i]), goals_scored=int(goals[i]),
                assists=int(assists[i]), clean_sheet=bool(clean_sheet[i]),
                goals_conceded=int(goals_conceded[i]),
            )
            expected = scoring.score_player_gameweek(events, rules)
            assert vectorized[i] == expected, f"mismatch at position={position} i={i}: {vectorized[i]} != {expected}"


def test_simulate_gameweek_converges_and_produces_sane_results():
    rules = scoring.load_scoring_rules("2026_27")
    m = score_matrix(1.6, 1.0)
    fixture = mc.FixtureInput(home_team="A", away_team="B", score_matrix=m)

    striker = mc.PlayerInput(
        player_id="striker", team="A", position="FWD",
        minutes_forecast=MinutesForecast(p_appearance=0.9, p_60_plus=0.8, expected_minutes_if_played=85, n_history_gws=6),
        expected_goals=0.6, expected_assists=0.1,
    )
    keeper = mc.PlayerInput(
        player_id="keeper", team="B", position="GK",
        minutes_forecast=MinutesForecast(p_appearance=1.0, p_60_plus=1.0, expected_minutes_if_played=90, n_history_gws=6),
        expected_goals=0.0, expected_assists=0.0,
    )
    bench_warmer = mc.PlayerInput(
        player_id="bench", team="A", position="MID",
        minutes_forecast=MinutesForecast(p_appearance=0.1, p_60_plus=0.0, expected_minutes_if_played=15, n_history_gws=6),
        expected_goals=0.02, expected_assists=0.01,
    )

    results = mc.simulate_gameweek([fixture], [striker, keeper, bench_warmer], rules, batch=2000, max_sims=20000, tol=0.03)

    assert results["striker"].mean_points > results["bench"].mean_points, "a nailed-on striker should out-score a fringe bench player on average"
    assert results["keeper"].mean_points > 0, "keeper facing the weaker attacking side should have positive expected points"
    for r in results.values():
        assert len(r.samples) >= 2000
        assert r.std_points >= 0


def test_single_fixture_per_team_is_byte_identical_to_the_pre_dgw_fix_baseline():
    """Regression guard for the double-gameweek fix: for the overwhelming
    common case (one fixture per team), the refactored per-fixture loop
    must reduce to EXACTLY the same RNG call sequence and formula as
    before -- not just statistically similar, byte-identical. These
    expected values were captured from the pre-fix code with the same
    seed/inputs; any drift here means the single-fixture path changed
    behavior, not just the new double-gameweek path."""
    rules = scoring.load_scoring_rules("2026_27")
    m = score_matrix(1.6, 1.0)
    fixture = mc.FixtureInput(home_team="A", away_team="B", score_matrix=m)
    striker = mc.PlayerInput(
        player_id="striker", team="A", position="FWD",
        minutes_forecast=MinutesForecast(p_appearance=0.9, p_60_plus=0.8, expected_minutes_if_played=85, n_history_gws=6),
        expected_goals=0.6, expected_assists=0.1,
    )
    keeper = mc.PlayerInput(
        player_id="keeper", team="B", position="GK",
        minutes_forecast=MinutesForecast(p_appearance=1.0, p_60_plus=1.0, expected_minutes_if_played=90, n_history_gws=6),
        expected_goals=0.0, expected_assists=0.0,
    )
    bench_warmer = mc.PlayerInput(
        player_id="bench", team="A", position="MID",
        minutes_forecast=MinutesForecast(p_appearance=0.1, p_60_plus=0.0, expected_minutes_if_played=15, n_history_gws=6),
        expected_goals=0.02, expected_assists=0.01,
    )

    results = mc.simulate_gameweek([fixture], [striker, keeper, bench_warmer], rules, batch=2000, max_sims=20000, tol=0.03)

    assert results["striker"].mean_points == 4.339
    assert results["keeper"].mean_points == 2.1965
    assert results["bench"].mean_points == 0.2305


def test_double_gameweek_player_can_score_appearance_points_from_both_fixtures():
    """The actual bug: a team playing twice used to silently overwrite
    team_fixture[team] with only the SECOND fixture, so a double-
    gameweek player's minutes/clean-sheet/goals-conceded for the FIRST
    fixture were dropped entirely -- not an approximation, an outright
    miss. A nailed-on player who plays both matches must be able to earn
    two lots of appearance points, not one."""
    rules = scoring.load_scoring_rules("2026_27")
    m = score_matrix(1.6, 1.0)
    fixture_1 = mc.FixtureInput(home_team="A", away_team="B", score_matrix=m)
    fixture_2 = mc.FixtureInput(home_team="A", away_team="C", score_matrix=m)

    nailed_on = mc.PlayerInput(
        player_id="nailed_on", team="A", position="MID",
        minutes_forecast=MinutesForecast(p_appearance=1.0, p_60_plus=1.0, expected_minutes_if_played=90, n_history_gws=6),
        expected_goals=0.0, expected_assists=0.0,
    )
    single_fixture_twin = mc.PlayerInput(
        player_id="single_fixture_twin", team="B", position="MID",
        minutes_forecast=MinutesForecast(p_appearance=1.0, p_60_plus=1.0, expected_minutes_if_played=90, n_history_gws=6),
        expected_goals=0.0, expected_assists=0.0,
    )

    results = mc.simulate_gameweek([fixture_1, fixture_2], [nailed_on, single_fixture_twin], rules, batch=2000, max_sims=20000, tol=0.03)

    # A guaranteed starter in both of a double gameweek's matches must earn
    # appearance points from BOTH -- roughly double a single-fixture player
    # with the same profile, not equal to it (the pre-fix bug would have
    # made these equal, silently dropping the first fixture).
    assert results["nailed_on"].mean_points > 1.5 * results["single_fixture_twin"].mean_points
    assert all(results["nailed_on"].minutes_samples == 180.0), "both fixtures' 90 minutes must be summed, not overwritten"


def test_double_gameweek_pooled_goals_are_credited_exactly_once_not_twice():
    """The pooled goals/assists draw must be scored on exactly one of the
    two fixtures' _vectorized_points calls -- summing goals*points_per_
    goal across BOTH fixture passes would silently double-count a
    player's combined-goals allocation (already summed once upstream).
    Deterministic minutes (p_appearance=p_60_plus=1.0) and FWD's own
    zero clean-sheet points isolate the comparison to exactly two
    effects: the DGW player's guaranteed extra appearance points from a
    second 90-minute match (+2, deterministic), and the pooled goals
    draw -- which must land ONCE either way, not doubled for the DGW
    player."""
    rules = scoring.load_scoring_rules("2026_27")
    assert rules["clean_sheet"]["FWD"] == 0, "test relies on FWD clean-sheet points being zero to isolate the goals-double-counting effect"
    m = score_matrix(1.6, 1.0)
    fixture_1 = mc.FixtureInput(home_team="A", away_team="B", score_matrix=m)
    fixture_2 = mc.FixtureInput(home_team="A", away_team="C", score_matrix=m)

    forward_dgw = mc.PlayerInput(
        player_id="forward_dgw", team="A", position="FWD",
        minutes_forecast=MinutesForecast(p_appearance=1.0, p_60_plus=1.0, expected_minutes_if_played=90, n_history_gws=6),
        expected_goals=1.2, expected_assists=0.0,
    )
    forward_single = mc.PlayerInput(
        player_id="forward_single", team="B", position="FWD",
        minutes_forecast=MinutesForecast(p_appearance=1.0, p_60_plus=1.0, expected_minutes_if_played=90, n_history_gws=6),
        expected_goals=1.2, expected_assists=0.0,
    )

    results = mc.simulate_gameweek([fixture_1, fixture_2], [forward_dgw, forward_single], rules, batch=3000, max_sims=30000, tol=0.03)

    expected_extra_appearance_points = rules["appearance"]["at_least_60_min"]  # exactly one extra guaranteed 90-min match
    goal_points = rules["goals_scored"]["FWD"]
    wrong_extra_if_goals_doubled = expected_extra_appearance_points + forward_dgw.expected_goals * goal_points
    actual_diff = results["forward_dgw"].mean_points - results["forward_single"].mean_points

    assert abs(actual_diff - expected_extra_appearance_points) < 0.5
    assert abs(actual_diff - wrong_extra_if_goals_doubled) > 1.5
