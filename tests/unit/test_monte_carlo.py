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
