"""Focused coverage for the double-gameweek fix in
scripts/run_production_recommendation.py -- this used to be a plain dict
overwrite that silently dropped a double-gameweek team's second fixture
entirely (found while building the automated pipeline, not previously
caught by anything). Testing the extracted aggregation function directly
rather than the full generate_recommendation() pipeline, which needs a
live Silver data layer this test has no business depending on.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_production_recommendation as rpr  # noqa: E402 -- matches this repo's existing scripts/ convention (not a package)


class _FakeTeamModel:
    """expected_goals returns a FIXED value per (home, away) pair,
    independent of date -- enough to prove the aggregation logic sums
    correctly, without needing the real time-decay model."""

    def __init__(self, fixed_goals: dict[tuple[str, str], tuple[float, float]]):
        self.fixed_goals = fixed_goals

    def expected_goals(self, home_team, away_team, at_date):
        return self.fixed_goals[(home_team, away_team)]


def _fixture(home, away, date="2026-08-21"):
    return {"home_team": home, "away_team": away, "date": datetime.fromisoformat(date)}


def test_single_fixture_per_team_is_unaffected():
    model = _FakeTeamModel({("Arsenal", "Chelsea"): (2.0, 1.0)})
    fixtures = [_fixture("Arsenal", "Chelsea")]

    _, _, team_goals = rpr.build_fixture_inputs_and_team_goals(fixtures, model)

    assert team_goals == {"Arsenal": [2.0], "Chelsea": [1.0]}


def test_double_gameweek_team_gets_both_fixtures_not_the_last_one_only():
    """The actual bug: Arsenal plays twice. The old code's plain dict
    assignment would have left team_goals['Arsenal'] holding ONLY the
    second fixture's value (1.5), silently discarding the first (2.0)."""
    model = _FakeTeamModel({
        ("Arsenal", "Chelsea"): (2.0, 1.0),
        ("Everton", "Arsenal"): (0.8, 1.5),
    })
    fixtures = [_fixture("Arsenal", "Chelsea", "2026-08-21"), _fixture("Everton", "Arsenal", "2026-08-24")]

    _, _, team_goals = rpr.build_fixture_inputs_and_team_goals(fixtures, model)

    assert team_goals["Arsenal"] == [2.0, 1.5]  # BOTH fixtures present, not just the last
    assert sum(team_goals["Arsenal"]) == 3.5
    assert team_goals["Chelsea"] == [1.0]
    assert team_goals["Everton"] == [0.8]


def test_fixture_inputs_and_meta_include_one_entry_per_fixture_not_per_team():
    model = _FakeTeamModel({
        ("Arsenal", "Chelsea"): (2.0, 1.0),
        ("Everton", "Arsenal"): (0.8, 1.5),
    })
    fixtures = [_fixture("Arsenal", "Chelsea"), _fixture("Everton", "Arsenal")]

    fixture_inputs, fixture_meta, _ = rpr.build_fixture_inputs_and_team_goals(fixtures, model)

    assert len(fixture_inputs) == 2  # both real matches simulated, not collapsed into one
    assert len(fixture_meta) == 2
    assert {(f["home_team"], f["away_team"]) for f in fixture_meta} == {("Arsenal", "Chelsea"), ("Everton", "Arsenal")}
