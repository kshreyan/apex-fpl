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


def _write_snapshot(raw_root, gw_dir, filename, highest_settled):
    """One captured snapshot representing the pipeline's state right
    after gameweek `highest_settled` settled -- callers write one of
    these PER settled gameweek to simulate the daily pipeline actually
    having run and captured evidence at each boundary in turn (a single
    snapshot jumping straight to a high highest_settled value does NOT
    imply the intermediate gameweeks were ever captured -- see
    apex_fpl.serving.gameweek_history's own "skip the gap, don't guess"
    behavior, which this count must reflect)."""
    d = raw_root / gw_dir / "bootstrap_static"
    d.mkdir(parents=True, exist_ok=True)
    events = [{"id": i, "finished": i <= highest_settled, "data_checked": i <= highest_settled} for i in range(1, 8)]
    (d / filename).write_text('{"events": %s, "elements": []}' % __import__("json").dumps(events))


def test_usable_settled_gameweek_count_below_transition_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(rpr.gwh, "RAW_DATA_ROOT", tmp_path)
    for gw in range(1, 5):
        _write_snapshot(tmp_path, f"gw0{gw}", "a.json", highest_settled=gw)

    count = rpr.usable_settled_gameweek_count(target_gw=5)

    assert count == 4
    assert count < rpr.IN_SEASON_TRANSITION_MIN_SETTLED_GWS


def test_usable_settled_gameweek_count_at_transition_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(rpr.gwh, "RAW_DATA_ROOT", tmp_path)
    for gw in range(1, 7):
        _write_snapshot(tmp_path, f"gw0{gw}", "a.json", highest_settled=gw)

    count = rpr.usable_settled_gameweek_count(target_gw=7)

    assert count == 6
    assert count >= rpr.IN_SEASON_TRANSITION_MIN_SETTLED_GWS


def test_usable_settled_gameweek_count_ignores_snapshots_after_target_gw(tmp_path, monkeypatch):
    """A settlement snapshot for gw=6 must not count toward a target_gw=5
    recommendation -- that would be leakage (using a result the target
    gameweek's own prediction must not have seen yet)."""
    monkeypatch.setattr(rpr.gwh, "RAW_DATA_ROOT", tmp_path)
    _write_snapshot(tmp_path, "gw06", "a.json", highest_settled=6)

    count = rpr.usable_settled_gameweek_count(target_gw=5)

    assert count == 0
