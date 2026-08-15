from __future__ import annotations

import csv

import pytest

from apex_fpl.serving import live_data as ld


@pytest.fixture
def canonical_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ld, "CANONICAL_DIR", tmp_path)
    return tmp_path


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_latest_by_key_deduplicates_to_most_recent_snapshot():
    rows = [
        {"club_id": "1", "name": "Old Name", "retrieved_at": "2026-08-01T00:00:00+00:00"},
        {"club_id": "1", "name": "New Name", "retrieved_at": "2026-08-15T00:00:00+00:00"},
        {"club_id": "2", "name": "Other", "retrieved_at": "2026-08-01T00:00:00+00:00"},
    ]
    latest = ld._latest_by_key(rows, "club_id")
    assert latest["1"]["name"] == "New Name"
    assert latest["2"]["name"] == "Other"
    assert len(latest) == 2


def test_load_clubs_parses_correctly(canonical_dir):
    _write_csv(canonical_dir / "clubs.csv", ["club_id", "code", "name", "short_name", "strength_overall_home", "strength_overall_away", "retrieved_at", "source_snapshot"], [
        {"club_id": "1", "code": "3", "name": "Arsenal", "short_name": "ARS", "strength_overall_home": "4", "strength_overall_away": "5", "retrieved_at": "2026-08-15T00:00:00+00:00", "source_snapshot": "x"},
        {"club_id": "2", "code": "7", "name": "Bournemouth", "short_name": "BOU", "strength_overall_home": "3", "strength_overall_away": "3", "retrieved_at": "2026-08-15T00:00:00+00:00", "source_snapshot": "x"},
    ])
    clubs = ld.load_clubs()
    assert clubs == {1: "Arsenal", 2: "Bournemouth"}


def test_load_players_maps_element_type_to_gk_convention(canonical_dir):
    _write_csv(canonical_dir / "clubs.csv", ["club_id", "name", "retrieved_at"], [{"club_id": "1", "name": "Arsenal", "retrieved_at": "t"}])
    _write_csv(canonical_dir / "players.csv", ["player_id", "web_name", "team_id", "element_type_id", "now_cost", "status", "retrieved_at"], [
        {"player_id": "10", "web_name": "Raya", "team_id": "1", "element_type_id": "1", "now_cost": "60", "status": "a", "retrieved_at": "t"},
        {"player_id": "11", "web_name": "Saka", "team_id": "1", "element_type_id": "3", "now_cost": "100", "status": "a", "retrieved_at": "t"},
    ])
    players = ld.load_players()
    assert players["10"]["position"] == "GK"  # element_type_id=1 maps to "GK", NOT bootstrap-static's own "GKP" string
    assert players["11"]["position"] == "MID"
    assert players["10"]["price"] == 6.0
    assert players["10"]["team"] == "Arsenal"


def test_load_finished_fixtures_filters_to_finished_only(canonical_dir):
    _write_csv(canonical_dir / "clubs.csv", ["club_id", "name", "retrieved_at"], [
        {"club_id": "1", "name": "Arsenal", "retrieved_at": "t"}, {"club_id": "2", "name": "Chelsea", "retrieved_at": "t"},
    ])
    _write_csv(canonical_dir / "fixtures.csv", ["fixture_id", "event_id", "team_h", "team_a", "kickoff_time_utc", "finished", "team_h_score", "team_a_score"], [
        {"fixture_id": "1", "event_id": "1", "team_h": "1", "team_a": "2", "kickoff_time_utc": "2026-08-21T19:00:00Z", "finished": "True", "team_h_score": "2", "team_a_score": "1"},
        {"fixture_id": "2", "event_id": "2", "team_h": "2", "team_a": "1", "kickoff_time_utc": "2026-08-28T19:00:00Z", "finished": "False", "team_h_score": "", "team_a_score": ""},
    ])
    finished = ld.load_finished_fixtures()
    assert len(finished) == 1
    assert finished[0].home_team == "Arsenal" and finished[0].away_score == 1


def test_load_gameweeks_parses_deadlines_and_flags(canonical_dir):
    _write_csv(canonical_dir / "gameweeks.csv", ["event_id", "name", "deadline_time_utc", "is_current", "is_next", "is_previous", "finished", "data_checked", "retrieved_at"], [
        {"event_id": "1", "name": "Gameweek 1", "deadline_time_utc": "2026-08-21T17:30:00Z", "is_current": "False", "is_next": "True", "is_previous": "False", "finished": "False", "data_checked": "False", "retrieved_at": "t"},
        {"event_id": "2", "name": "Gameweek 2", "deadline_time_utc": "2026-08-28T17:30:00Z", "is_current": "False", "is_next": "False", "is_previous": "False", "finished": "False", "data_checked": "False", "retrieved_at": "t"},
    ])
    gws = ld.load_gameweeks()
    assert gws[1]["is_next"] is True
    assert gws[1]["finished"] is False
    assert gws[1]["deadline_time"].year == 2026 and gws[1]["deadline_time"].month == 8
    assert gws[2]["is_next"] is False
