"""Unit tests for Silver-layer parsing logic, using small synthetic payloads
(no network access) so these run offline and fast."""
from __future__ import annotations

from apex_fpl.entities import schema, silver

SAMPLE_BOOTSTRAP = {
    "teams": [
        {"id": 1, "code": 3, "name": "Arsenal", "short_name": "ARS",
         "strength_overall_home": 4, "strength_overall_away": 5},
    ],
    "element_types": [
        {"id": 1, "singular_name": "Goalkeeper", "plural_name_short": "GKP",
         "squad_select": 2, "squad_min_play": 1, "squad_max_play": 1},
    ],
    "events": [
        {"id": 1, "name": "Gameweek 1", "deadline_time": "2026-08-21T17:30:00Z",
         "is_current": False, "is_next": True, "is_previous": False,
         "finished": False, "data_checked": False},
    ],
    "elements": [
        {"id": 1, "code": 154561, "web_name": "Raya", "first_name": "David",
         "second_name": "Raya Martin", "team": 1, "element_type": 1, "status": "a",
         "news": "", "news_added": None, "chance_of_playing_this_round": None,
         "chance_of_playing_next_round": None, "now_cost": 60, "selected_by_percent": "31.2",
         "event_points": 0, "total_points": 162, "minutes": 3330, "goals_scored": 0,
         "assists": 0, "clean_sheets": 19, "goals_conceded": 26, "bonus": 11, "bps": 633,
         "saves": 60, "defensive_contribution": 0, "expected_goals": "0.00",
         "expected_assists": "0.07", "form": "0.0", "points_per_game": "4.4"},
    ],
}

SAMPLE_FIXTURES = [
    {"id": 1, "event": 1, "team_h": 1, "team_a": 7, "kickoff_time": "2026-08-21T19:00:00Z",
     "finished": False, "team_h_score": None, "team_a_score": None,
     "team_h_difficulty": 2, "team_a_difficulty": 5},
]

SAMPLE_META = {"retrieved_at": "2026-08-14T16:49:35+00:00"}


def test_parse_bootstrap_produces_all_tables_with_correct_row_counts():
    tables = silver.parse_bootstrap(SAMPLE_BOOTSTRAP, SAMPLE_META, "snap1.json")
    assert len(tables["clubs"]) == 1
    assert len(tables["positions"]) == 1
    assert len(tables["gameweeks"]) == 1
    assert len(tables["players"]) == 1
    assert len(tables["player_stats"]) == 1


def test_parse_bootstrap_player_row_matches_source_and_excludes_stats():
    tables = silver.parse_bootstrap(SAMPLE_BOOTSTRAP, SAMPLE_META, "snap1.json")
    player = tables["players"][0]
    assert player["player_id"] == 1
    assert player["web_name"] == "Raya"
    assert player["now_cost"] == 60
    assert "total_points" not in player, "identity/status table must not carry ambiguous cumulative stats"


def test_parse_bootstrap_stats_row_carries_period_ambiguity_flag():
    tables = silver.parse_bootstrap(SAMPLE_BOOTSTRAP, SAMPLE_META, "snap1.json")
    stats = tables["player_stats"][0]
    assert stats["total_points"] == 162
    assert stats["stat_period_note"] == schema.PLAYER_STATS_PERIOD_NOTE


def test_parse_fixtures_row_shape():
    rows = silver.parse_fixtures(SAMPLE_FIXTURES, SAMPLE_META, "fx1.json")
    assert len(rows) == 1
    assert rows[0]["fixture_id"] == 1
    assert rows[0]["event_id"] == 1
    assert rows[0]["retrieved_at"] == SAMPLE_META["retrieved_at"]


def test_every_row_across_all_tables_has_retrieved_at(monkeypatch):
    """Point-in-time correctness collapses to one invariant: every emitted
    row must carry a retrieved_at. If this ever fails, some downstream
    Gold-layer feature could be built without a known 'as of' timestamp."""
    tables = silver.parse_bootstrap(SAMPLE_BOOTSTRAP, SAMPLE_META, "snap1.json")
    for name, rows in tables.items():
        for row in rows:
            assert row.get("retrieved_at"), f"{name} row missing retrieved_at: {row}"
    for row in silver.parse_fixtures(SAMPLE_FIXTURES, SAMPLE_META, "fx1.json"):
        assert row.get("retrieved_at")
