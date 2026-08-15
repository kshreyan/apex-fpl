"""Idempotency and no-duplication guarantees for the Silver build.

If run_build() ever double-counts a snapshot, every downstream row-count
based sanity check (and worse, any model accidentally weighting duplicated
rows) silently corrupts. This is treated as a leakage-adjacent integrity
test, not an ordinary unit test.
"""
from __future__ import annotations

import json

from apex_fpl.entities import silver


def _write_bronze_snapshot(bronze_root, source, name, payload, retrieved_at):
    out_dir = bronze_root / source
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_path = out_dir / f"{name}.json"
    payload_path.write_text(json.dumps(payload))
    payload_bytes = payload_path.read_bytes()
    import hashlib
    meta = {
        "source": source, "retrieved_at": retrieved_at,
        "raw_payload_hash": hashlib.sha256(payload_bytes).hexdigest(),
    }
    payload_path.with_suffix(".meta.json").write_text(json.dumps(meta))
    return payload_path


MINI_BOOTSTRAP = {
    "teams": [{"id": 1, "code": 3, "name": "Arsenal", "short_name": "ARS",
               "strength_overall_home": 4, "strength_overall_away": 5}],
    "element_types": [{"id": 1, "singular_name": "Goalkeeper", "plural_name_short": "GKP",
                        "squad_select": 2, "squad_min_play": 1, "squad_max_play": 1}],
    "events": [{"id": 1, "name": "Gameweek 1", "deadline_time": "2026-08-21T17:30:00Z",
                "is_current": False, "is_next": True, "is_previous": False,
                "finished": False, "data_checked": False}],
    "elements": [{"id": 1, "code": 154561, "web_name": "Raya", "first_name": "David",
                  "second_name": "Raya Martin", "team": 1, "element_type": 1, "status": "a",
                  "news": "", "news_added": None, "chance_of_playing_this_round": None,
                  "chance_of_playing_next_round": None, "now_cost": 60, "selected_by_percent": "31.2",
                  "event_points": 0, "total_points": 162, "minutes": 3330, "goals_scored": 0,
                  "assists": 0, "clean_sheets": 19, "goals_conceded": 26, "bonus": 11, "bps": 633,
                  "saves": 60, "defensive_contribution": 0, "expected_goals": "0.00",
                  "expected_assists": "0.07", "form": "0.0", "points_per_game": "4.4"}],
}


def _setup(tmp_path, monkeypatch):
    bronze_root = tmp_path / "bronze"
    canonical_root = tmp_path / "canonical"
    monkeypatch.setattr(silver, "BRONZE_ROOT", bronze_root)
    monkeypatch.setattr(silver, "CANONICAL_ROOT", canonical_root)
    monkeypatch.setattr(silver, "STATE_PATH", canonical_root / "_silver_build_state.json")
    return bronze_root, canonical_root


def test_rerunning_build_on_same_snapshots_appends_nothing(tmp_path, monkeypatch):
    bronze_root, canonical_root = _setup(tmp_path, monkeypatch)
    _write_bronze_snapshot(bronze_root, "bootstrap_static", "20260814T000000Z",
                            MINI_BOOTSTRAP, "2026-08-14T00:00:00+00:00")

    first = silver.run_build()
    assert first["players"] == 1

    second = silver.run_build()
    assert second == {}, "re-running with no new snapshots must append zero rows"

    players_csv = (canonical_root / "players.csv").read_text()
    assert players_csv.count("\n") == 2  # header + 1 data row, not duplicated


def test_content_identical_snapshot_is_deduplicated_by_hash(tmp_path, monkeypatch):
    """Two snapshot files with byte-identical payloads (a legitimate outcome
    when structural data hasn't changed between captures) must be
    recognized as the same observation and not double-appended."""
    bronze_root, canonical_root = _setup(tmp_path, monkeypatch)
    _write_bronze_snapshot(bronze_root, "bootstrap_static", "20260814T000000Z",
                            MINI_BOOTSTRAP, "2026-08-14T00:00:00+00:00")
    _write_bronze_snapshot(bronze_root, "bootstrap_static", "20260814T010000Z",
                            MINI_BOOTSTRAP, "2026-08-14T01:00:00+00:00")

    result = silver.run_build()
    assert result["players"] == 1, "identical-content second snapshot must not create a duplicate row"


def test_genuinely_new_snapshot_after_first_build_appends_incrementally(tmp_path, monkeypatch):
    bronze_root, canonical_root = _setup(tmp_path, monkeypatch)
    _write_bronze_snapshot(bronze_root, "bootstrap_static", "20260814T000000Z",
                            MINI_BOOTSTRAP, "2026-08-14T00:00:00+00:00")
    silver.run_build()

    changed = json.loads(json.dumps(MINI_BOOTSTRAP))
    changed["elements"][0]["selected_by_percent"] = "31.5"
    _write_bronze_snapshot(bronze_root, "bootstrap_static", "20260814T020000Z",
                            changed, "2026-08-14T02:00:00+00:00")

    second = silver.run_build()
    assert second["players"] == 1, "a genuinely changed payload must append exactly one new row, not zero"

    players_csv = (canonical_root / "players.csv").read_text()
    assert players_csv.count("\n") == 3  # header + 2 distinct observations
