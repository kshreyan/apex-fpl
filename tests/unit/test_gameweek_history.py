"""Unit tests for apex_fpl.serving.gameweek_history's per-gameweek delta
reconstruction, using small synthetic raw-snapshot fixtures written
directly to a tmp_path (no real captured data, no network)."""
from __future__ import annotations

import json

from apex_fpl.serving import gameweek_history as gh


def _write_snapshot(raw_root, gw_dir: str, filename: str, events: list[dict], elements: list[dict]) -> None:
    d = raw_root / gw_dir / "bootstrap_static"
    d.mkdir(parents=True, exist_ok=True)
    (d / filename).write_text(json.dumps({"events": events, "elements": elements}))


def _events(highest_settled: int, total: int = 5) -> list[dict]:
    return [{"id": i, "finished": i <= highest_settled, "data_checked": i <= highest_settled} for i in range(1, total + 1)]


def _element(code: int, minutes: int, goals: int, assists: int) -> dict:
    return {"code": code, "minutes": minutes, "goals_scored": goals, "assists": assists}


def test_find_settlement_snapshots_picks_earliest_per_gameweek(tmp_path):
    _write_snapshot(tmp_path, "gw01", "a.json", _events(0), [_element(100, 0, 0, 0)])
    _write_snapshot(tmp_path, "gw02", "b.json", _events(1), [_element(100, 70, 1, 0)])
    _write_snapshot(tmp_path, "gw02", "c.json", _events(1), [_element(100, 70, 1, 0)])  # later, same gw

    result = gh.find_settlement_snapshots(raw_root=tmp_path)

    assert set(result) == {1}
    assert result[1]["elements"][0]["minutes"] == 70


def test_reconstruct_deltas_diffs_consecutive_settled_gameweeks(tmp_path):
    _write_snapshot(tmp_path, "gw01", "a.json", _events(1), [_element(100, 70, 1, 0)])
    _write_snapshot(tmp_path, "gw02", "b.json", _events(2), [_element(100, 135, 1, 1)])

    deltas = gh.reconstruct_player_gameweek_deltas(raw_root=tmp_path)

    assert deltas["100"][1] == {"minutes": 70, "goals_scored": 1, "assists": 0}
    assert deltas["100"][2] == {"minutes": 65, "goals_scored": 0, "assists": 1}


def test_reconstruct_deltas_skips_a_gameweek_whose_predecessor_is_missing(tmp_path):
    """GW1's snapshot is missing (pipeline outage); GW2 and GW3 exist.
    GW2 can't be cleanly attributed (would fold GW1+GW2 together) so
    it's skipped; GW3 IS computable relative to GW2's own snapshot."""
    _write_snapshot(tmp_path, "gw02", "b.json", _events(2), [_element(100, 135, 1, 1)])
    _write_snapshot(tmp_path, "gw03", "c.json", _events(3), [_element(100, 200, 2, 1)])

    deltas = gh.reconstruct_player_gameweek_deltas(raw_root=tmp_path)

    assert 2 not in deltas.get("100", {})
    assert deltas["100"][3] == {"minutes": 65, "goals_scored": 1, "assists": 0}


def test_reconstruct_deltas_clamps_negative_delta_to_zero(tmp_path):
    """A rare post-data_checked downward correction must not corrupt a
    downstream model with a negative minutes/goals count."""
    _write_snapshot(tmp_path, "gw01", "a.json", _events(1), [_element(100, 90, 2, 0)])
    _write_snapshot(tmp_path, "gw02", "b.json", _events(2), [_element(100, 85, 2, 0)])  # revised down

    deltas = gh.reconstruct_player_gameweek_deltas(raw_root=tmp_path)

    assert deltas["100"][2]["minutes"] == 0


def test_reconstruct_deltas_respects_max_gw(tmp_path):
    _write_snapshot(tmp_path, "gw01", "a.json", _events(1), [_element(100, 90, 0, 0)])
    _write_snapshot(tmp_path, "gw02", "b.json", _events(2), [_element(100, 180, 0, 0)])

    deltas = gh.reconstruct_player_gameweek_deltas(max_gw=1, raw_root=tmp_path)

    assert set(deltas["100"]) == {1}


def test_minutes_history_by_code_is_chronological():
    deltas = {"100": {2: {"minutes": 65, "goals_scored": 0, "assists": 1}, 1: {"minutes": 70, "goals_scored": 1, "assists": 0}}}
    assert gh.minutes_history_by_code(deltas) == {"100": [70, 65]}


def test_goals_assists_history_by_code_is_chronological():
    deltas = {"100": {2: {"minutes": 65, "goals_scored": 0, "assists": 1}, 1: {"minutes": 70, "goals_scored": 1, "assists": 0}}}
    assert gh.goals_assists_history_by_code(deltas) == {"100": [(1, 0), (0, 1)]}


def test_empty_raw_root_produces_no_deltas(tmp_path):
    assert gh.reconstruct_player_gameweek_deltas(raw_root=tmp_path) == {}
