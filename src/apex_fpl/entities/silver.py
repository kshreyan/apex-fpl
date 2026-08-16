"""Silver-layer canonical entity builder.

Reads Bronze snapshots (immutable raw API captures) and appends canonical,
typed rows to data/canonical/*.csv. Idempotent: each Bronze snapshot is
processed at most once, tracked by its payload hash in
data/canonical/_silver_build_state.json, so re-running `run_build()` after
new snapshots have arrived only appends the new rows — it never duplicates
or rewrites what's already there.

This module intentionally does not use pandas/DuckDB/Parquet: the data
volumes here (hundreds of players, tens of teams/gameweeks, hundreds of
fixtures) don't yet justify the dependency, per the spec's instruction not
to introduce technology merely because it's on an allowed list. Revisit
once Gold-layer feature engineering needs real dataframe operations.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from apex_fpl.entities import schema

REPO_ROOT = Path(__file__).resolve().parents[3]
BRONZE_ROOT = REPO_ROOT / "data" / "snapshots" / "bronze"
CANONICAL_ROOT = REPO_ROOT / "data" / "canonical"
STATE_PATH = CANONICAL_ROOT / "_silver_build_state.json"


def _load_state() -> dict[str, list[str]]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"processed_hashes": []}


def _save_state(state: dict[str, list[str]]) -> None:
    CANONICAL_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def _iter_snapshots(source: str, bronze_root: Path) -> list[Path]:
    src_dir = bronze_root / source
    if not src_dir.exists():
        return []
    return sorted(p for p in src_dir.glob("*.json") if not p.name.endswith(".meta.json"))


def _read_snapshot(payload_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(payload_path.read_text())
    meta = json.loads(payload_path.with_suffix(".meta.json").read_text())
    return payload, meta


def _append_csv(table_name: str, fields: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    CANONICAL_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = CANONICAL_ROOT / f"{table_name}.csv"
    is_new = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_bootstrap(payload: dict[str, Any], meta: dict[str, Any], snapshot_name: str) -> dict[str, list[dict[str, Any]]]:
    retrieved_at = meta["retrieved_at"]

    clubs = [
        {
            "club_id": t["id"], "code": t["code"], "name": t["name"], "short_name": t["short_name"],
            "strength_overall_home": t.get("strength_overall_home"),
            "strength_overall_away": t.get("strength_overall_away"),
            "retrieved_at": retrieved_at, "source_snapshot": snapshot_name,
        }
        for t in payload.get("teams", [])
    ]

    positions = [
        {
            "element_type_id": et["id"], "singular_name": et["singular_name"],
            "plural_name_short": et["plural_name_short"],
            "squad_select": et["squad_select"], "squad_min_play": et["squad_min_play"],
            "squad_max_play": et["squad_max_play"],
            "retrieved_at": retrieved_at, "source_snapshot": snapshot_name,
        }
        for et in payload.get("element_types", [])
    ]

    gameweeks = [
        {
            "event_id": e["id"], "name": e["name"], "deadline_time_utc": e["deadline_time"],
            "is_current": e.get("is_current"), "is_next": e.get("is_next"),
            "is_previous": e.get("is_previous"), "finished": e.get("finished"),
            "data_checked": e.get("data_checked"),
            "retrieved_at": retrieved_at, "source_snapshot": snapshot_name,
        }
        for e in payload.get("events", [])
    ]

    players = []
    player_stats = []
    for el in payload.get("elements", []):
        players.append({
            "player_id": el["id"], "code": el["code"], "web_name": el["web_name"],
            "first_name": el["first_name"], "second_name": el["second_name"],
            "team_id": el["team"], "element_type_id": el["element_type"], "status": el["status"],
            "news": el.get("news", ""), "news_added": el.get("news_added"),
            "chance_of_playing_this_round": el.get("chance_of_playing_this_round"),
            "chance_of_playing_next_round": el.get("chance_of_playing_next_round"),
            "now_cost": el["now_cost"], "selected_by_percent": el["selected_by_percent"],
            "retrieved_at": retrieved_at, "source_snapshot": snapshot_name,
        })
        player_stats.append({
            "player_id": el["id"], "event_points": el.get("event_points"),
            "total_points": el.get("total_points"), "minutes": el.get("minutes"),
            "goals_scored": el.get("goals_scored"), "assists": el.get("assists"),
            "clean_sheets": el.get("clean_sheets"), "goals_conceded": el.get("goals_conceded"),
            "bonus": el.get("bonus"), "bps": el.get("bps"), "saves": el.get("saves"),
            "defensive_contribution": el.get("defensive_contribution"),
            "expected_goals": el.get("expected_goals"), "expected_assists": el.get("expected_assists"),
            "form": el.get("form"), "points_per_game": el.get("points_per_game"),
            "stat_period_note": schema.PLAYER_STATS_PERIOD_NOTE,
            "retrieved_at": retrieved_at, "source_snapshot": snapshot_name,
        })

    return {
        "clubs": clubs, "positions": positions, "gameweeks": gameweeks,
        "players": players, "player_stats": player_stats,
    }


def parse_fixtures(payload: list[dict[str, Any]], meta: dict[str, Any], snapshot_name: str) -> list[dict[str, Any]]:
    retrieved_at = meta["retrieved_at"]
    return [
        {
            "fixture_id": fx["id"], "event_id": fx["event"], "team_h": fx["team_h"], "team_a": fx["team_a"],
            "kickoff_time_utc": fx.get("kickoff_time"), "finished": fx.get("finished"),
            "team_h_score": fx.get("team_h_score"), "team_a_score": fx.get("team_a_score"),
            "team_h_difficulty": fx.get("team_h_difficulty"), "team_a_difficulty": fx.get("team_a_difficulty"),
            "retrieved_at": retrieved_at, "source_snapshot": snapshot_name,
        }
        for fx in payload
    ]


def run_build(bronze_root: Path | None = None) -> dict[str, int]:
    """Process every not-yet-processed Bronze snapshot, appending Silver rows.
    Returns a count of rows appended per table.

    `bronze_root` defaults to this module's own BRONZE_ROOT (looked up
    fresh from the module namespace, not bound at def time, for the same
    monkeypatch-friendliness reason as bronze.py's snapshot_root — see
    that module's docstring). Pass it explicitly to build Silver from a
    different capture location, e.g. the live pipeline's per-gameweek
    data/raw/gw{n}/ directory instead of data/snapshots/bronze/.
    Deduplication is by payload content hash regardless of which root a
    snapshot came from, so pointing this at a different root on a later
    call is always safe -- it can only add rows Silver hasn't seen yet,
    never duplicate ones it has.
    """
    resolved_root = bronze_root if bronze_root is not None else BRONZE_ROOT
    state = _load_state()
    processed = set(state["processed_hashes"])
    counts: dict[str, int] = {}

    for payload_path in _iter_snapshots("bootstrap_static", resolved_root):
        payload, meta = _read_snapshot(payload_path)
        h = meta["raw_payload_hash"]
        if h in processed:
            continue
        tables = parse_bootstrap(payload, meta, payload_path.name)
        for name, fields in [
            ("clubs", schema.CLUBS_FIELDS), ("positions", schema.POSITIONS_FIELDS),
            ("gameweeks", schema.GAMEWEEKS_FIELDS), ("players", schema.PLAYERS_FIELDS),
            ("player_stats", schema.PLAYER_STATS_FIELDS),
        ]:
            _append_csv(name, fields, tables[name])
            counts[name] = counts.get(name, 0) + len(tables[name])
        processed.add(h)

    for payload_path in _iter_snapshots("fixtures", resolved_root):
        payload, meta = _read_snapshot(payload_path)
        h = meta["raw_payload_hash"]
        if h in processed:
            continue
        rows = parse_fixtures(payload, meta, payload_path.name)
        _append_csv("fixtures", schema.FIXTURES_FIELDS, rows)
        counts["fixtures"] = counts.get("fixtures", 0) + len(rows)
        processed.add(h)

    state["processed_hashes"] = sorted(processed)
    _save_state(state)
    return counts


if __name__ == "__main__":
    result = run_build()
    if not result:
        print("no new snapshots to process")
    for table, n in result.items():
        print(f"{table}: +{n} rows")
