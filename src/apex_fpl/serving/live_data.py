"""Live 2026/27 data adapter (Phase 12) — bridges the Bronze/Silver
canonical tables (`apex_fpl.data.bronze`, `apex_fpl.entities.silver`,
already running daily since 2026-08-14 via the launchd job captured in
this environment's LaunchAgents) to the SAME data structures every
research phase's validated models already expect
(`apex_fpl.models.teams.attack_defense.Fixture`,
`apex_fpl.simulation.monte_carlo.PlayerInput`/`FixtureInput`) — no model
is rebuilt or reimplemented here, only the live-data plumbing that was
never built during the research phases (which all ran against the
historical Vaastav archive's DIFFERENT schema instead).

Two real, honestly-flagged integration facts discovered building this:

1. **The canonical CSVs are append-only logs across daily snapshots**,
   not single-row-per-entity tables — every loader here groups by
   primary key and keeps only the row with the latest `retrieved_at`,
   or callers would silently see duplicate/stale rows.
2. **Live `element_type_id` maps to "GKP", not "GK"** in bootstrap-
   static's own naming (`positions.csv`'s `plural_name_short`), but
   every optimizer/simulator in this project (`squad.SQUAD_QUOTAS`,
   `STARTING_XI_MIN/MAX`, etc.) is keyed on the historical Vaastav
   archive's convention: "GK", not "GKP". `POSITION_BY_ELEMENT_TYPE`
   maps element_type_id directly to the "GK" convention this project
   already uses everywhere, rather than trusting either archive's
   position STRING to already agree with the other's.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from apex_fpl.models.teams.attack_defense import Fixture

REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_DIR = REPO_ROOT / "data" / "canonical"

POSITION_BY_ELEMENT_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def _read_csv(name: str) -> list[dict]:
    path = CANONICAL_DIR / name
    with path.open() as f:
        return list(csv.DictReader(f))


def _latest_by_key(rows: list[dict], key_field: str) -> dict[str, dict]:
    """Keeps only the most-recently-retrieved row per primary key — the
    canonical CSVs are append-only across daily snapshots (see module
    docstring)."""
    latest: dict[str, dict] = {}
    for r in rows:
        k = r[key_field]
        if k not in latest or r["retrieved_at"] > latest[k]["retrieved_at"]:
            latest[k] = r
    return latest


def load_clubs() -> dict[int, str]:
    rows = _latest_by_key(_read_csv("clubs.csv"), "club_id")
    return {int(r["club_id"]): r["name"] for r in rows.values()}


def load_players() -> dict[str, dict]:
    """player_id -> {name, position, team, price} using the LATEST
    snapshot's players.csv row (team/position/price can legitimately
    change week to week — price via transfers, position/team far more
    rarely, e.g. a very rare mid-season reclassification)."""
    clubs = load_clubs()
    players = _latest_by_key(_read_csv("players.csv"), "player_id")
    out = {}
    for pid, r in players.items():
        out[pid] = {
            "name": r["web_name"],
            "position": POSITION_BY_ELEMENT_TYPE[int(r["element_type_id"])],
            "team": clubs[int(r["team_id"])],
            "price": int(r["now_cost"]) / 10.0,
            "status": r["status"],  # "a"=available, "i"=injured, "s"=suspended, etc. — bootstrap-static's own flag
        }
    return out


def load_gameweeks() -> dict[int, dict]:
    """event_id -> {name, deadline_time (datetime), finished, is_next, is_current}."""
    rows = _latest_by_key(_read_csv("gameweeks.csv"), "event_id")
    out = {}
    for eid, r in rows.items():
        out[int(eid)] = {
            "name": r["name"],
            "deadline_time": datetime.strptime(r["deadline_time_utc"], "%Y-%m-%dT%H:%M:%SZ"),
            "finished": r["finished"] == "True",
            "is_next": r["is_next"] == "True",
            "is_current": r["is_current"] == "True",
        }
    return out


def load_finished_fixtures() -> list[Fixture]:
    """Real 2026/27 fixtures with a final score, converted to the SAME
    Fixture dataclass every team-model fit() call already expects.
    Expected to be EMPTY before GW1 finishes — that's the real situation
    this module exists to handle (see build_team_model_fixtures)."""
    clubs = load_clubs()
    rows = _read_csv("fixtures.csv")
    out = []
    for r in rows:
        if r["finished"] != "True":
            continue
        out.append(Fixture(
            date=datetime.strptime(r["kickoff_time_utc"], "%Y-%m-%dT%H:%M:%SZ"),
            home_team=clubs[int(r["team_h"])], away_team=clubs[int(r["team_a"])],
            home_score=int(r["team_h_score"]), away_score=int(r["team_a_score"]),
        ))
    return sorted(out, key=lambda f: f.date)


def load_target_gw_fixtures(target_gw: int) -> list[dict]:
    clubs = load_clubs()
    rows = _read_csv("fixtures.csv")
    out = []
    for r in rows:
        if int(r["event_id"]) != target_gw:
            continue
        out.append({
            "home_team": clubs[int(r["team_h"])], "away_team": clubs[int(r["team_a"])],
            "date": datetime.strptime(r["kickoff_time_utc"], "%Y-%m-%dT%H:%M:%SZ"),
        })
    return out


def build_team_model_fixtures(fallback_seasons: tuple[str, ...] = ("2024-25",)) -> list[Fixture]:
    """Combines any REAL, already-finished 2026/27 fixtures with
    historical fixtures from `fallback_seasons`, so the existing
    time-decayed team model (halflife_days=380 by default — already
    designed to span well over a year, per
    apex_fpl.models.teams.attack_defense's own docstring) has SOMETHING
    to fit on even before GW1 of a new season has been played, rather
    than raising `fit() requires at least one completed fixture` outright.

    Known, stated limitation: the historical archive's most recent
    season is 2024-25 (ending ~May 2025) — 2025/26 is NOT YET in this
    project's historical archive, so the best available prior for a
    2026/27 GW1 recommendation is already well over a year old by the
    time it's used, meaningfully staler than the 380-day halflife was
    designed around. This is flagged here and in
    docs/phase12_production_system_report.md, not silently accepted as
    equivalent to a fresher prior."""
    from apex_fpl.backtesting import vaastav_loader as vl

    fixtures = list(load_finished_fixtures())
    for season in fallback_seasons:
        raw = vl.load_fixtures(season)
        for r in raw:
            if r["home_score"] is None:
                continue
            fixtures.append(Fixture(date=r["date"], home_team=r["home_team"], away_team=r["away_team"], home_score=r["home_score"], away_score=r["away_score"]))
    return sorted(fixtures, key=lambda f: f.date)
