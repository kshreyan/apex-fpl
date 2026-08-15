"""Loaders for the audited Vaastav historical archive (spec Part XXXIII).

Only used for historical replay/backtesting against past seasons — see
docs/vaastav_archive_audit.md for what is and isn't trustworthy in this
archive. Never used for 2026/27 live data (the Bronze/Silver pipeline in
src/apex_fpl/data and src/apex_fpl/entities is the sole source for that).
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from apex_fpl.models.teams.attack_defense import Fixture

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_ROOT = REPO_ROOT / "data" / "external" / "vaastav"


def _season_dir(season: str) -> Path:
    return EXTERNAL_ROOT / season


def load_team_id_to_name(season: str) -> dict[int, str]:
    path = _season_dir(season) / "teams.csv"
    with path.open() as f:
        return {int(r["id"]): r["name"] for r in csv.DictReader(f)}


def _parse_kickoff(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def load_fixtures(season: str) -> list[dict]:
    """Returns raw fixture dicts (not yet filtered by gameweek), each with
    event (int), date (datetime), home_team/away_team (names), and
    home_score/away_score (int or None if not yet played)."""
    teams = load_team_id_to_name(season)
    path = _season_dir(season) / "fixtures.csv"
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            finished = r["finished"] == "True"
            rows.append({
                "event": int(r["event"]) if r["event"] else None,
                "date": _parse_kickoff(r["kickoff_time"]),
                "home_team": teams[int(r["team_h"])],
                "away_team": teams[int(r["team_a"])],
                "home_score": int(r["team_h_score"]) if finished and r["team_h_score"] else None,
                "away_score": int(r["team_a_score"]) if finished and r["team_a_score"] else None,
            })
    return sorted(rows, key=lambda r: r["date"])


def fixtures_before_gw(season: str, target_gw: int) -> list[Fixture]:
    """Team-model training fixtures: strictly before target_gw, chronologically sorted."""
    raw = load_fixtures(season)
    return [
        Fixture(r["date"], r["home_team"], r["away_team"], r["home_score"], r["away_score"])
        for r in raw
        if r["event"] is not None and r["event"] < target_gw
    ]


def fixtures_at_gw(season: str, target_gw: int) -> list[dict]:
    raw = load_fixtures(season)
    return [r for r in raw if r["event"] == target_gw]


def season_gameweeks(season: str) -> list[int]:
    """The real gameweek numbers for a season, sorted — NOT assumed to be
    a contiguous 1-38 range. 2019-20 (COVID-19 disrupted) is numbered 1-29
    then 39-47; FPL never reused gameweek numbers 30-38 after the restart.
    Discovering this from the actual fixture data, rather than assuming
    range(1, 39), is what caught that gap in the first place — callers
    that iterate a hardcoded range silently skip real gameweeks for this
    season."""
    return sorted({r["event"] for r in load_fixtures(season) if r["event"] is not None})


def load_merged_gw(season: str) -> list[dict]:
    path = _season_dir(season) / "gws" / "merged_gw.csv" if (_season_dir(season) / "gws").exists() \
        else _season_dir(season) / "merged_gw.csv"
    with path.open() as f:
        return list(csv.DictReader(f))
