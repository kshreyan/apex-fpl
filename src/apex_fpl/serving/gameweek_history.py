"""Reconstructs real per-gameweek player history (minutes, goals,
assists) from this pipeline's OWN captured raw bootstrap-static
snapshots (data/raw/gw*/bootstrap_static/*.json), by diffing cumulative
season-to-date totals at consecutive settled-gameweek boundaries.

Why this exists: `apex_fpl.models.minutes.challengers.exponential_decay`
(the Phase 4b champion minutes model) and
`apex_fpl.models.attacking.challengers.shrinkage_share` (the champion
attacking-allocation model) both need real PER-GAMEWEEK history to do
anything other than degrade to their own uninformative-prior case --
`scripts/run_production_recommendation.py` had no source for that at
all. Only cumulative season-to-date totals are captured anywhere else
in this pipeline (`data/canonical/player_stats.csv`), and the FPL API's
per-gameweek `element-summary` endpoint has never been integrated. This
derives it from data already captured daily for a different purpose
(the automated pipeline's own raw-capture ledger), rather than adding
new API load.

No network calls here. Only reads what's already committed under
data/raw/. Cannot produce data for a gameweek whose settlement snapshot
was never captured (a pipeline outage, or a gameweek that happened
before this pipeline started running) -- skips it rather than guessing,
so a caller's resulting history can have real gaps. This directly
bounds when the in-season model transition
(scripts/run_production_recommendation.py) can actually fire: it
requires real reconstructed history to exist, not just a high
gameweek number.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_ROOT = REPO_ROOT / "data" / "raw"

CUMULATIVE_FIELDS = ("minutes", "goals_scored", "assists")


def _iter_bootstrap_snapshots(raw_root: Path):
    paths = sorted(raw_root.glob("gw*/bootstrap_static/*.json"))
    for p in paths:
        if p.name.endswith(".meta.json"):
            continue
        yield json.loads(p.read_text())


def _highest_settled_gw(events: list[dict]) -> int:
    settled = [e["id"] for e in events if e.get("finished") and e.get("data_checked")]
    return max(settled) if settled else 0


def find_settlement_snapshots(max_gw: int | None = None, raw_root: Path = RAW_DATA_ROOT) -> dict[int, dict]:
    """Returns {gw_number: parsed_bootstrap_json} for each gameweek N
    where a captured snapshot exists whose highest finished+data_checked
    gameweek is EXACTLY N -- i.e. captured after N settled but before
    N+1 did, so its cumulative totals represent "through gameweek N"
    exactly, not N plus part of N+1. Keeps the EARLIEST such snapshot
    per gameweek (closest to right after settlement -- a data_checked
    revision landing later would go unnoticed here, a known limitation
    this shares with pipeline/score.py's own event_points reliance, not
    solved here either)."""
    result: dict[int, dict] = {}
    for payload in _iter_bootstrap_snapshots(raw_root):
        gw = _highest_settled_gw(payload.get("events", []))
        if gw == 0 or (max_gw is not None and gw > max_gw):
            continue
        if gw not in result:
            result[gw] = payload
    return result


def reconstruct_player_gameweek_deltas(max_gw: int | None = None, raw_root: Path = RAW_DATA_ROOT) -> dict[str, dict[int, dict]]:
    """Returns {player_code: {gw_number: {"minutes": int, "goals_scored":
    int, "assists": int}}}. A gameweek's delta is only computed when its
    own settlement snapshot AND the immediately preceding gameweek's
    settlement snapshot are both available (gw=1's "preceding" is the
    implicit all-zero season start) -- if gameweek N's snapshot exists
    but N-1's doesn't, N is skipped rather than producing a delta that
    silently folds N-1's contribution into N. Negative deltas (a rare
    post-data_checked correction, e.g. a bonus-points revision touching
    a cumulative field) are clamped to 0 rather than passed through --
    downstream models expect non-negative per-gameweek counts."""
    snapshots = find_settlement_snapshots(max_gw, raw_root)

    def cumulative(gw: int) -> dict[str, dict]:
        return {str(el["code"]): {f: el[f] for f in CUMULATIVE_FIELDS} for el in snapshots[gw]["elements"]}

    deltas: dict[str, dict[int, dict]] = {}
    for gw in sorted(snapshots):
        prev_gw = gw - 1
        if prev_gw != 0 and prev_gw not in snapshots:
            continue  # can't compute a clean single-gameweek delta -- skip, don't guess
        baseline = cumulative(prev_gw) if prev_gw in snapshots else {}
        current = cumulative(gw)
        zero = {f: 0 for f in CUMULATIVE_FIELDS}
        for code, cur in current.items():
            prev = baseline.get(code, zero)
            deltas.setdefault(code, {})[gw] = {f: max(0, cur[f] - prev[f]) for f in CUMULATIVE_FIELDS}
    return deltas


def minutes_history_by_code(deltas: dict[str, dict[int, dict]]) -> dict[str, list[int]]:
    """Chronological (oldest first) per-player minutes list, the shape
    `apex_fpl.models.minutes.challengers.exponential_decay` expects."""
    return {code: [gw_deltas[gw]["minutes"] for gw in sorted(gw_deltas)] for code, gw_deltas in deltas.items()}


def goals_assists_history_by_code(deltas: dict[str, dict[int, dict]]) -> dict[str, list[tuple[int, int]]]:
    """Chronological (oldest first) per-player [(goals, assists), ...],
    the shape `apex_fpl.models.attacking.challengers.shrinkage_share`
    expects (as the values of its team_player_history dict)."""
    return {
        code: [(gw_deltas[gw]["goals_scored"], gw_deltas[gw]["assists"]) for gw in sorted(gw_deltas)]
        for code, gw_deltas in deltas.items()
    }
