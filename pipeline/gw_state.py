"""Gameweek phase state machine (Phase 13, Stage 2).

Two independent primitives instead of one global "current phase" — see
the Stage-1/2 design conversation this was built from for why a single
phase value is wrong: GW1 can be IN_PROGRESS (finished, awaiting
data_checked) at the exact same moment GW2 is PRE_DEADLINE. predict.py
and score.py ask different questions of the same underlying data and
must not share one answer:

- `next_prediction_gameweek()` — which gameweek should predict.py be
  targeting right now (the earliest with a future deadline)?
- `gameweek_phase()` — what phase is a SPECIFIC gameweek in (asked once
  per gameweek that has open business — a fresh prediction to make, or a
  prediction awaiting a result)?

Phase is derived only from hard facts already on the bootstrap-static
payload (deadline_time, finished, data_checked) — never from a
hardcoded calendar, and `is_current`/`is_next` are cross-checked as a
sanity signal (logged on disagreement) rather than trusted as the
source of truth, since they're FPL's own bookkeeping and this project
doesn't have any evidence they're never stale for an edge-case moment
right at a deadline.

SETTLED specifically requires BOTH `finished` and `data_checked` — the
gap between a gameweek's last kickoff finishing and its bonus points
being confirmed is real and explicitly not conflated with "done."
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Phase(str, Enum):
    PRE_DEADLINE = "PRE_DEADLINE"
    IN_PROGRESS = "IN_PROGRESS"
    SETTLED = "SETTLED"


@dataclass(frozen=True)
class GameweekPhaseInfo:
    gameweek: int
    phase: Phase
    deadline_utc: datetime
    hours_until_deadline: float  # negative once the deadline has passed
    finished: bool
    data_checked: bool
    teams_without_fixture: frozenset[str]
    teams_with_double_fixture: frozenset[str]


def _parse_deadline(deadline_time: str) -> datetime:
    return datetime.strptime(deadline_time, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _event_by_id(bootstrap_static: dict[str, Any], gameweek: int) -> dict[str, Any]:
    for e in bootstrap_static["events"]:
        if e["id"] == gameweek:
            return e
    raise ValueError(f"gameweek {gameweek} not found in bootstrap-static events")


def next_prediction_gameweek(bootstrap_static: dict[str, Any], now: datetime) -> int | None:
    """The earliest gameweek whose deadline is still in the future, or
    None if the season has ended (every deadline has passed). This is
    intentionally blind to whether any EARLIER gameweek is still waiting
    on data_checked — that's a separate, independent fact score.py asks
    about via gameweek_phase(), not this function's concern."""
    upcoming = [e for e in bootstrap_static["events"] if _parse_deadline(e["deadline_time"]) > now]
    if not upcoming:
        return None
    return min(upcoming, key=lambda e: _parse_deadline(e["deadline_time"]))["id"]


def gameweek_phase(bootstrap_static: dict[str, Any], fixtures: list[dict[str, Any]], gameweek: int, now: datetime) -> GameweekPhaseInfo:
    event = _event_by_id(bootstrap_static, gameweek)
    deadline_utc = _parse_deadline(event["deadline_time"])
    hours_until_deadline = (deadline_utc - now).total_seconds() / 3600.0
    finished = bool(event["finished"])
    data_checked = bool(event["data_checked"])

    if now < deadline_utc:
        phase = Phase.PRE_DEADLINE
    elif finished and data_checked:
        phase = Phase.SETTLED
    else:
        phase = Phase.IN_PROGRESS

    all_teams = {t["name"] for t in bootstrap_static["teams"]}
    gw_fixtures = [f for f in fixtures if f.get("event") == gameweek]
    team_fixture_counts: dict[str, int] = {}
    team_by_id = {t["id"]: t["name"] for t in bootstrap_static["teams"]}
    for f in gw_fixtures:
        for side_id in (f["team_h"], f["team_a"]):
            name = team_by_id.get(side_id)
            if name is not None:
                team_fixture_counts[name] = team_fixture_counts.get(name, 0) + 1

    teams_without_fixture = frozenset(all_teams - team_fixture_counts.keys())
    teams_with_double_fixture = frozenset(team for team, count in team_fixture_counts.items() if count > 1)

    _cross_check_fpl_flags(event, phase, gameweek)

    return GameweekPhaseInfo(
        gameweek=gameweek, phase=phase, deadline_utc=deadline_utc, hours_until_deadline=hours_until_deadline,
        finished=finished, data_checked=data_checked,
        teams_without_fixture=teams_without_fixture, teams_with_double_fixture=teams_with_double_fixture,
    )


def _cross_check_fpl_flags(event: dict[str, Any], derived_phase: Phase, gameweek: int) -> None:
    """FPL's own is_next/is_current flags are a sanity signal, not the
    source of truth (see module docstring) — log, don't raise, on
    disagreement, since this project has no evidence they're ever
    actually wrong, only that trusting an undocumented API's bookkeeping
    over hard facts (deadline/finished/data_checked) would be the wrong
    default."""
    if derived_phase == Phase.PRE_DEADLINE and not event.get("is_next") and not event.get("is_current"):
        logger.warning("gameweek %s derived as PRE_DEADLINE but FPL flags neither is_next nor is_current", gameweek)
    if derived_phase in (Phase.IN_PROGRESS, Phase.SETTLED) and event.get("is_next"):
        logger.warning("gameweek %s derived as %s but FPL still flags is_next=True", gameweek, derived_phase.value)
