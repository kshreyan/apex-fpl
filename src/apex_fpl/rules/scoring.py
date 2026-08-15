"""Deterministic FPL scoring engine (spec Part XV).

Converts a player's gameweek match events into FPL points using the
season's official rules from configs/seasons/<season>.yaml. Scoring rules
are exactly known — this module must never learn or approximate them, only
apply them deterministically.

Phase 2 baseline scope: appearance, goals, assists, clean sheets, goals
conceded, saves, penalties, cards, own goals — the core rules that have
been stable across the seasons used for validation (2016/17-2026/27).
Defensive contributions and BPS-derived bonus are NOT computed here; both
require event-level match data (CBIT/CBIRT action counts, full BPS
weights) this project doesn't have a source for yet (see
docs/fpl_gap_analysis.md). When reconstructing historical actuals for
validation, `bonus` is taken as a given input (the archive's recorded
value) rather than derived — simulating it from a joint BPS model is
Phase 6 work (spec Part XIV), not Phase 2.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]

POSITION_MAP = {"GK": "GK", "GKP": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}


def load_scoring_rules(season: str = "2026_27") -> dict:
    path = REPO_ROOT / "configs" / "seasons" / f"{season}.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg["scoring"]


@dataclass(frozen=True)
class PlayerMatchEvents:
    position: str  # "GK" | "GKP" | "DEF" | "MID" | "FWD"
    minutes: int
    goals_scored: int = 0
    assists: int = 0
    clean_sheet: bool = False
    goals_conceded: int = 0
    own_goals: int = 0
    penalties_saved: int = 0
    penalties_missed: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    saves: int = 0
    bonus: int = 0  # given directly; not derived from BPS in this baseline
    defensive_contribution_points: int = 0  # not computed in this baseline


def score_player_gameweek(events: PlayerMatchEvents, rules: dict) -> int:
    pos = POSITION_MAP[events.position]
    points = 0

    if events.minutes >= 60:
        points += rules["appearance"]["at_least_60_min"]
    elif events.minutes > 0:
        points += rules["appearance"]["under_60_min"]

    points += events.goals_scored * rules["goals_scored"][pos]
    points += events.assists * rules["assist"]

    if events.clean_sheet and events.minutes >= rules["clean_sheet_min_minutes"]:
        points += rules["clean_sheet"][pos]

    gc_rate = rules["goals_conceded"][pos]
    if gc_rate != 0:
        points += (events.goals_conceded // rules["goals_conceded_divisor"]) * gc_rate

    if pos == "GK" and events.saves:
        points += (events.saves // rules["saves_divisor"]) * rules["saves"]

    points += events.penalties_saved * rules["penalty_save"]
    points += events.penalties_missed * rules["penalty_miss"]
    points += events.yellow_cards * rules["yellow_card"]
    points += events.red_cards * rules["red_card"]
    points += events.own_goals * rules["own_goal"]

    points += events.bonus
    points += events.defensive_contribution_points

    return points
