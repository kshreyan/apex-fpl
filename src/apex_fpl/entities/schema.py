"""Canonical (Silver-layer) table schemas.

Every table here is APPEND-ONLY: each build run adds one row per entity per
processed Bronze snapshot, tagged with `retrieved_at` and `source_snapshot`.
Nothing is ever overwritten in place. This is what makes effective-dated
history possible (Part IV/V of the spec) — if a player changes club or a
gameweek's `is_current` flips, both the old and new observations remain on
disk, and "what did we know as of time T" is answerable by filtering rows
on `retrieved_at <= T`, not by trusting whatever the table currently shows.

Canonical IDs are the FPL API's own integer IDs (element id for players,
team id for clubs, event id for gameweeks, fixture id for fixtures). These
are stable within a season. Cross-season stability (e.g. a player's FPL
`code` surviving a team_id change) is NOT yet handled — flagged in
docs/fpl_gap_analysis.md as future work once a second season of data
exists to test against.

IMPORTANT — see PLAYER_STATS_FIELDS docstring: cumulative performance
stats observed in a pre-season snapshot are of AMBIGUOUS PERIOD and must
never be silently treated as 2026/27 in-season observations.
"""
from __future__ import annotations

CLUBS_FIELDS = [
    "club_id", "code", "name", "short_name",
    "strength_overall_home", "strength_overall_away",
    "retrieved_at", "source_snapshot",
]

POSITIONS_FIELDS = [
    "element_type_id", "singular_name", "plural_name_short",
    "squad_select", "squad_min_play", "squad_max_play",
    "retrieved_at", "source_snapshot",
]

GAMEWEEKS_FIELDS = [
    "event_id", "name", "deadline_time_utc",
    "is_current", "is_next", "is_previous", "finished", "data_checked",
    "retrieved_at", "source_snapshot",
]

# Identity/status fields: things that are true "as of this snapshot" and
# meaningfully change week to week (price, availability, ownership). Safe
# to treat as point-in-time observations.
PLAYERS_FIELDS = [
    "player_id", "code", "web_name", "first_name", "second_name",
    "team_id", "element_type_id", "status",
    "news", "news_added", "chance_of_playing_this_round", "chance_of_playing_next_round",
    "now_cost", "selected_by_percent",
    "retrieved_at", "source_snapshot",
]

# Cumulative/aggregate performance fields (total_points, minutes, bps,
# goals_scored, etc.). CONFIRMED FROM LIVE DATA on 2026-08-14 (pre-season,
# GW1 not yet played, event_points=0 for all players) that these fields are
# NON-ZERO and plausible-looking as full-season totals — almost certainly
# stale 2025/26 season-end values still populated by the API ahead of
# GW1, NOT 2026/27 season-to-date observations. This is not yet verified
# against a second source. Every row in this table carries an explicit
# stat_period_note so downstream code cannot silently treat these as
# current-season truth; the Gold-layer feature builder MUST NOT consume
# this table for 2026/27 "as of deadline" features until the ambiguity is
# resolved (see docs/fpl_gap_analysis.md unresolved_gaps).
PLAYER_STATS_FIELDS = [
    "player_id", "event_points", "total_points", "minutes",
    "goals_scored", "assists", "clean_sheets", "goals_conceded",
    "bonus", "bps", "saves",
    # DefCon (Defensive Contribution, live from 2025/26) raw stats --
    # defensive_contribution was already here; the other 4 were captured
    # in the live payload but not this table until Phase 13 Block 1.4.
    "defensive_contribution", "defensive_contribution_per_90",
    "clearances_blocks_interceptions", "recoveries", "tackles",
    "expected_goals", "expected_assists",
    "form", "points_per_game",
    "stat_period_note",
    "retrieved_at", "source_snapshot",
]
PLAYER_STATS_PERIOD_NOTE = (
    "AMBIGUOUS PERIOD: captured pre-season (event_points=0, no gameweek "
    "finished). Likely stale 2025/26 season-end totals, not verified. "
    "Do not treat as 2026/27 in-season observations without confirmation."
)

FIXTURES_FIELDS = [
    "fixture_id", "event_id", "team_h", "team_a", "kickoff_time_utc",
    "finished", "team_h_score", "team_a_score",
    "team_h_difficulty", "team_a_difficulty",
    "retrieved_at", "source_snapshot",
]
