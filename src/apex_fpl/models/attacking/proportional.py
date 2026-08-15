"""Proportional player-event allocation baseline (spec Part X).

Simplest honest baseline: allocate a team's expected goals across its
players in proportion to each player's share of the team's actual goals
(and, separately, assists) over a recent trailing history window.

This is explicitly NOT the target architecture — Part X calls this exact
approach out by name as something to move beyond ("Do not allocate team
expected goals using simplistic proportional averages"). It is used here
only because it is the correct Phase 2 starting point: cheap, transparent,
and easy to falsify once real challenger models exist to compare against
(Phase 4 model tournaments).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttackingShare:
    goal_share: float  # this player's share of the TEAM's goals over the lookback window
    assist_share: float  # this player's share of the TEAM's assists over the lookback window


def compute_shares(team_player_history: dict[str, list[tuple[int, int]]]) -> dict[str, AttackingShare]:
    """team_player_history: player_id -> list of (goals_scored, assists)
    tuples, one per gameweek in the lookback window, for ONE team's
    players only, strictly before the target deadline."""
    team_goals = sum(g for hist in team_player_history.values() for g, _ in hist)
    team_assists = sum(a for hist in team_player_history.values() for _, a in hist)
    n_players = len(team_player_history) or 1

    shares = {}
    for pid, hist in team_player_history.items():
        pg = sum(g for g, _ in hist)
        pa = sum(a for _, a in hist)
        goal_share = (pg / team_goals) if team_goals > 0 else (1.0 / n_players)
        assist_share = (pa / team_assists) if team_assists > 0 else (1.0 / n_players)
        shares[pid] = AttackingShare(goal_share, assist_share)
    return shares


def allocate(team_expected_goals: float, shares: dict[str, AttackingShare]) -> dict[str, tuple[float, float]]:
    """Returns player_id -> (expected_goals, expected_assists) for the
    upcoming fixture, given the team model's expected goals for that
    match. Assists are allocated off the same team_expected_goals figure
    (an assist requires a teammate's goal) rather than a separately
    modelled team assist total — a simplification, not a claim of
    precision."""
    return {
        pid: (team_expected_goals * s.goal_share, team_expected_goals * s.assist_share)
        for pid, s in shares.items()
    }
