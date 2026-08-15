"""Attacking-allocation challengers (spec Part X / XXXVI mandatory
baselines) for the Phase 4 model tournament, competing against
apex_fpl.models.attacking.proportional (the raw-share champion from
Phase 2/3, which spec Part X explicitly names as something to move past).

- equal_split: every player on the team gets an identical share — the
  "no information" mandatory baseline the champion must be shown to beat.
- shrinkage_share: the champion's known weakness is small-sample noise (a
  player who scored a fluke goal in a 3-game lookback window gets a wildly
  overstated share). This blends the observed share toward a uniform prior
  with a tunable pseudo-count `alpha` (Dirichlet/empirical-Bayes shrinkage)
  — a genuinely different mechanism, not a hyperparameter of the champion.
"""
from __future__ import annotations

from apex_fpl.models.attacking.proportional import AttackingShare


def equal_split(team_player_ids: list[str]) -> dict[str, AttackingShare]:
    n = len(team_player_ids) or 1
    share = 1.0 / n
    return {pid: AttackingShare(share, share) for pid in team_player_ids}


def shrinkage_share(team_player_history: dict[str, list[tuple[int, int]]], alpha: float = 3.0) -> dict[str, AttackingShare]:
    """team_player_history: player_id -> [(goals, assists), ...] over the
    lookback window (same input shape as proportional.compute_shares).
    alpha: pseudo-count strength of the uniform prior. alpha=0 reduces to
    the champion's raw proportional share; alpha -> infinity approaches
    equal_split."""
    team_goals = sum(g for hist in team_player_history.values() for g, _ in hist)
    team_assists = sum(a for hist in team_player_history.values() for _, a in hist)
    n_players = len(team_player_history) or 1
    prior = 1.0 / n_players

    shares = {}
    for pid, hist in team_player_history.items():
        pg = sum(g for g, _ in hist)
        pa = sum(a for _, a in hist)
        goal_share = (pg + alpha * prior) / (team_goals + alpha)
        assist_share = (pa + alpha * prior) / (team_assists + alpha)
        shares[pid] = AttackingShare(goal_share, assist_share)
    return shares
